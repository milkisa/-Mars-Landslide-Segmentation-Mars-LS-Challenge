import optuna
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time, os
import os
import sys
os.environ["HOME"] = "/home/milkisayebasse/Mars/.cache"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.losses import bce_dice_loss, bce_tversky_loss, focal_dice_loss,focal_loss,dice_loss
from src.metrics import dice_iou_from_logits, segmentation_metrics_from_logits
# import your project stuff exactly as in train.py
from src.utils import set_seed, ensure_dir, load_json, AverageMeter
from src.datasets import MarsLSDataset
from src.losses import bce_dice_loss
from src.models.twoStreamSMP import TwoStreamSegformer

# --- paste validate_with_thrsearch here ---
# (from the section above)
def get_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def train_one_epoch(model, loader, device, optimizer, scaler):
    model.train()
    loss_meter = AverageMeter()

    for x, y, filename in loader:
        optimizer.zero_grad(set_to_none=True)

        if isinstance(x, (tuple, list)):
            xA, xB, XC = x
            xA = xA.to(device, non_blocking=True)
            xB = xB.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)

            with autocast():
                logits = model(xA, xB)
                loss = bce_dice_loss(logits, y)
            bs = xA.size(0)
        else:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast():
                logits = model(x)
                loss = bce_dice_loss(logits, y)
            bs = x.size(0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), bs)

    return loss_meter.avg


def objective(trial):
    # ---- tune threshold (optional) ----
    thr = 0.5

    # ---- choose optimizer ----
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])

    # LR ranges differ by optimizer
    if optimizer_name == "SGD":
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 0.80, 0.99)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    elif optimizer_name == "Adam":
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        momentum = None
    else:  # AdamW
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        momentum = None

    # ---- choose scheduler ----
    scheduler_name = trial.suggest_categorical(
        "scheduler",
        ["None", "CosineAnnealing", "StepLR", "ExponentialLR"]
    )

    # ---- fixed config ----
    data_root  = "/mnt/data/mars_2"
    stats_json = "/mnt/data/mars_2/stats.json"
    batch_size = 32
    num_workers = 8
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    stats = load_json(stats_json) if os.path.exists(stats_json) else None

    train_ds = MarsLSDataset(
        img_dir=os.path.join(data_root, "train/images"),
        mask_dir=os.path.join(data_root, "train/masks"),
        return_streams=True,
        transforms=None,
        stats=stats,
        probe=False,
    )
    val_ds = MarsLSDataset(
        img_dir=os.path.join(data_root, "val/images"),
        mask_dir=os.path.join(data_root, "val/masks"),
        return_streams=True,
        transforms=None,
        stats=stats,
        probe=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # ---- model ----
    model = TwoStreamSegformer(classes=1, stem_out=32, inA=5, inB=2).to(device)

    # ---- training budget ----
    max_epochs = 50
    best_val = -1.0

    # ---- optimizer with param groups (decay / no_decay) ----
    param_groups = get_param_groups(model, weight_decay)

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr)

    scaler = GradScaler()

    # ---- scheduler (chosen by Optuna) ----
    scheduler = None
    if scheduler_name == "CosineAnnealing":
        eta_min = trial.suggest_float("eta_min", 1e-7, 1e-5, log=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=eta_min
        )

    elif scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 3, max(3, max_epochs // 2))
        gamma = trial.suggest_float("gamma", 0.1, 0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    elif scheduler_name == "ExponentialLR":
        exp_gamma = trial.suggest_float("exp_gamma", 0.85, 0.99)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=exp_gamma
        )

    # ---- loop ----
    for epoch in range(1, max_epochs + 1):
        _ = train_one_epoch(model, train_loader, device, optimizer, scaler)

        # val f1 at thr
        with torch.no_grad():
            model.eval()
            f1_meter = AverageMeter()
            for x, y, filename in val_loader:
                xA, xB, XC = x
                xA = xA.to(device, non_blocking=True)
                xB = xB.to(device, non_blocking=True)
                y  = y.to(device, non_blocking=True)

                logits = model(xA, xB)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                seg = segmentation_metrics_from_logits(
                    logits, y, thr=thr, epoch=epoch, filename=filename
                )
                f1_meter.update(seg["f1"], xA.size(0))

            val_f1 = f1_meter.avg

        best_val = max(best_val, val_f1)

        if scheduler is not None:
            scheduler.step()

        trial.report(val_f1, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val


def main():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=30)

    print("\nBest trial:")
    print("  value:", study.best_trial.value)

    p = study.best_trial.params
    print("  optimizer:", p.get("optimizer"))
    print("  lr:", p.get("lr"))
    print("  weight_decay:", p.get("weight_decay"))
    if p.get("optimizer") == "SGD":
        print("  momentum:", p.get("momentum"))

    print("  scheduler:", p.get("scheduler"))
    if p.get("scheduler") == "CosineAnnealing":
        print("  eta_min:", p.get("eta_min"))
    elif p.get("scheduler") == "StepLR":
        print("  step_size:", p.get("step_size"))
        print("  gamma:", p.get("gamma"))
    elif p.get("scheduler") == "ExponentialLR":
        print("  exp_gamma:", p.get("exp_gamma"))

    print("  thr (fixed):", 0.5)
    print("\n  full params:", p)

if __name__ == "__main__":
    main()