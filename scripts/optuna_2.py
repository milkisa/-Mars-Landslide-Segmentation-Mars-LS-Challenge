# scripts/optuna_encoder_then_opt_lr.py
# Phase 1: choose encoder (LR fixed, thr fixed)
# Phase 2: fix best encoder, tune optimizer + LR (+ wd/momentum) (thr fixed)

import os, sys, inspect
import optuna
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# -----------------------------
# IMPORTANT: cache & imports
# -----------------------------
# In docker, you should also set HOME/TORCH_HOME/XDG_CACHE_HOME via docker run env vars
# so pretrained weights download to a writable mounted folder.
os.environ["HOME"] = "/home/milkisayebasse/Mars/.cache"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed, load_json, AverageMeter
from src.datasets import MarsLSDataset
from src.losses import bce_dice_loss
from src.metrics import segmentation_metrics_from_logits
from src.models.twoStreamSMP import TwoStreamSegformer


# -----------------------------
# Config (edit if needed)
# -----------------------------
DATA_ROOT = "/mnt/data/mars"
STATS_JSON = "/mnt/data/mars/stats.json"
BATCH_SIZE = 32
NUM_WORKERS = 8
SEED = 42

THR_FIXED = 0.5       # as you requested
PHASE1_LR_FIXED = 1e-4  # LR used only in Phase 1 (encoder search)
PHASE1_EPOCHS = 20
PHASE1_TRIALS = 12       # for 3 encoders, 9-15 is fine

PHASE2_EPOCHS = 20
PHASE2_TRIALS = 40       # optimizer+lr search is bigger; 40-80 recommended


# -----------------------------
# Helpers
# -----------------------------
def make_loaders():
    stats = load_json(STATS_JSON) if os.path.exists(STATS_JSON) else None

    train_ds = MarsLSDataset(
        img_dir=os.path.join(DATA_ROOT, "train/images"),
        mask_dir=os.path.join(DATA_ROOT, "train/masks"),
        return_streams=True,
        transforms=None,  # you can plug your TorchSegTransform here
        stats=stats,
        probe=False,
    )
    val_ds = MarsLSDataset(
        img_dir=os.path.join(DATA_ROOT, "val/images"),
        mask_dir=os.path.join(DATA_ROOT, "val/masks"),
        return_streams=True,
        transforms=None,
        stats=stats,
        probe=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(device, encoder_name=None, encoder_weights="imagenet"):
    """
    Builds your TwoStreamSegformer, but only passes encoder_name/encoder_weights
    if TwoStreamSegformer.__init__ actually supports them.
    """
    kwargs = dict(classes=1, stem_out=32, inA=5, inB=2)

    sig = inspect.signature(TwoStreamSegformer.__init__)
    if encoder_name is not None and "encoder_name" in sig.parameters:
        kwargs["encoder_name"] = encoder_name
    if encoder_weights is not None and "encoder_weights" in sig.parameters:
        kwargs["encoder_weights"] = encoder_weights

    return TwoStreamSegformer(**kwargs).to(device)


def make_optimizer(trial, optimizer_name, model_params):
    """
    Returns (optimizer, lr) and suggests any extra optimizer params via Optuna.
    Note: LR ranges differ by optimizer.
    """
    if optimizer_name == "SGD":
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 0.80, 0.99)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        opt = torch.optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return opt, lr

    if optimizer_name == "Adam":
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        # Adam weight_decay often kept smaller than AdamW; tune narrow:
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        opt = torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        return opt, lr

    # default: AdamW
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    opt = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    return opt, lr


def train_one_epoch(model, loader, device, optimizer, scaler, grad_clip=1.0):
    model.train()
    loss_meter = AverageMeter()

    for x, y, filename in loader:
        optimizer.zero_grad(set_to_none=True)

        # your dataset returns (xA, xB, XC); model uses (xA, xB)
        xA, xB, XC = x
        xA = xA.to(device, non_blocking=True)
        xB = xB.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        with autocast():
            logits = model(xA, xB)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = bce_dice_loss(logits, y)

        scaler.scale(loss).backward()
        # stabilize larger LR trials
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), xA.size(0))

    return loss_meter.avg


@torch.no_grad()
def eval_val_f1(model, val_loader, device, thr, epoch=0):
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

        seg = segmentation_metrics_from_logits(logits, y, thr=thr, epoch=epoch, filename=filename)
        f1_meter.update(seg["f1"], xA.size(0))

    return f1_meter.avg


# -----------------------------
# Phase 1: encoder search (LR fixed, thr fixed)
# -----------------------------
def objective_phase1(trial):
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = make_loaders()

    encoder_name = trial.suggest_categorical(
        "encoder_name",
        ["resnet18", "resnet50", "resnext50_32x4d", "resnext101_32x4d"],
    )

    # If downloads are a problem, set encoder_weights=None
    model = build_model(device, encoder_name=encoder_name, encoder_weights="swsl")

    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE1_LR_FIXED, weight_decay=1e-2)
    scaler = GradScaler()

    best_val = -1.0
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_one_epoch(model, train_loader, device, optimizer, scaler)
        val_f1 = eval_val_f1(model, val_loader, device, thr=THR_FIXED, epoch=epoch)

        best_val = max(best_val, val_f1)
        trial.report(val_f1, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val


# -----------------------------
# Phase 2: optimizer + LR search (encoder fixed, thr fixed)
# -----------------------------
def make_objective_phase2(best_encoder: str):
    def objective_phase2(trial):
        set_seed(SEED)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader = make_loaders()

        optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])

        model = build_model(device, encoder_name=best_encoder, encoder_weights="swsl")
        optimizer, lr = make_optimizer(trial, optimizer_name, model.parameters())
        scaler = GradScaler()

        best_val = -1.0
        for epoch in range(1, PHASE2_EPOCHS + 1):
            train_one_epoch(model, train_loader, device, optimizer, scaler)
            val_f1 = eval_val_f1(model, val_loader, device, thr=THR_FIXED, epoch=epoch)

            best_val = max(best_val, val_f1)
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val

    return objective_phase2


def main():
    # -------- Phase 1 --------
    study1 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study1.optimize(objective_phase1, n_trials=PHASE1_TRIALS)

    best_encoder = study1.best_trial.params["encoder_name"]
    print("\n[PHASE 1] Best encoder:", best_encoder)
    print("[PHASE 1] Best value:", study1.best_value)

    # -------- Phase 2 --------
    study2 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study2.optimize(make_objective_phase2(best_encoder), n_trials=PHASE2_TRIALS)

    print("\n[PHASE 2] Best value:", study2.best_value)
    print("[PHASE 2] Best params:", {"encoder_name": best_encoder, **study2.best_trial.params})
    print("[NOTE] Threshold fixed at:", THR_FIXED)


if __name__ == "__main__":
    main()