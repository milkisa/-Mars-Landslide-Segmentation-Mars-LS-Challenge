"""
Train + validate + infer for Mars Landslide segmentation (two-stream SegFormer).

Changes vs your original:
- Removed duplicate imports and redundant blocks.
- Centralized dataloader + dataset creation.
- Cleaner validate() with correct forward for 2-stream vs 3-stream.
- Clear separation: setup -> train loop -> validate -> checkpoint -> inference.
- Well-commented and easier to maintain.

NOTE:
- Your MarsLSDataset returns (x, y, filename). When return_streams=True it returns x as a tuple.
- Your current model call is: model(xA, xB). (XC is currently unused for this model.)
"""

import os
import sys
import time
import argparse
from typing import Dict, Tuple, Union, List

# ---------------------------------------------------------------------
# Cache / environment (your original path)
# ---------------------------------------------------------------------
os.environ["HOME"] = "/home/milkisayebasse/Mars/.cache"

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm



# Project root so `src.*` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from validate_metrics import validate,run_test_val_inference
from inference_save import run_test_inference
from src.utils import set_seed, ensure_dir, load_json, AverageMeter
from src.datasets import MarsLSDataset
from src.losses import bce_dice_loss
from src.metrics import dice_iou_from_logits, segmentation_metrics_from_logits
from src.augmentation import TorchSegTransform

# Your model
from src.models.twoStreamSMP import TwoStreamSegformer


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------
def get_transforms(train: bool = True, debug: bool = False):
    """Return augmentation / preprocessing pipeline."""
    return TorchSegTransform(train=train, debug=debug)





# ---------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------
def build_weight_decay_param_groups(model: torch.nn.Module, weight_decay: float):
    """
    Split parameters into decay / no_decay groups:
    - no_decay: biases, norm params (BatchNorm/LayerNorm), and 1D params
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # Data / IO
    ap.add_argument("--data_root", type=str, default="/mnt/data/mars")
    ap.add_argument("--stats_json", type=str, default="/mnt/data/mars/stats.json")
    ap.add_argument("--out_dir", type=str, default="outputs/exp01")

    # Training
    ap.add_argument("--epochs", type=int, default=801)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # Misc
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision AMP")

    args = ap.parse_args()

    # Reproducibility + output dirs
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)

    # Load normalization stats if available
    stats = load_json(args.stats_json) if os.path.exists(args.stats_json) else None
    print("[DEBUG] stats_json:", args.stats_json, "exists:", os.path.exists(args.stats_json))
    print("[DEBUG] stats loaded:", "YES" if stats is not None else "NO")
    if stats is not None:
        print("[DEBUG] p1[0], p99[0]:", stats["p1"][0], stats["p99"][0])

    # --------------------------
    # Datasets / loaders
    # --------------------------
    train_ds = MarsLSDataset(
        img_dir=os.path.join(args.data_root, "train/images"),
        mask_dir=os.path.join(args.data_root, "train/masks"),
        return_streams=True,
        transforms=get_transforms(train=True),
        stats=stats,
        probe=args.probe,
    )

    val_ds = MarsLSDataset(
        img_dir=os.path.join(args.data_root, "val/images"),
        mask_dir=os.path.join(args.data_root, "val/masks"),
        return_streams=True,
        transforms=get_transforms(train=False),
        stats=stats,
        probe=False,
    )

    test_ds = MarsLSDataset(
        img_dir=os.path.join(args.data_root, "test/images"),
        mask_dir=None,
        return_streams=True,
        transforms=get_transforms(train=False),
        stats=stats,
        probe=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --------------------------
    # Model
    # --------------------------
    # Two-stream: A has 5 channels, B has 2 channels
    model = TwoStreamSegformer(classes=1, stem_out=32, inA=5, inB=2).to(device)

    # --------------------------
    # Optimizer + scheduler
    # --------------------------
    param_groups = build_weight_decay_param_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
     #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP scaler
    scaler = GradScaler(enabled=(args.amp and device == "cuda"))

    # --------------------------
    # Checkpoint paths
    # --------------------------
    best_f1 = -1.0
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_meter = AverageMeter()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            ncols=100,
            mininterval=5.0,
            leave=False,
        )

        for x, y, filename in pbar:
            optimizer.zero_grad(set_to_none=True)

            y = y.to(device, non_blocking=True)

            # Forward
            if isinstance(x, (tuple, list)):
                # For your TwoStreamSegformer: use first two streams
                if len(x) == 2:
                    xA, xB = x
                else:
                    xA, xB, _ = x  # ignore XC for this model

                xA = xA.to(device, non_blocking=True)
                xB = xB.to(device, non_blocking=True)
                bs = xA.size(0)

                with autocast(enabled=(args.amp and device == "cuda")):
                    logits = model(xA, xB)
                    loss = bce_dice_loss(logits, y)
            else:
                x = x.to(device, non_blocking=True)
                bs = x.size(0)

                with autocast(enabled=(args.amp and device == "cuda")):
                    logits = model(x)
                    loss = bce_dice_loss(logits, y)

            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            loss_meter.update(loss.item(), bs)
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        # Step scheduler once per epoch
        scheduler.step()

        # --------------------------
        # Validate
        # --------------------------
        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            loss_fn=bce_dice_loss,
            thr=0.5,
        )
        dt = time.time() - t0

        print(
            "[VAL] epoch={epoch} loss={loss:.4f} dice={dice:.4f} iou={iou:.4f} "
            "miou={miou:.4f} iou_fg={iou_fg:.4f} iou_bg={iou_bg:.4f} "
            "precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} time={dt:.1f}s".format(
                epoch=epoch, dt=dt, **val_metrics
            )
        )

        # --------------------------
        # Save checkpoints
        # --------------------------
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "val_f1": val_metrics["f1"]},
            last_path,
        )

        if (val_metrics["f1"] > best_f1) and epoch >= 50:  # save best only after some warmup epochs
            best_f1 = val_metrics["f1"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_f1": best_f1},
                best_path,
            )
            print(f"[SAVE] best -> {best_path} (f1={best_f1:.4f})")

    # --------------------------
    # Inference with best checkpoint
    # --------------------------
    print("[INFO] Loading best model for inference...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    run_test_inference(model, test_loader, device)
    #run_test_val_inference(model, val_loader, device, thresh=0.5)

    print(
        f"[DONE] Best model from epoch {checkpoint['epoch']} "
        f"(val_f1={checkpoint['val_f1']:.4f})"
    )


if __name__ == "__main__":
    main()