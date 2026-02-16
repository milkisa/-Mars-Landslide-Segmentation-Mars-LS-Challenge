#!/usr/bin/env python3
import os
import sys
import argparse

# ---------------------------------------------------------------------
# Cache / environment (your original path)
# ---------------------------------------------------------------------
os.environ["HOME"] = "/home/milkisayebasse/Mars/.cache"

# ---------------------------------------------------------------------
# Project root so `src.*` imports work
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, ensure_dir, load_json, AverageMeter
from src.datasets import MarsLSDataset
from src.losses import bce_dice_loss
from src.metrics import dice_iou_from_logits, segmentation_metrics_from_logits

from src.augmentation import TorchSegTransform
from src.models.twoStreamSMP import TwoStreamSegformer

from inference_save import run_test_inference
from validate_metrics import validate,run_test_val_inference


# -----------------------------
# Transforms
# -----------------------------
def get_transforms(train: bool = True, debug: bool = False):
    return TorchSegTransform(train=train, debug=debug)


# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def validate(model, loader, device, epoch: int):
    model.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter  = AverageMeter()
    miou_meter = AverageMeter()
    iou_fg_meter = AverageMeter()
    iou_bg_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()

    for x, y, filename in loader:
        # Dataset may return x as tuple/list when return_streams=True
        if isinstance(x, (tuple, list)):
            # Your dataset returns (xA, xB, XC) but this model uses only A,B
            xA, xB, *_ = x
            xA = xA.to(device, non_blocking=True)
            xB = xB.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)

            logits = model(xA, xB)  # <-- TwoStreamSegformer forward
            bs = xA.size(0)
        else:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            bs = x.size(0)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        loss = bce_dice_loss(logits, y)

        dice, iou = dice_iou_from_logits(logits, y, thr=0.5)
        seg_metrics = segmentation_metrics_from_logits(
            logits, y, thr=0.5, epoch=epoch, filename=filename
        )

        loss_meter.update(loss.item(), bs)
        dice_meter.update(dice, bs)
        iou_meter.update(iou, bs)
        miou_meter.update(seg_metrics["miou"], bs)
        iou_fg_meter.update(seg_metrics["iou_fg"], bs)
        iou_bg_meter.update(seg_metrics["iou_bg"], bs)
        precision_meter.update(seg_metrics["precision"], bs)
        recall_meter.update(seg_metrics["recall"], bs)
        f1_meter.update(seg_metrics["f1"], bs)

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "miou": miou_meter.avg,
        "iou_fg": iou_fg_meter.avg,
        "iou_bg": iou_bg_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
        "f1": f1_meter.avg,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/mnt/data/mars")
    ap.add_argument("--stats_json", type=str, default="/mnt/data/mars/stats.json")
    ap.add_argument("--out_dir", type=str, default="outputs/exp01")
    ap.add_argument("--epochs", type=int, default=201)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--debug_tfms", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to load (defaults to out_dir/checkpoints/last.pt)")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "checkpoints"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = load_json(args.stats_json) if os.path.exists(args.stats_json) else None
    print("[DEBUG] stats_json:", args.stats_json, "exists:", os.path.exists(args.stats_json))
    print("[DEBUG] stats loaded:", "YES" if stats is not None else "NO")
    if stats is not None:
        print("[DEBUG] p1[0], p99[0]:", stats["p1"][0], stats["p99"][0])

    # -----------------------------
    # Datasets / loaders
    # -----------------------------
    train_ds = MarsLSDataset(
        img_dir=os.path.join(args.data_root, "train/images"),
        mask_dir=os.path.join(args.data_root, "train/masks"),
        return_streams=True,
        transforms=get_transforms(train=True, debug=args.debug_tfms),
        stats=stats,
        probe=args.probe,
    )
    val_ds = MarsLSDataset(
        img_dir=os.path.join(args.data_root, "val/images"),
        mask_dir=os.path.join(args.data_root, "val/masks"),
        return_streams=True,
        transforms=get_transforms(train=False, debug=args.debug_tfms),
        stats=stats,
        probe=False,
    )
    test_ds = MarsLSDataset(
        img_dir=os.path.join(args.data_root, "test/images"),
        mask_dir=None,
        return_streams=True,
        transforms=get_transforms(train=False, debug=args.debug_tfms),
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

    # -----------------------------
    # Model
    # -----------------------------
    model = TwoStreamSegformer(classes=1, stem_out=32, inA=5, inB=2).to(device)

    # -----------------------------
    # Load checkpoint + run inference
    # -----------------------------
    ckpt_path = args.ckpt or os.path.join(args.out_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    print(
        f"[LOAD] {ckpt_path} | epoch={checkpoint.get('epoch', 'NA')} "
        f"val_f1={checkpoint.get('val_f1', float('nan')):.4f}"
    )

    run_test_inference(model, test_loader, device)
    #run_test_val_inference(model, val_loader, device)

    print("[DONE]")


if __name__ == "__main__":
    main()