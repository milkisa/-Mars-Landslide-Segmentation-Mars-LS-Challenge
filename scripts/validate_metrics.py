# validate.py
"""
Validation utilities for Mars landslide segmentation.

Exports:
- validate(model, loader, device, epoch, loss_fn, thr)
"""

from typing import Dict
import torch

from src.utils import AverageMeter
from src.metrics import dice_iou_from_logits, segmentation_metrics_from_logits
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import torch

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    device: str,
    epoch: int,
    loss_fn,
    thr: float = 0.5,
) -> Dict[str, float]:
    """
    Run validation and return averaged metrics.

    Supports:
      - single input: logits = model(x)
      - multi-stream: logits = model(xA, xB)  (and ignores XC if present)
    """
    model.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    miou_meter = AverageMeter()
    iou_fg_meter = AverageMeter()
    iou_bg_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()

    for x, y, filename in loader:
        y = y.to(device, non_blocking=True)

        # Forward
        if isinstance(x, (tuple, list)):
            # Expected: (xA, xB) or (xA, xB, xC)
            if len(x) == 2:
                xA, xB = x
            elif len(x) == 3:
                xA, xB, _ = x  # ignore third stream if not needed
            else:
                raise ValueError(f"Unexpected number of streams: {len(x)}")

            xA = xA.to(device, non_blocking=True)
            xB = xB.to(device, non_blocking=True)
            logits = model(xA, xB)
            bs = xA.size(0)
        else:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            bs = x.size(0)

        # Some models return (logits, aux...)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # Loss
        loss = loss_fn(logits, y)

        # Metrics
        dice, iou = dice_iou_from_logits(logits, y, thr=thr)
        seg_metrics = segmentation_metrics_from_logits(
            logits, y, thr=thr, epoch=epoch, filename=filename
        )

        # Update meters
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

def run_test_val_inference(model, loader, device, tag="best", thresh=0.5):
    model.eval()

   
  

    base_dir = "val_preds"

    # Prediction folders
    out_png_pred  = os.path.join(base_dir, "png")
    out_tiff_pred = os.path.join(base_dir, "tiff")

    # GT folders
    out_png_gt  = os.path.join(base_dir, "gt_png")
    out_tiff_gt = os.path.join(base_dir, "gt_tiff")

    os.makedirs(out_png_pred, exist_ok=True)
    os.makedirs(out_tiff_pred, exist_ok=True)
    os.makedirs(out_png_gt, exist_ok=True)
    os.makedirs(out_tiff_gt, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (x, y, filename) in enumerate(loader):

            # ----------------------------
            # Load inputs (streams or single)
            # ----------------------------
            if isinstance(x, (tuple, list)):
                xA, xB, XC = x
                xA = xA.to(device, non_blocking=True)
                xB = xB.to(device, non_blocking=True)
                XC = XC.to(device, non_blocking=True)
                y  = y.to(device, non_blocking=True)

                logits = model(xA, xB)
                bs = xA.size(0)
            else:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                bs = x.size(0)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            # ----------------------------
            # Convert logits -> mask
            # ----------------------------
            prob = torch.sigmoid(logits)
            pred = (prob > thresh).float()

            if pred.dim() == 3:
                pred = pred.unsqueeze(1)

            pred = pred.detach().cpu().numpy()

            # ----------------------------
            # Process Ground Truth
            # ----------------------------
            if y.dim() == 4:
                y = y[:, 0]  # remove channel if [B,1,H,W]

            gt = y.detach().cpu().numpy()

            # ----------------------------
            # Save per-sample
            # ----------------------------
            if isinstance(filename, (list, tuple)):
                names = filename
            else:
                names = [str(filename)] * bs

            for i in range(bs):
                name = os.path.splitext(os.path.basename(names[i]))[0]

                # Prediction
                mask_pred = (pred[i, 0] * 255).astype(np.uint8)

                png_pred_path = os.path.join(out_png_pred, f"{name}.png")
                tif_pred_path = os.path.join(out_tiff_pred, f"{name}.tif")

                plt.imsave(png_pred_path, mask_pred, cmap="gray")
                tiff.imwrite(
                    tif_pred_path,
                    mask_pred,
                    compression="lzw",
                    resolution=(96, 96),
                    resolutionunit="INCH",
                )

                # Ground Truth
                mask_gt = (gt[i] * 255).astype(np.uint8)

                png_gt_path = os.path.join(out_png_gt, f"{name}.png")
                tif_gt_path = os.path.join(out_tiff_gt, f"{name}.tif")

                plt.imsave(png_gt_path, mask_gt, cmap="gray")
                tiff.imwrite(
                    tif_gt_path,
                    mask_gt,
                    compression="lzw",
                    resolution=(96, 96),
                    resolutionunit="INCH",
                )

    print(f"✅ Saved predictions + GT to: {base_dir}")