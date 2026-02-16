import torch

@torch.no_grad()
def segmentation_metrics_from_logits(logits, targets, eps=1e-6, thr=0.5, epoch=None,filename=None):
    import os
    import matplotlib.pyplot as plt
    """
    logits: (B,1,H,W)
    targets: (B,1,H,W) float {0,1}
    Returns dict of mean metrics across batch.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    """
    if epoch == 200:
       
            

            os.makedirs("val_gt", exist_ok=True)
            os.makedirs("val_pred", exist_ok=True)
                

            gt = targets.detach().cpu()         # [B,1,H,W] or [B,H,W]
            pred = preds.detach().cpu()  # [B,1,H,W]
            #print("pred min/max:", pred.min().item(), pred.max().item())
            # print("pred unique (first 10):", torch.unique(pred)[:10])

            if gt.dim() == 4:
                gt = gt[:, 0]              # [B,H,W]
            if pred.dim() == 4:
                pred = pred[:, 0]          # [B,H,W]

            for i in range(gt.shape[0]):   # 🔑 loop over batch
                name = os.path.splitext(filename[i])[0]  # remove .png/.jpg
                plt.imsave(
                    f"val_gt/gt_{name}.png",
                    gt[i],
                    cmap="gray"
                )

                plt.imsave(
                    f"val_pred/pred_{name}.png",
                    pred[i],
                    cmap="gray"
                )
    """
    tp = (preds * targets).sum(dim=(2, 3))
    fp = (preds * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - preds) * targets).sum(dim=(2, 3))
    tn = ((1 - preds) * (1 - targets)).sum(dim=(2, 3))

    iou_fg = (tp + eps) / (tp + fp + fn + eps)
    iou_bg = (tn + eps) / (tn + fp + fn + eps)
    miou = 0.5 * (iou_fg + iou_bg)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return {
        "miou": miou.mean().item(),
        "iou_fg": iou_fg.mean().item(),
        "iou_bg": iou_bg.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }

@torch.no_grad()
def dice_iou_from_logits(logits, targets, eps=1e-6, thr=0.5):
    """
    logits: (B,1,H,W)
    targets: (B,1,H,W) float {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    inter = (preds * targets).sum(dim=(2,3))
    union = (preds + targets).clamp(0,1).sum(dim=(2,3))  # for IoU we need (pred|tgt)
    # correct union for IoU: union = pred + tgt - inter
    pred_sum = preds.sum(dim=(2,3))
    tgt_sum  = targets.sum(dim=(2,3))
    union_iou = pred_sum + tgt_sum - inter

    dice = (2*inter + eps) / (pred_sum + tgt_sum + eps)
    iou  = (inter + eps) / (union_iou + eps)

    return dice.mean().item(), iou.mean().item()