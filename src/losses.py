import torch
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2,3))
    den = (probs + targets).sum(dim=(2,3)) + eps
    dice = num / den
    return 1 - dice.mean()

def bce_dice_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dsc = dice_loss(logits, targets)
    return 0.5*bce + 0.5*dsc


def tversky_loss(logits, targets, alpha=0.7, beta=0.3, eps=1e-6):
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(dim=(2,3))
    fp = (probs * (1 - targets)).sum(dim=(2,3))
    fn = ((1 - probs) * targets).sum(dim=(2,3))
    tversky = (tp + eps) / (tp + alpha*fp + beta*fn + eps)
    return 1 - tversky.mean()

def bce_tversky_loss(logits, targets, alpha=0.7, beta=0.3):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    tv  = tversky_loss(logits, targets, alpha=alpha, beta=beta)
    return 0.4 * bce + 0.6 * tv

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal = alpha_t * (1 - pt) ** gamma * bce
    return focal.mean()


def focal_dice_loss(logits, targets, alpha=0.25, gamma=2.0):
    fl = focal_loss(logits, targets, alpha=alpha, gamma=gamma)
    dl = dice_loss(logits, targets)
    return 0.2 * fl + 0.8 * dl