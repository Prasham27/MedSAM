import torch
import torch.nn.functional as F


def bce_dice_loss(logits, gt, eps=1e-6):
    logits, gt = logits.float(), gt.float()
    dims = tuple(range(1, logits.ndim))
    bce = F.binary_cross_entropy_with_logits(logits, gt)
    p = torch.sigmoid(logits)
    inter = (p * gt).sum(dims)
    dice = (2 * inter + eps) / (p.sum(dims) + gt.sum(dims) + eps)
    return bce + (1 - dice).mean()


@torch.no_grad()
def slice_dice(logits, gt, eps=1e-6):
    """Per-slice hard Dice (logit > 0 vs gt > 0.5). Both-empty slices score 1. Returns list of floats."""
    dims = tuple(range(1, logits.ndim))
    pred = (logits.float() > 0).float()
    gt = (gt.float() > 0.5).float()
    inter = (pred * gt).sum(dims)
    s = pred.sum(dims) + gt.sum(dims)
    d = torch.where(s == 0, torch.ones_like(s), 2 * inter / (s + eps))
    return d.tolist()
