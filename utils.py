"""
utils.py
--------
Loss functions and evaluation metrics for Brain MRI segmentation.

Metrics:
    - Dice Coefficient
    - Jaccard Index (IoU)
    - Pixel Accuracy

Losses:
    - Dice Loss
    - Combined Dice + BCE Loss (used as training objective)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Metrics ──────────────────────────────────────────────────────────────────

def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor,
                     threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Dice coefficient for binary segmentation.

    Args:
        logits    : raw model output [B, 1, H, W] — sigmoid applied here
        targets   : binary ground truth, same shape
        threshold : binarization cutoff on probabilities
        smooth    : Laplace smoothing to avoid division by zero

    Returns:
        mean Dice over batch (float)
    """
    preds   = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()

    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean().item()


def jaccard_index(logits: torch.Tensor, targets: torch.Tensor,
                  threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Jaccard Index (IoU) for binary segmentation.
    """
    preds   = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()

    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union        = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                   threshold: float = 0.5) -> float:
    """Pixel-level accuracy (fraction of correctly classified pixels)."""
    preds   = (torch.sigmoid(logits) > threshold).float()
    correct = (preds == targets.float()).float()
    return correct.mean().item()


def compute_all_metrics(logits: torch.Tensor, targets: torch.Tensor,
                        threshold: float = 0.5) -> dict:
    """Return a dict with all three metrics."""
    return {
        "dice":     dice_coefficient(logits, targets, threshold),
        "iou":      jaccard_index(logits, targets, threshold),
        "accuracy": pixel_accuracy(logits, targets, threshold),
    }


# ── Loss Functions ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Differentiable soft Dice Loss that accepts RAW LOGITS.

    Applies sigmoid internally before computing soft dice, so it is safe
    to use under torch.amp.autocast (unlike BCELoss which requires probabilities
    and is explicitly blocked by PyTorch's AMP dispatcher).

    soft Dice = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)
    where P = sigmoid(logits)
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds   = torch.sigmoid(logits)          # logits → probabilities
        preds   = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        intersection = (preds * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss that accepts RAW LOGITS — fully AMP-safe.

    BCELoss (requires probabilities) is unsafe under autocast because FP16
    cannot represent values close to 0 or 1 accurately, causing NaN gradients.
    BCEWithLogitsLoss folds sigmoid + BCE into one numerically stable op and
    is explicitly whitelisted by PyTorch's AMP dispatcher.

    DiceLoss here also applies sigmoid internally (see above).

    loss = α·DiceLoss(logits) + (1-α)·BCEWithLogitsLoss(logits)
    Default α = 0.5.
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.dice  = DiceLoss(smooth=smooth)
        # BCEWithLogitsLoss: numerically stable, AMP-safe, takes raw logits
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        return (self.alpha * self.dice(logits, targets)
                + (1 - self.alpha) * self.bce(logits, targets))