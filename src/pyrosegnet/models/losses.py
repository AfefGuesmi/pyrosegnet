# Losses for binary segmentation with class imbalance.
import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    # Weighted sum of BCEWithLogitsLoss and soft Dice loss.
    # pos_weight handles class imbalance (burned is ~3% of pixels).
    def __init__(self, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_weight = bce_weight

    @staticmethod
    def dice_loss(logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

    def forward(self, logits, targets):
        return (self.bce_weight * self.bce(logits, targets)
                + (1 - self.bce_weight) * self.dice_loss(logits, targets))
