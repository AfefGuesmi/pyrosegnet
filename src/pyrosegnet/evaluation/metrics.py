"""Binary segmentation metrics computed from confusion-matrix counts."""
import numpy as np


def metrics_from_confusion(TP, FP, TN, FN):
    """Compute all standard binary metrics from integer counts."""
    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    # Balanced accuracy = mean of sensitivity and specificity
    specificity = TN / (TN + FP + eps)
    balanced_accuracy = (recall + specificity) / 2
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def confusion_from_pred_mask(pred, mask):
    """Return (TP, FP, TN, FN) as Python ints."""
    pred = pred.astype(np.int8)
    mask = mask.astype(np.int8)
    TP = int(((pred == 1) & (mask == 1)).sum())
    FP = int(((pred == 1) & (mask == 0)).sum())
    TN = int(((pred == 0) & (mask == 0)).sum())
    FN = int(((pred == 0) & (mask == 1)).sum())
    return TP, FP, TN, FN
