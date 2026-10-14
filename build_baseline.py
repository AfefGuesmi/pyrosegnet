"""Build the complete threshold-baseline pipeline."""
from pathlib import Path

ROOT = Path(__file__).parent


def write(path, content):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    print(f"  [WROTE] {path}")


# ============ src/pyrosegnet/data/splits.py ============
write("src/pyrosegnet/data/splits.py", '''"""Use CaBuAr's official fold attributes for train/val/test splits."""
import h5py


# Map each fold -> split role (chosen to match the CaBuAr paper convention)
FOLD_TO_SPLIT = {
    0: "test",     # 219 samples, 68 fires
    1: "val",      # 136 samples, 68 fires
    2: "train",    # 133 samples
    3: "train",    # 137 samples
    4: "train",    # 102 samples
}


def get_split_samples(hdf5_path):
    """Return {'train': [...], 'val': [...], 'test': [...]} from official folds."""
    splits = {"train": [], "val": [], "test": []}
    with h5py.File(hdf5_path, "r") as f:
        for k in f.keys():
            if not isinstance(f[k], h5py.Group):
                continue
            fold = f[k].attrs.get("fold", None)
            if fold is None:
                continue
            role = FOLD_TO_SPLIT.get(int(fold))
            if role:
                splits[role].append(k)
    return splits
''')


# ============ src/pyrosegnet/data/dataset.py ============
write("src/pyrosegnet/data/dataset.py", '''"""PyTorch Dataset for CaBuAr HDF5 (handles HWC -> CHW)."""
import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset


class CaBuArDataset(Dataset):
    """
    Loads pre_fire, post_fire, mask from CaBuAr HDF5.
    Returns CHW tensors. Values kept in [0, 255] as float32.
    Normalization should be done in the model or transform if needed.
    """

    def __init__(self, hdf5_path, sample_keys):
        self.hdf5_path = str(hdf5_path)
        self.sample_keys = list(sample_keys)

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        key = self.sample_keys[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            g = f[key]
            pre = g["pre_fire"][()]      # (512, 512, 12) uint8
            post = g["post_fire"][()]
            mask = g["mask"][()]         # (1, 512, 512) uint8

        # HWC -> CHW
        pre = np.transpose(pre, (2, 0, 1)).astype(np.float32)
        post = np.transpose(post, (2, 0, 1)).astype(np.float32)
        mask = mask[0].astype(np.float32)      # (512, 512)

        return {
            "pre": torch.from_numpy(pre),
            "post": torch.from_numpy(post),
            "mask": torch.from_numpy(mask),
            "key": key,
        }
''')


# ============ src/pyrosegnet/features/spectral_indices.py ============
write("src/pyrosegnet/features/spectral_indices.py", '''"""Spectral indices for burned-area detection."""
import numpy as np


# CaBuAr 12-band order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
B02_BLUE = 1
B03_GREEN = 2
B04_RED = 3
B08_NIR = 7
B11_SWIR1 = 10
B12_SWIR2 = 11


def compute_nbr(image):
    """
    Normalized Burn Ratio (NBR).
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    image: (12, H, W) float
    returns: (H, W) in [-1, 1]
    """
    nir = image[B08_NIR].astype(np.float32)
    swir2 = image[B12_SWIR2].astype(np.float32)
    return (nir - swir2) / (nir + swir2 + 1e-8)


def compute_dnbr(pre_image, post_image):
    """
    Differenced NBR (dNBR).
    dNBR = NBR_pre - NBR_post
    Positive values indicate vegetation loss (burning).
    """
    return compute_nbr(pre_image) - compute_nbr(post_image)


def compute_bai(image):
    """
    Burn Area Index (BAI).
    BAI = 1 / ((0.1 - Red)^2 + (0.06 - NIR)^2)
    Higher values indicate burned area.
    """
    red = image[B04_RED].astype(np.float32) / 255.0
    nir = image[B08_NIR].astype(np.float32) / 255.0
    return 1.0 / ((0.1 - red) ** 2 + (0.06 - nir) ** 2 + 1e-8)
''')


# ============ src/pyrosegnet/evaluation/metrics.py ============
write("src/pyrosegnet/evaluation/metrics.py", '''"""Binary segmentation metrics computed from confusion-matrix counts."""
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
''')


# ============ src/pyrosegnet/traditional/threshold.py ============
write("src/pyrosegnet/traditional/threshold.py", '''"""dNBR threshold baseline."""
import h5py
import numpy as np
from pyrosegnet.features.spectral_indices import compute_dnbr
from pyrosegnet.evaluation.metrics import confusion_from_pred_mask


def load_dnbr_and_mask(hdf5_path, sample_key):
    """Load one sample; return (dNBR, mask) both as float32/int8."""
    with h5py.File(hdf5_path, "r") as f:
        g = f[sample_key]
        pre = np.transpose(g["pre_fire"][()], (2, 0, 1)).astype(np.float32)
        post = np.transpose(g["post_fire"][()], (2, 0, 1)).astype(np.float32)
        mask = g["mask"][()][0].astype(np.int8)
    return compute_dnbr(pre, post), mask


def sweep_thresholds(hdf5_path, sample_keys, thresholds, verbose=False):
    """
    For each threshold t, accumulate confusion counts over all samples.
    Returns: {t: {"TP": int, "FP": int, "TN": int, "FN": int}}
    """
    results = {float(t): {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for t in thresholds}
    n = len(sample_keys)
    for i, key in enumerate(sample_keys):
        dnbr, mask = load_dnbr_and_mask(hdf5_path, key)
        for t in thresholds:
            pred = (dnbr >= t).astype(np.int8)
            TP, FP, TN, FN = confusion_from_pred_mask(pred, mask)
            r = results[float(t)]
            r["TP"] += TP; r["FP"] += FP; r["TN"] += TN; r["FN"] += FN
        if verbose and (i + 1) % 50 == 0:
            print(f"    processed {i+1}/{n}")
    return results
''')


# ============ src/pyrosegnet/visualization/plots.py ============
write("src/pyrosegnet/visualization/plots.py", '''"""Plotting utilities for CaBuAr samples and predictions."""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _norm_rgb(rgb):
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def plot_prediction(hdf5_path, sample_key, pred_mask, dnbr=None,
                    save_path=None, title_extra=""):
    """
    5-panel figure: pre-fire RGB, post-fire RGB, GT mask, prediction, error map.
    pred_mask: (H, W) binary numpy array (0/1).
    """
    with h5py.File(hdf5_path, "r") as f:
        g = f[sample_key]
        pre = g["pre_fire"][()]     # (512, 512, 12)
        post = g["post_fire"][()]
        mask = g["mask"][()][0]

    pre_rgb = _norm_rgb(pre[:, :, [3, 2, 1]].astype(np.float32))
    post_rgb = _norm_rgb(post[:, :, [3, 2, 1]].astype(np.float32))

    # Error map: TP=green, FP=red, FN=blue, TN=light
    error = np.zeros((*mask.shape, 3), dtype=np.float32)
    error[...] = [0.95, 0.95, 0.95]  # TN
    error[(mask == 1) & (pred_mask == 1)] = [0.1, 0.8, 0.1]  # TP
    error[(mask == 0) & (pred_mask == 1)] = [0.9, 0.1, 0.1]  # FP
    error[(mask == 1) & (pred_mask == 0)] = [0.1, 0.3, 0.9]  # FN

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(pre_rgb); axes[0].set_title("Pre-fire (RGB)"); axes[0].axis("off")
    axes[1].imshow(post_rgb); axes[1].set_title("Post-fire (RGB)"); axes[1].axis("off")
    axes[2].imshow(mask, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title(f"Ground Truth (burned={int(mask.sum()):,} px)"); axes[2].axis("off")
    axes[3].imshow(pred_mask, cmap="hot", vmin=0, vmax=1)
    axes[3].set_title(f"Prediction (burned={int(pred_mask.sum()):,} px)"); axes[3].axis("off")
    axes[4].imshow(error)
    axes[4].set_title("Errors: TP=green, FP=red, FN=blue"); axes[4].axis("off")

    plt.suptitle(f"{sample_key} {title_extra}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
''')


# ============ scripts/03_run_threshold_baseline.py ============
write("scripts/03_run_threshold_baseline.py", '''"""Run the dNBR threshold baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from pyrosegnet.config import load_config
from pyrosegnet.data.splits import get_split_samples
from pyrosegnet.traditional.threshold import sweep_thresholds, load_dnbr_and_mask
from pyrosegnet.evaluation.metrics import metrics_from_confusion
from pyrosegnet.visualization.plots import plot_prediction

if __name__ == "__main__":
    config = load_config()
    hdf5_path = config["data"]["hdf5_file"]
    result_dir = Path(config["outputs"]["result_dir"])
    fig_dir = Path(config["outputs"]["figure_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load splits from official folds ----
    print("Loading official folds...")
    splits = get_split_samples(hdf5_path)
    print(f"  train: {len(splits['train'])} samples")
    print(f"  val:   {len(splits['val'])} samples")
    print(f"  test:  {len(splits['test'])} samples")

    # ---- 2. Sweep thresholds on validation ----
    thresholds = np.round(np.arange(
        config["threshold"]["dnbr_range"][0],
        config["threshold"]["dnbr_range"][1] + 1e-9,
        config["threshold"]["dnbr_step"],
    ), 3)
    print(f"\\nSweeping {len(thresholds)} thresholds on validation set...")
    val_results = sweep_thresholds(hdf5_path, splits["val"], thresholds, verbose=True)

    val_rows = []
    for t in thresholds:
        r = val_results[float(t)]
        m = metrics_from_confusion(r["TP"], r["FP"], r["TN"], r["FN"])
        val_rows.append({"threshold": float(t),
                         "TP": r["TP"], "FP": r["FP"],
                         "TN": r["TN"], "FN": r["FN"], **m})
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(result_dir / "threshold_selection.csv", index=False)

    # ---- 3. Pick best threshold by F1 ----
    best_row = val_df.loc[val_df["f1"].idxmax()]
    best_t = float(best_row["threshold"])
    print(f"\\nBest validation threshold: dNBR >= {best_t:.3f}")
    print(f"  Val precision: {best_row['precision']:.4f}")
    print(f"  Val recall:    {best_row['recall']:.4f}")
    print(f"  Val F1:        {best_row['f1']:.4f}")
    print(f"  Val IoU:       {best_row['iou']:.4f}")

    # ---- 4. Evaluate on test with fixed threshold ----
    print(f"\\nEvaluating on test set with threshold={best_t:.3f}...")
    test_results = sweep_thresholds(hdf5_path, splits["test"], [best_t], verbose=True)
    r = test_results[float(best_t)]
    test_metrics = metrics_from_confusion(r["TP"], r["FP"], r["TN"], r["FN"])

    print("\\n=== TEST SET RESULTS ===")
    print(f"  Threshold: {best_t:.3f}")
    print(f"  TP={r['TP']:,}  FP={r['FP']:,}  TN={r['TN']:,}  FN={r['FN']:,}")
    print(f"  Precision (burned): {test_metrics['precision']:.4f}")
    print(f"  Recall (burned):    {test_metrics['recall']:.4f}")
    print(f"  F1 (burned):        {test_metrics['f1']:.4f}")
    print(f"  IoU (burned):       {test_metrics['iou']:.4f}")
    print(f"  Dice (burned):      {test_metrics['dice']:.4f}")
    print(f"  Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Specificity:        {test_metrics['specificity']:.4f}")

    # Save test metrics
    test_row = {"model": "dNBR_threshold", "threshold": best_t,
                "TP": r["TP"], "FP": r["FP"], "TN": r["TN"], "FN": r["FN"],
                **test_metrics}
    pd.DataFrame([test_row]).to_csv(result_dir / "threshold_test_metrics.csv", index=False)

    # ---- 5. Visualization on 3 test samples ----
    print("\\nGenerating prediction visualizations for 3 test samples...")
    # Sort by burned area to get diversity
    sample_burn = []
    for key in splits["test"]:
        _, mask = load_dnbr_and_mask(hdf5_path, key)
        sample_burn.append((float(mask.mean()), key))
    sample_burn.sort()

    picks = [sample_burn[0][1],
             sample_burn[len(sample_burn) // 2][1],
             sample_burn[-1][1]]

    for i, key in enumerate(picks):
        dnbr, mask = load_dnbr_and_mask(hdf5_path, key)
        pred = (dnbr >= best_t).astype(np.int8)
        plot_prediction(
            hdf5_path, key, pred,
            save_path=str(fig_dir / f"threshold_pred_{i}.png"),
            title_extra=f"| t={best_t:.2f}",
        )
        print(f"  saved threshold_pred_{i}.png (burn ratio={mask.mean():.3f})")

    print(f"\\nResults saved to {result_dir}")
    print(f"Figures saved to {fig_dir}")
''')


print("\\nAll baseline files written.")
print("Next: python scripts/03_run_threshold_baseline.py")