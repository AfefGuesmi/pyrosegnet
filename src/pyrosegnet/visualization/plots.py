"""Plotting utilities for CaBuAr samples and predictions."""
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
