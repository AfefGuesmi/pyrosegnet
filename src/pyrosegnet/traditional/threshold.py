"""dNBR threshold baseline."""
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
