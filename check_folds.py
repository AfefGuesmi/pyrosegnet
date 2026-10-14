"""Check fold attribute distribution across CaBuAr samples."""
import h5py
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

HDF5_PATH = Path("data/raw/512x512.hdf5")

with h5py.File(HDF5_PATH, "r") as f:
    sample_keys = [k for k in f.keys() if isinstance(f[k], h5py.Group)]

    fold_per_sample = {}
    missing = 0
    for k in sample_keys:
        fold = f[k].attrs.get("fold", None)
        if fold is None:
            missing += 1
            fold_per_sample[k] = None
        else:
            fold_per_sample[k] = int(fold)

    print("=" * 60)
    print("FOLD ATTRIBUTE DISTRIBUTION")
    print("=" * 60)
    print(f"Total samples: {len(sample_keys)}")
    print(f"Samples missing fold attribute: {missing}")

    # Distribution of fold values
    folds = [v for v in fold_per_sample.values() if v is not None]
    counts = Counter(folds)
    print(f"\nSamples per fold:")
    for fold in sorted(counts.keys()):
        print(f"  Fold {fold}: {counts[fold]} samples")

    # Check: do all patches from the same fire share the same fold?
    fire_to_folds = defaultdict(set)
    for k, fold in fold_per_sample.items():
        uuid = k.rsplit("_", 1)[0] if "_" in k else k
        if fold is not None:
            fire_to_folds[uuid].add(fold)

    leaks = [uuid for uuid, folds_set in fire_to_folds.items()
             if len(folds_set) > 1]
    print(f"\nFires with patches in multiple folds (LEAKAGE): {len(leaks)}")
    if leaks[:5]:
        print(f"  Examples: {leaks[:5]}")
    else:
        print("  ✅ No leakage — each fire's patches share the same fold.")

    # Per-fold sample counts by fire
    fold_fires = defaultdict(set)
    for k, fold in fold_per_sample.items():
        uuid = k.rsplit("_", 1)[0] if "_" in k else k
        if fold is not None:
            fold_fires[fold].add(uuid)
    print(f"\nFires per fold:")
    for fold in sorted(fold_fires.keys()):
        print(f"  Fold {fold}: {len(fold_fires[fold])} unique fires")