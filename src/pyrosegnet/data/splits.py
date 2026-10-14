"""Use CaBuAr's official fold attributes for train/val/test splits."""
import h5py


# Map each fold -> split role (matches the CaBuAr paper convention)
FOLD_TO_SPLIT = {
    0: "test",
    1: "val",
    2: "train",
    3: "train",
    4: "train",
}

REQUIRED_DATASETS = {"pre_fire", "post_fire", "mask"}


def _is_real_sample(group):
    """A real sample has pre_fire, post_fire, and mask as datasets."""
    if not isinstance(group, h5py.Group):
        return False
    return REQUIRED_DATASETS.issubset(set(group.keys()))


def get_split_samples(hdf5_path):
    """
    Return {'train': [...], 'val': [...], 'test': [...]} from official folds.
    Only includes groups that are actual samples (contain pre_fire, post_fire, mask).
    """
    splits = {"train": [], "val": [], "test": []}
    skipped_no_fold = 0
    skipped_not_sample = 0

    with h5py.File(hdf5_path, "r") as f:
        for k in f.keys():
            g = f[k]
            if not _is_real_sample(g):
                skipped_not_sample += 1
                continue
            fold = g.attrs.get("fold", None)
            if fold is None:
                skipped_no_fold += 1
                continue
            role = FOLD_TO_SPLIT.get(int(fold))
            if role:
                splits[role].append(k)

    print(f"  [split] skipped {skipped_not_sample} non-sample groups")
    print(f"  [split] skipped {skipped_no_fold} samples missing fold attr")
    return splits
