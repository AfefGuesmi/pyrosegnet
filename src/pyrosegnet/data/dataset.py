"""PyTorch Dataset for CaBuAr HDF5 with optional resizing."""
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CaBuArDataset(Dataset):
    """Loads pre_fire, post_fire, mask. Optionally resizes to img_size."""

    def __init__(self, hdf5_path, sample_keys, img_size=None):
        self.hdf5_path = str(hdf5_path)
        self.sample_keys = list(sample_keys)
        self.img_size = img_size

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        key = self.sample_keys[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            g = f[key]
            pre = g["pre_fire"][()]
            post = g["post_fire"][()]
            mask = g["mask"][()]

        pre = np.transpose(pre, (2, 0, 1)).astype(np.float32)
        post = np.transpose(post, (2, 0, 1)).astype(np.float32)
        mask = mask[0].astype(np.float32)

        pre = torch.from_numpy(pre)
        post = torch.from_numpy(post)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.img_size is not None and pre.shape[-1] != self.img_size:
            size = (self.img_size, self.img_size)
            pre = F.interpolate(pre.unsqueeze(0), size=size,
                                mode="bilinear", align_corners=False)[0]
            post = F.interpolate(post.unsqueeze(0), size=size,
                                 mode="bilinear", align_corners=False)[0]
            mask = F.interpolate(mask.unsqueeze(0), size=size,
                                 mode="nearest")[0]

        return {"pre": pre, "post": post,
                "mask": mask.squeeze(0), "key": key}
