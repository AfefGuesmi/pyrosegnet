"""Build pixel-level feature matrices for classical ML."""
import numpy as np
from pyrosegnet.features.spectral_indices import compute_nbr


def extract_pixel_features(pre, post):
    """
    Build a (H*W, 39) feature matrix from a pre/post pair.

    Features:
      - 12 pre-fire bands
      - 12 post-fire bands
      - 12 per-band temporal differences
      - NBR_pre, NBR_post, dNBR

    pre, post: (12, H, W) float32
    returns:   (H*W, 39) float32
    """
    bands, H, W = pre.shape
    feats = []
    for b in range(bands):
        feats.append(pre[b].reshape(-1))
    for b in range(bands):
        feats.append(post[b].reshape(-1))
    for b in range(bands):
        feats.append((post[b] - pre[b]).reshape(-1))
    nbr_pre = compute_nbr(pre)
    nbr_post = compute_nbr(post)
    feats.append(nbr_pre.reshape(-1))
    feats.append(nbr_post.reshape(-1))
    feats.append((nbr_pre - nbr_post).reshape(-1))
    return np.stack(feats, axis=1).astype(np.float32)


FEATURE_NAMES = (
    [f"pre_B{i:02d}" for i in range(12)] +
    [f"post_B{i:02d}" for i in range(12)] +
    [f"diff_B{i:02d}" for i in range(12)] +
    ["NBR_pre", "NBR_post", "dNBR"]
)