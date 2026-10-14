"""Fix the inspection code to handle the actual flat HDF5 structure."""
from pathlib import Path

ROOT = Path(__file__).parent


def write(path, content):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    print(f"  [WROTE] {path}")


# ============ src/pyrosegnet/data/inspect.py ============
write("src/pyrosegnet/data/inspect.py", '''"""Inspect the CaBuAr HDF5 structure (flat layout)."""
import h5py
import hdf5plugin
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def find_hdf5_file(search_dirs=None):
    if search_dirs is None:
        search_dirs = [Path("data/raw"), Path("data"), Path(".")]
    candidates = []
    for d in search_dirs:
        if d.exists():
            candidates.extend(d.rglob("*.hdf5"))
            candidates.extend(d.rglob("*.h5"))
    if not candidates:
        return None
    for c in candidates:
        if "512" in c.name:
            return c
    return candidates[0]


def list_all_samples(hdf5_path):
    """Return list of all top-level sample keys."""
    with h5py.File(hdf5_path, "r") as f:
        return [k for k in f.keys() if isinstance(f[k], h5py.Group)]


def summarize_dataset(hdf5_path):
    """Print a full summary of the flat HDF5 structure."""
    print("=" * 70)
    print(f"Inspecting: {hdf5_path}")
    print(f"File size: {hdf5_path.stat().st_size / 1e9:.2f} GB")
    print("=" * 70)

    with h5py.File(hdf5_path, "r") as f:
        sample_keys = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        print(f"\\nTotal top-level samples: {len(sample_keys)}")

        # Inspect first sample in detail
        first = sample_keys[0]
        g = f[first]
        print(f"\\nFirst sample key: {first}")
        print(f"  Datasets: {list(g.keys())}")
        for k in g.keys():
            ds = g[k]
            print(f"    {k}: shape={ds.shape}, dtype={ds.dtype}")

        # Find fire-event UUIDs (prefix before last underscore)
        fire_uuids = set()
        for k in sample_keys:
            if "_" in k:
                uuid_part = k.rsplit("_", 1)[0]
                fire_uuids.add(uuid_part)
            else:
                fire_uuids.add(k)
        print(f"\\nUnique fire events (UUID prefixes): {len(fire_uuids)}")
        print(f"Average patches per fire: {len(sample_keys) / len(fire_uuids):.1f}")

        # Count patches per fire
        from collections import Counter
        counts = Counter(k.rsplit("_", 1)[0] for k in sample_keys)
        print(f"Patches per fire (first 10): {list(counts.items())[:10]}")

    return sample_keys, fire_uuids


def inspect_one_sample(hdf5_path, sample_key):
    """Print detailed info about one sample."""
    with h5py.File(hdf5_path, "r") as f:
        g = f[sample_key]
        print(f"\\n{'=' * 70}")
        print(f"SAMPLE: {sample_key}")
        print(f"Keys: {list(g.keys())}")
        print(f"Attributes: {dict(g.attrs)}")
        print("=" * 70)

        for k in g.keys():
            ds = g[k]
            data = ds[()]
            print(f"\\n  {k}: shape={data.shape}, dtype={data.dtype}")
            print(f"    min={data.min()}, max={data.max()}, mean={data.mean():.2f}")

            # Mask: show class distribution
            if "mask" in k.lower():
                unique, counts = np.unique(data, return_counts=True)
                total = data.size
                print(f"    Unique values:")
                for v, c in zip(unique, counts):
                    print(f"      {v}: {c:,} px ({c/total*100:.2f}%)")


def analyze_normalization(hdf5_path, n_samples=10):
    """Check whether NBR is meaningful under the current normalization."""
    print(f"\\n{'=' * 70}")
    print("NORMALIZATION CHECK: Is NBR meaningful?")
    print("=" * 70)

    with h5py.File(hdf5_path, "r") as f:
        sample_keys = [k for k in f.keys() if isinstance(f[k], h5py.Group)][:n_samples]

        # Compute per-band global stats across samples
        band_stats = {i: {"min": 255, "max": 0, "sum": 0, "count": 0}
                      for i in range(12)}

        for key in sample_keys:
            g = f[key]
            post = g["post_fire"][()]  # (512, 512, 12)
            for b in range(12):
                band = post[:, :, b]
                band_stats[b]["min"] = min(band_stats[b]["min"], band.min())
                band_stats[b]["max"] = max(band_stats[b]["max"], band.max())
                band_stats[b]["sum"] += band.sum()
                band_stats[b]["count"] += band.size

        print("\\nPer-band statistics (post-fire, first few samples):")
        print(f"{'Band':<6} {'Min':<6} {'Max':<6} {'Mean':<8}")
        for b in range(12):
            s = band_stats[b]
            mean = s["sum"] / s["count"]
            print(f"B{b:<4} {s['min']:<6} {s['max']:<6} {mean:<8.2f}")

        # Compute NBR for one sample
        print("\\nComputing NBR on first sample...")
        g = f[sample_keys[0]]
        pre = g["pre_fire"][()].astype(np.float32)
        post = g["post_fire"][()].astype(np.float32)

        # Assuming band indices: B04 (red)=3, B08 (nir)=7, B11 (swir1)=10, B12 (swir2)=11
        nir_pre = pre[:, :, 7]
        swir2_pre = pre[:, :, 11]
        nbr_pre = (nir_pre - swir2_pre) / (nir_pre + swir2_pre + 1e-8)

        nir_post = post[:, :, 7]
        swir2_post = post[:, :, 11]
        nbr_post = (nir_post - swir2_post) / (nir_post + swir2_post + 1e-8)

        dnbr = nbr_pre - nbr_post

        print(f"\\n  NBR_pre:  min={nbr_pre.min():.3f}, max={nbr_pre.max():.3f}, "
              f"mean={nbr_pre.mean():.3f}")
        print(f"  NBR_post: min={nbr_post.min():.3f}, max={nbr_post.max():.3f}, "
              f"mean={nbr_post.mean():.3f}")
        print(f"  dNBR:     min={dnbr.min():.3f}, max={dnbr.max():.3f}, "
              f"mean={dnbr.mean():.3f}")

        print("\\n  Interpretation:")
        print("  - NBR should be in [-1, 1] for real reflectance data.")
        print("  - dNBR should be positive in burned areas.")
        print("  - If NBR values look extreme, the data is per-band normalized")
        print("    and we must download the RAW version instead.")


def visualize_sample(hdf5_path, sample_key, save_path=None):
    """Visualize one sample: pre-fire RGB, post-fire RGB, mask, overlay."""
    with h5py.File(hdf5_path, "r") as f:
        g = f[sample_key]
        pre = g["pre_fire"][()]      # (512, 512, 12)
        post = g["post_fire"][()]
        mask = g["mask"][()]

        if mask.ndim == 3:
            mask = mask[0]

        # RGB composite: bands B04=idx3, B03=idx2, B02=idx1
        pre_rgb = pre[:, :, [3, 2, 1]].astype(np.float32)
        post_rgb = post[:, :, [3, 2, 1]].astype(np.float32)

        def norm(img):
            lo, hi = np.percentile(img, 2), np.percentile(img, 98)
            return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(norm(pre_rgb)); axes[0].set_title("Pre-fire (RGB)"); axes[0].axis("off")
        axes[1].imshow(norm(post_rgb)); axes[1].set_title("Post-fire (RGB)"); axes[1].axis("off")
        axes[2].imshow(mask, cmap="hot", vmin=0, vmax=1)
        axes[2].set_title(f"Mask (burned={int(mask.sum()):,} px)"); axes[2].axis("off")
        axes[3].imshow(norm(post_rgb))
        axes[3].imshow(mask, cmap="Reds", alpha=0.4, vmin=0, vmax=1)
        axes[3].set_title("Post-fire + Burned Overlay"); axes[3].axis("off")
        plt.suptitle(f"Sample: {sample_key}", fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"Saved figure: {save_path}")
        plt.close()
''')


# ============ src/pyrosegnet/data/splits.py ============
write("src/pyrosegnet/data/splits.py", '''"""Split CaBuAr samples by fire event (UUID prefix) to avoid leakage."""
import random
from collections import defaultdict


def group_by_fire(sample_keys):
    """Group sample keys by their UUID prefix (fire event)."""
    groups = defaultdict(list)
    for k in sample_keys:
        uuid_part = k.rsplit("_", 1)[0] if "_" in k else k
        groups[uuid_part].append(k)
    return dict(groups)


def split_by_fire(sample_keys, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split samples into train/val/test by fire event.
    Ensures all patches from the same fire are in the same split.
    """
    groups = group_by_fire(sample_keys)
    fire_ids = sorted(groups.keys())

    rng = random.Random(seed)
    rng.shuffle(fire_ids)

    n = len(fire_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_fires = fire_ids[:n_train]
    val_fires = fire_ids[n_train:n_train + n_val]
    test_fires = fire_ids[n_train + n_val:]

    train_samples = [s for fid in train_fires for s in groups[fid]]
    val_samples = [s for fid in val_fires for s in groups[fid]]
    test_samples = [s for fid in test_fires for s in groups[fid]]

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "n_fires": {"train": len(train_fires), "val": len(val_fires),
                    "test": len(test_fires)},
    }
''')


# ============ scripts/02_inspect_data.py ============
write("scripts/02_inspect_data.py", '''"""Inspect the CaBuAr HDF5 structure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyrosegnet.data.inspect import (
    find_hdf5_file, summarize_dataset, inspect_one_sample,
    analyze_normalization, visualize_sample,
)
from pyrosegnet.data.splits import split_by_fire
from pyrosegnet.config import load_config

if __name__ == "__main__":
    config = load_config()
    hdf5_path = find_hdf5_file([Path(config["data"]["raw_dir"])])
    if hdf5_path is None:
        print("No HDF5 file found. Run scripts/01_download_data.py first.")
        sys.exit(1)

    # Step 1: dataset summary
    sample_keys, fire_uuids = summarize_dataset(hdf5_path)

    # Step 2: one sample in detail
    inspect_one_sample(hdf5_path, sample_keys[0])

    # Step 3: NBR sanity check
    analyze_normalization(hdf5_path, n_samples=10)

    # Step 4: split check
    print(f"\\n{'=' * 70}")
    print("SPLIT PREVIEW (by fire event)")
    print("=" * 70)
    splits = split_by_fire(sample_keys)
    for name, samples in splits.items():
        if name == "n_fires":
            continue
        print(f"  {name}: {len(samples)} samples across "
              f"{splits['n_fires'][name]} fires")

    # Step 5: save visualization of first sample
    out = Path(config["outputs"]["figure_dir"]) / "sample_inspection.png"
    visualize_sample(hdf5_path, sample_keys[0], save_path=str(out))
''')


print("\\nFiles updated.")
print("Next: python scripts/02_inspect_data.py")