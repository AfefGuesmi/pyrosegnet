"""Inspect the CaBuAr HDF5 structure."""
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
    print(f"\n{'=' * 70}")
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
