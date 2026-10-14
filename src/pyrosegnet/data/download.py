"""Download the CaBuAr dataset from Hugging Face."""
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from pyrosegnet.config import load_config


REPO_ID = "DarthReca/california_burned_areas"


def find_hdf5_in_repo():
    """List HDF5 files in the Hugging Face repo and pick the 512x512 one."""
    files = list_repo_files(REPO_ID, repo_type="dataset")
    hdf5_files = [f for f in files if f.endswith(".hdf5") or f.endswith(".h5")]
    if not hdf5_files:
        raise RuntimeError(
            f"No HDF5 files found in {REPO_ID}. Files present: {files[:20]}"
        )
    print(f"HDF5 files in repo: {hdf5_files}")
    for f in hdf5_files:
        if "512" in f:
            return f
    return hdf5_files[0]


def download_cabuar(config=None):
    """Download the CaBuAr HDF5 file to the configured location."""
    if config is None:
        config = load_config()

    target = Path(config["data"]["hdf5_file"])
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        size_gb = target.stat().st_size / 1e9
        print(f"Dataset already exists at {target} ({size_gb:.2f} GB)")
        return target

    filename = find_hdf5_in_repo()
    print(f"Downloading {filename} from {REPO_ID}...")
    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )
    print(f"Downloaded to HF cache: {cached}")
    print(f"Copying to project location: {target}")
    shutil.copy(cached, target)

    size_gb = target.stat().st_size / 1e9
    print(f"Done. File size: {size_gb:.2f} GB")
    return target


if __name__ == "__main__":
    download_cabuar()
