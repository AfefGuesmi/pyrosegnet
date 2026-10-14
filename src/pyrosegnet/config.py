"""Load and provide access to the project configuration."""
from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _PROJECT_ROOT


def load_config(config_path=None):
    """Load the YAML config and convert relative paths to absolute."""
    if config_path is None:
        config_path = _PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key in ("raw_dir", "hdf5_file", "processed_dir"):
        p = Path(config["data"][key])
        if not p.is_absolute():
            config["data"][key] = str(_PROJECT_ROOT / p)
    for key in ("checkpoint_dir", "prediction_dir", "figure_dir", "result_dir"):
        p = Path(config["outputs"][key])
        if not p.is_absolute():
            config["outputs"][key] = str(_PROJECT_ROOT / p)
    return config
