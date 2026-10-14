# PyroSegNet

Burned-area segmentation from paired pre-fire and post-fire Sentinel-2 imagery.

## Overview

PyroSegNet is a research project comparing traditional remote-sensing methods
(spectral indices, Random Forest) with a lightweight dual-encoder CNN for
binary burned-area segmentation. It is inspired by the general idea of paired
pre/post-fire analysis but uses a simplified architecture optimized for small
datasets.

## Dataset

CaBuAr (California Burned Areas): 424 pre/post-fire Sentinel-2 pairs,
12 spectral bands, 512x512 pixels, binary burn masks.

## Project structure

- `src/pyrosegnet/` - main package
- `scripts/` - runnable entry points
- `configs/` - YAML configuration
- `tests/` - unit tests

## Setup
conda create -n pyrosegnet python=3.11 -y
conda activate pyrosegnet
pip install -r requirements.txt
python scripts/01_download_data.py
python scripts/02_inspect_data.py
python scripts/03_run_threshold_baseline.py

text

## Reproducibility

- Seed: 42
- Splits: CaBuAr provided folds (train=[0,1,2], val=[3], test=[4])
- Config: `configs/default.yaml`

## License

MIT (code). Dataset license: CDLA-Permissive-2.0.
