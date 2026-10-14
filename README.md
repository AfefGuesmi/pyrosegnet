# PyroSegNet
### Lightweight Deep Learning for Wildfire Burned-Area Segmentation

![Wildfire](https://www.brailleinstitute.org/wp-content/uploads/2025/01/photo-7752746_1280.jpg)

PyroSegNet is a research project focused on **pixel-level wildfire burned-area segmentation** using paired pre-fire and post-fire Sentinel-2 satellite imagery.

The project compares a classical spectral-index baseline based on the differenced Normalized Burn Ratio (dNBR) with a lightweight dual-encoder convolutional neural network (CNN). By learning from changes between pre-fire and post-fire imagery, PyroSegNet aims to identify burned areas at the pixel level.

The implementation includes dataset inspection, spectral-index calculation, official fold-based data splitting, model training, and quantitative evaluation.

**Key features**

- Paired pre-fire and post-fire Sentinel-2 imagery.
- Classical dNBR threshold baseline.
- Lightweight dual-encoder CNN with multi-scale temporal fusion.
- Official fold-based training, validation, and test splits.
- CPU-friendly development with resized images and GPU-oriented full-resolution training.
- Evaluation using precision, recall, F1-score, Intersection over Union (IoU), Dice, and balanced accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Dataset Limitations](#dataset-limitations)
- [Data Splits](#data-splits)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Download the Dataset](#download-the-dataset)
- [Run the Experiments](#run-the-experiments)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Limitations and Future Work](#limitations-and-future-work)
- [Dataset Reference](#dataset-reference)

---

## Overview

Wildfires can significantly alter vegetation and surface characteristics. Satellite imagery provides a way to analyze these changes by comparing observations acquired before and after a fire.

PyroSegNet investigates burned-area segmentation using two approaches:

1. **Spectral-index baseline:** applies dNBR thresholding to identify potential burned areas.
2. **Deep-learning approach:** processes paired pre-fire and post-fire images using a dual-encoder CNN and predicts a binary segmentation mask.

The goal is to evaluate whether learned temporal features improve burned-area delineation compared with a simple spectral-index threshold.

## Results

PyroSegNet was evaluated against a dNBR threshold baseline on the same held-out test set: **fold 0, containing 72 samples from 68 unique fires**. The baseline threshold was selected using the validation set, fold 1.

The metrics below are the supplied experimental results. They should be traceable to the saved evaluation files and training logs.

### Test-set performance

| Method | Precision | Recall | F1-score | IoU | Dice | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| dNBR threshold (≥ 0.28) | 0.4855 | 0.6706 | 0.5633 | 0.3920 | 0.5633 | 0.7966 |
| **PyroSegNet (128 × 128, 50 epochs)** | **0.7328** | **0.7002** | **0.7161** | **0.5578** | **0.7161** | **0.8362** |

Higher values indicate better performance for all metrics shown.

### Improvement over the baseline

| Metric | Absolute improvement | Relative improvement |
|---|---:|---:|
| Precision | +0.2473 | +51.0% |
| Recall | +0.0296 | +4.4% |
| F1-score | +0.1528 | +27.1% |
| IoU | +0.1658 | +42.3% |
| Balanced accuracy | +0.0396 | +5.0% |

Relative improvements are calculated with respect to the baseline metric.

### Interpretation

- **Higher precision:** PyroSegNet achieves 0.7328 compared with 0.4855 for dNBR thresholding. This means that a larger proportion of the model's predicted burned pixels are correct.
- **Improved recall:** Recall increases from 0.6706 to 0.7002, indicating that the model identifies a slightly larger proportion of reference burned pixels.
- **Better spatial overlap:** IoU increases from 0.3920 to 0.5578, showing improved overlap between predicted and reference burned areas.
- **Higher F1-score:** F1 increases from 0.5633 to 0.7161, reflecting a better balance between precision and recall.

Overall, the supplied results favor PyroSegNet over the tested single-threshold dNBR baseline on this test set. These findings do not establish that the model will outperform other methods on different datasets or geographic regions.

### Training-run comparison

Two training runs using the same architecture were reported without a fixed random seed.

| Run | Epochs | Best validation IoU | Test IoU |
|---|---:|---:|---:|
| A | 10 | 0.4408 | 0.5577 |
| B | 50 | 0.5154 | 0.5578 |

The test IoU values differ by only 0.0001. Although this indicates similar test IoU for these two runs, two runs are insufficient to establish general training stability or statistical significance.

### Training configuration

The following configuration was reported for Run B.

| Parameter | Value |
|---|---|
| Input resolution | 128 × 128 |
| Original patch resolution | 512 × 512 |
| Batch size | 4 |
| Maximum epochs | 50 |
| Best checkpoint epoch | 31 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.00001 |
| Loss function | 0.5 × BCEWithLogitsLoss + 0.5 × Dice loss |
| Positive-class weight | 3.0 |
| Hardware | Local laptop CPU |
| Reported training time | 94 minutes |

Resizing the input images reduces computational requirements but also removes spatial detail compared with full-resolution training.

### Scope of the comparison

The baseline uses a single dNBR threshold selected on the validation set. It is a simple and interpretable reference rather than an exhaustive comparison against traditional machine-learning methods.

Future experiments should compare PyroSegNet with stronger baselines, including Random Forest models using spectral features, alternative threshold-selection strategies, and full-resolution training.

---

## Dataset

**Dataset:** [CaBuAr — California Burned Areas](https://huggingface.co/datasets/DarthReca/california_burned_areas)

**Inspected file:** `normalized/pacthed/burned_512x512.hdf5`

**Reported file size:** approximately 3.59 GB.

The following statistics describe the inspected HDF5 file.

| Property | Value |
|---|---:|
| Total top-level groups | 727 |
| Complete samples containing `pre_fire`, `post_fire`, and `mask` | **354** |
| Incomplete groups missing `pre_fire` | 373 |
| Spectral bands | 12 |
| Image data type | `uint8` |
| Stored image value range | 0–255 |
| Patch dimensions | 512 × 512 pixels |
| Mask values | `{0, 1}` |
| Official folds | 5 |

### Spectral band order

The verified channel order is:

```text
B01, B02, B03, B04, B05, B06,
B07, B08, B8A, B09, B11, B12
```

Important bands used for spectral-index calculations:

| Band | Description | Zero-based index |
|---|---|---:|
| `B04` | Red | 3 |
| `B08` | Near-infrared (NIR) | 7 |
| `B11` | Short-wave infrared 1 (SWIR1) | 10 |
| `B12` | Short-wave infrared 2 (SWIR2) | 11 |

The imagery provides twelve spectral channels per observation, while the binary mask provides the segmentation target.

## Dataset Limitations

The inspected normalized HDF5 file contains **373 groups without a `pre_fire` dataset**.

These groups cannot be used for paired pre-fire and post-fire change detection in the current workflow. Consequently, only 354 of the 727 groups are complete samples.

A separate file, `raw/patched/512x512.hdf5`, exists in the dataset repository. It may be worth inspecting to determine whether additional samples can be recovered, but its completeness must be verified first.

## Data Splits

PyroSegNet uses the official dataset fold assignments rather than creating random splits.

| Split | Fold(s) | Samples | Fires |
|---|---|---:|---:|
| Test | 0 | 72 | 68 |
| Validation | 1 | 60 | 68 |
| Training | 2, 3, 4 | 222 | 204 |
| **Total** | **0–4** | **354** | — |

The current dataset inspection found no fire event appearing in more than one fold.

The validation split is used for model selection and threshold tuning. The test split is reserved for final evaluation to reduce the risk of data leakage.

---

## Project Structure

```text
pyrosegnet/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_inspect_data.py
│   ├── 03_run_threshold_baseline.py
│   ├── 05_train_pyrosegnet.py
│   └── 06_local_train.py
├── src/
│   └── pyrosegnet/
│       ├── config.py
│       ├── data/
│       │   ├── dataset.py
│       │   ├── download.py
│       │   ├── inspect.py
│       │   └── splits.py
│       ├── features/
│       │   ├── spectral_indices.py
│       │   └── pixel_features.py
│       ├── models/
│       │   ├── pyrosegnet.py
│       │   └── losses.py
│       ├── training/
│       │   └── train.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── traditional/
│       │   └── threshold.py
│       └── visualization/
│           └── plots.py
└── tests/
```

### Main components

| Component | Responsibility |
|---|---|
| `dataset.py` | Loads paired imagery and masks |
| `splits.py` | Creates official fold-based splits |
| `spectral_indices.py` | Calculates spectral indices |
| `pyrosegnet.py` | Defines the dual-encoder CNN |
| `losses.py` | Implements BCE and Dice loss |
| `train.py` | Handles model training and early stopping |
| `metrics.py` | Computes segmentation metrics |
| `threshold.py` | Implements the dNBR baseline |
| `plots.py` | Creates visualizations and error maps |

---

## Installation

Run the following commands from the repository root.

### 1. Create the environment

```bash
conda create -n pyrosegnet python=3.11 -y
conda activate pyrosegnet
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip

python -m pip install \
    numpy pandas scikit-learn matplotlib seaborn tqdm PyYAML

python -m pip install \
    h5py hdf5plugin huggingface_hub pytest
```

### 3. Install PyTorch

For a CPU-only environment:

```bash
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA GPU training, install the PyTorch build compatible with your system and CUDA version. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Download the Dataset

Run the download script:

```bash
python scripts/01_download_data.py
```

The expected local dataset path is:

```text
data/raw/512x512.hdf5
```

The file is approximately 3.6 GB. Ensure that sufficient disk space is available before downloading.

---

## Run the Experiments

Run commands from the project root with the `pyrosegnet` environment activated.

### Step 1 — Inspect the dataset

```bash
python scripts/02_inspect_data.py
```

This script inspects the HDF5 structure, examines sample metadata, performs NBR/dNBR sanity checks, and previews the official data splits.

Expected figure:

```text
outputs/figures/sample_inspection.png
```

### Step 2 — Run the dNBR baseline

```bash
python scripts/03_run_threshold_baseline.py
```

The described baseline workflow:

- Evaluates 51 dNBR thresholds between 0.05 and 0.55 on the validation split.
- Selects the threshold that maximizes validation F1-score.
- Evaluates the selected threshold on the held-out test split.
- Saves the threshold-selection results, test metrics, and example visualizations.

Expected outputs:

```text
outputs/results/threshold_selection.csv
outputs/results/threshold_test_metrics.csv
outputs/figures/threshold_pred_{0,1,2}.png
```

These paths describe expected artifacts, not a claim that the scripts have already run.

### Step 3 — Train PyroSegNet on CPU

For local development, use resized images:

```bash
python scripts/06_local_train.py \
    --img_size 128 \
    --epochs 10 \
    --batch_size 4
```

Resizing the original 512 × 512 patches to 128 × 128 reduces computational requirements but also reduces spatial detail.

| Argument | Default | Description |
|---|---:|---|
| `--img_size` | 128 | Input image resolution |
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 4 | Number of samples per batch |
| `--lr` | `1e-4` | Learning rate |
| `--no_augment` | Disabled | Disables data augmentation |

Check supported arguments before changing the configuration:

```bash
python scripts/06_local_train.py --help
```

Expected outputs:

```text
outputs/checkpoints/pyrosegnet_local_best.pth
outputs/results/pyrosegnet_local_history.csv
outputs/results/pyrosegnet_local_test_metrics.csv
```

Actual runtime depends on hardware, memory, storage, and data-loading performance.

### Step 4 — Train PyroSegNet on GPU

For full-resolution training:

```bash
python scripts/05_train_pyrosegnet.py
```

Configure the batch size and other hyperparameters according to the options supported by the script and the available GPU memory.

The full-resolution pipeline is intended for GPU-capable environments, such as a Kaggle notebook with a compatible GPU.

---

## Model Architecture

PyroSegNet uses a dual-encoder convolutional neural network to learn from paired pre-fire and post-fire imagery.

The encoders extract features from both images. A temporal fusion module combines their representations at multiple scales, and a U-Net-style decoder generates the segmentation logits.

```text
       Pre-fire image                    Post-fire image
       (12 × H × W)                      (12 × H × W)
             │                                 │
             ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │ Pre-fire Encoder│               │Post-fire Encoder│
    │                 │               │                 │
    │ ConvBlocks      │               │ ConvBlocks      │
    │ 32 → 64 → 128   │               │ 32 → 64 → 128   │
    │        → 256    │               │        → 256    │
    └────────┬────────┘               └────────┬────────┘
             │                                 │
             └───────────────┬─────────────────┘
                             ▼
                ┌────────────────────────┐
                │ Multi-scale Temporal   │
                │ Fusion                 │
                │ Concatenation and      │
                │ Feature Differences    │
                └────────────┬───────────┘
                             ▼
                ┌────────────────────────┐
                │ U-Net-style Decoder    │
                │ Upsampling and Skip    │
                │ Connections            │
                └────────────┬───────────┘
                             ▼
                ┌────────────────────────┐
                │ Segmentation Logits    │
                │       (1 × H × W)      │
                └────────────────────────┘
```

### Architectural design

- **Dual encoders:** process pre-fire and post-fire images through separate feature extractors.
- **Temporal differences:** represent changes between the two observations.
- **Feature concatenation:** combines complementary information from both dates.
- **Multi-scale fusion:** integrates changes at different feature resolutions.
- **Skip connections:** preserve spatial information during decoding.
- **Lightweight design:** the project estimates approximately 3.4 million parameters; verify the count against the implemented model before treating it as definitive.

The model produces segmentation logits that can be converted into a binary burned-area prediction using the appropriate activation and thresholding procedure.

## Evaluation Metrics

PyroSegNet uses standard binary segmentation metrics to compare predicted masks against reference masks.

| Metric | Purpose |
|---|---|
| Precision | Proportion of predicted burned pixels that are correct |
| Recall | Proportion of reference burned pixels that are detected |
| F1-score | Harmonic mean of precision and recall |
| Intersection over Union (IoU) | Overlap between predicted and reference regions |
| Dice coefficient | Similarity between predicted and reference masks |
| Balanced accuracy | Average recall across the positive and negative classes |

F1-score and Dice are mathematically equivalent for the same binary confusion counts when calculated using the same convention. Their matching values in the reported results are therefore expected.

## Outputs

Generated artifacts are organized under `outputs/`.

| Location | Contents |
|---|---|
| `outputs/checkpoints/` | Saved model weights |
| `outputs/results/` | Training history, threshold selections, and evaluation metrics |
| `outputs/figures/` | Inspection plots, predictions, and error maps |

Downloaded data, cached files, checkpoints, and experiment outputs should generally be excluded from Git version control when they are large or reproducible.

## Reproducibility

To make experiments easier to reproduce:

- Use the official fold assignments.
- Keep the test split separate from model selection and threshold tuning.
- Record the configuration and hyperparameters for each run.
- Verify the HDF5 path, group structure, and fold metadata.
- Confirm that command-line arguments match the current scripts.
- Save evaluation metrics and training history for every run.
- Report metrics only after verifying the corresponding evaluation files.
- Use repeated runs with controlled random seeds when assessing stability.

## Limitations and Future Work

Current limitations include:

- Missing pre-fire imagery in a substantial portion of the inspected normalized HDF5 file.
- A limited number of complete paired samples.
- Reduced spatial detail when training on resized images.
- A baseline comparison limited to a single dNBR threshold.
- Only two reported training runs, which are insufficient to establish statistical significance or general stability.

Potential future improvements include:

- Inspecting the alternative HDF5 variant for additional usable samples.
- Evaluating stronger traditional machine-learning baselines.
- Training and evaluating at full resolution.
- Conducting repeated experiments with controlled random seeds.
- Reporting confidence intervals and per-fire performance.
- Evaluating generalization on independent geographic regions or datasets.

## Dataset Reference

The dataset is available at:

[DarthReca/california_burned_areas — Hugging Face](https://huggingface.co/datasets/DarthReca/california_burned_areas)

Refer to the dataset page for current metadata, licensing, and citation requirements before redistributing data or publishing research based on it.
