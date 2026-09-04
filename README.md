# Surgical Tool Classification

<p align="center">
  <b>Deep learning pipeline for classification of surgical tools in laparoscopic images</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-ResNet18-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/Model-ResNet18-green" alt="Model">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" alt="Status">
</p>

---

## Overview

This repository provides an end-to-end image classification pipeline for identifying surgical tools in laparoscopic images.

The model classifies images into four categories:

- **Clipper**
- **Grasper**
- **Hook**
- **Scissor**

The final model uses a pretrained **ResNet18** with partial fine-tuning of the final residual block (`layer4`) and classifier head.

---

## Model Performance

### Final held-out accuracy: **88.5%**

The final model was evaluated on the original held-out validation dataset, which was kept separate from model training and model selection.

| Class | Baseline | Final Model |
|---|---:|---:|
| Clipper | 81.6% | **89.8%** |
| Grasper | 87.1% | **93.1%** |
| Hook | 92.0% | **96.0%** |
| Scissor | 37.0% | **40.7%** |
| **Overall** | **83.0%** | **88.5%** |

**Final approach:** Partial fine-tuning of ResNet18 `layer4` and the classifier head.

For experiment details and model comparisons, see [`docs/model_baseline.md`](docs/model_baseline.md).

---

## Pipeline

```text
Surgical Images
      │
      ▼
   Training
   ResNet18
      │
      ▼
 Best Checkpoint
      │
      ├──────────────► Evaluation
      │
      ▼
  Prediction
      │
      ▼
 predictions.csv
      │
      ▼
Submission / Preflight Check
```

The training pipeline records the newly generated checkpoint so that evaluation and prediction use the same trained model.

---

## Dataset

The expected dataset structure is:

```text
data/
└── cholec-tinytools/
    ├── train/
    │   ├── clipper/
    │   ├── grasper/
    │   ├── hook/
    │   └── scissor/
    │
    └── validation/
        ├── clipper/
        ├── grasper/
        ├── hook/
        └── scissor/
```

The original `train/` directory is used for model development.

A stratified portion of the training data is used for training-time validation and model selection. The original `validation/` directory remains held out and is used only for final model evaluation.

---

# Installation

## Requirements

The project uses Python and PyTorch.

Main dependencies include:

- Python 3.11
- PyTorch
- torchvision
- NumPy
- scikit-learn
- Pillow

Create and activate a Python environment before installing the required packages.

```bash
conda create -n surgical-tools python=3.11
conda activate surgical-tools
```

Install the active pipeline dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

## Quick Start

1. Create and activate the environment:

```bash
conda create -n surgical-tools python=3.11
conda activate surgical-tools
pip install -r requirements.txt
```

2. Run the complete pipeline:

```bash
./setup.sh /path/to/cholec-tinytools
```

The supplied dataset directory must contain `train/` and `validation/`. `setup.sh` validates the dataset path, sets `DATA_ROOT`, and launches `run_all.sh`.

## Training Configuration

The adopted training settings are not yet fully centralized in `config.py`:

- `train_v2.py` supplies `batch_size=8`, `epochs=10`, and nominal `lr=1e-3`.
- `train.py` uses separate learning rates for partial fine-tuning: `layer4=1e-4` and `fc=1e-3`.
- For a quick test, change the number of epochs in `train_v2.py`, not `config.py`.
- Centralizing these settings in `config.py` is listed as future work.

## Train

Run:

```bash
python train_v2.py
```

Training creates:

- a unique run ID
- per-epoch metrics
- a best-model checkpoint
- a training-history record

Model checkpoints are stored in:

```text
checkpoints/
```

Experiment metrics are stored in:

```text
runs/
```

---

## Evaluate

Evaluate a specific checkpoint:

```bash
python evaluate_model.py --checkpoint checkpoints/<checkpoint>.pt
```

Evaluation reports:

- overall accuracy
- per-class accuracy
- confusion matrix

Evaluation history is stored under:

```text
runs/
```

---

## Predict

The required prediction interface is:

```bash
python predict.py --data-dir <DIR> --out <CSV>
```

A specific checkpoint can also be supplied:

```bash
python predict.py \
    --checkpoint checkpoints/<checkpoint>.pt \
    --data-dir <DIR> \
    --out predictions.csv
```

The prediction pipeline uses the same image preprocessing and model architecture used during training.

---

## Run the Complete Pipeline

The complete workflow can be run with:

```bash
./run_all.sh
```

This executes:

```text
Training
   ↓
Evaluation
   ↓
Prediction
   ↓
Submission Check
```

The checkpoint generated during training is passed to both evaluation and prediction so that the complete run uses the same model.

---

# Experiment Tracking

Each training run receives a unique identifier.

```text
runs/
├── training_history.csv
├── evaluation_history.csv
└── <run_id>/
    └── metrics.csv
```

Checkpoints use unique filenames:

```text
checkpoints/
└── resnet18_<run_id>.pt
```

This prevents model checkpoints from being overwritten and links model artifacts to their corresponding experiment results.

---

# Model Development

A reproducible ResNet18 baseline achieved:

**83.0% held-out accuracy**

Several approaches were evaluated during model development.

| Experiment | Overall Accuracy | Scissor Accuracy | Decision |
|---|---:|---:|---|
| Stabilized ResNet18 baseline | 83.0% | 37.0% | Baseline |
| Class-weighted loss | 79.4% | 48.2% | Not adopted |
| Square-root class weighting | 81.6% | 40.7% | Not adopted |
| Partial ResNet18 fine-tuning | **88.5%** | **40.7%** | **Adopted** |

The final approach fine-tunes `layer4` and the classifier head while keeping earlier ResNet18 layers frozen.

Detailed experiment results are available in [`docs/model_baseline.md`](docs/model_baseline.md).

---

# Repository Structure

```text
surgical-tool-classification/
│
├── train.py
├── train_v2.py
├── evaluate_model.py
├── predict.py
├── check_submission.py
├── run_all.sh
├── config.py
├── contract.py
├── stats.json
│
├── checkpoints/
├── runs/
├── docs/
└── archive/
```

### Core Files

| File | Purpose |
|---|---|
| `train.py` | ResNet18 training pipeline |
| `train_v2.py` | Training entry point and experiment configuration |
| `evaluate_model.py` | Held-out model evaluation |
| `predict.py` | Required inference interface |
| `check_submission.py` | Prediction/preflight validation |
| `run_all.sh` | End-to-end pipeline |
| `setup.sh` | Dataset setup and end-to-end pipeline launcher |
| `config.py` | Shared configuration |
| `contract.py` | Candidate/grading interface contract |

Legacy and exploratory files are retained under `archive/` for reference but are not part of the active pipeline.

---

# Documentation

Additional project documentation is available under `docs/`.

- [`model_baseline.md`](docs/model_baseline.md) — baseline and model experiment results
- [`project_plan.md`](docs/project_plan.md) — development plan and implementation decisions
- [`repository_structure.md`](docs/repository_structure.md) — repository organization

---

# Reproducibility

The pipeline includes:

- deterministic random seeds
- stratified train/validation splitting
- independent held-out evaluation
- unique experiment IDs
- per-epoch metric logging
- best-model checkpoint selection
- checkpoint metadata
- training and evaluation history
- consistent preprocessing across training, evaluation, and inference

These features allow individual model runs and their corresponding results to be traced and reproduced.

---

## Final Model

**Architecture:** ResNet18  
**Training strategy:** Partial fine-tuning (`layer4` + classifier head)  
**Held-out accuracy:** **88.5%**  
**Classes:** Clipper, Grasper, Hook, Scissor

---

## Future Work

Potential extensions include:

- comparison of ImageNet and dataset-specific normalization
- additional minority-class strategies for the scissor class
- consolidated hyperparameter configuration
- evaluation of additional fine-tuning strategies
