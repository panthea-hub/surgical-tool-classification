# Repository Structure

This document describes the layout of the surgical tool classification repository and the purpose of its major directories and files.

## Directory Tree

```text
surgical-tool-classification/
│
├── configs/
│   ├── default.yaml
│   └── best_config_v3.yaml
│
├── checkpoints/                 # Generated locally; excluded from Git
├── data/                        # Dataset location expected by existing code; excluded from Git
│
├── docs/
│   ├── change_log.md
│   ├── project_architecture.md
│   ├── project_plan.md
│   └── repository_structure.md
│
├── experiments/
│   └── exp02_efficientnet/
│       ├── model_def.py
│       ├── run_experiment.py
│       └── results.json
│
├── helpers/
│   ├── __init__.py
│   ├── image_helpers.py
│   └── metrics.py
│
├── legacy/
│   ├── cnn_baseline_v2.py
│   └── old_train.py
│
├── notebooks/
│   └── EDA_final.ipynb
│
├── splits/
│   └── split_seed42.json
│
├── README.md
├── config.py
├── contract.py
├── check_submission.py
├── data_utils.py
├── evaluate_model.py
├── labels.csv
├── predict.py
├── run_all.sh
├── stats.json
├── train.py
├── train_v2.py
├── utils.py
└── .gitignore
```

## Directories

### `configs/`

Contains YAML files describing alternative training parameter sets. These configurations are currently reference artifacts and are not loaded by the active training scripts.

- `default.yaml` contains an earlier default parameter set for training.
- `best_config_v3.yaml` records a manually selected parameter set and experiment notes.

### `checkpoints/`

Stores generated model weights such as `model_best.pt`, `resnet18_final.pt`, and `efficientnet_b0.pt`. The directory is excluded from Git, so checkpoints must be generated or transferred separately.

### `data/`

Represents the repository-local dataset path expected by several existing scripts. The actual dataset is excluded from Git and may instead be stored at the configured Mac or Google Drive location.

### `docs/`

Contains the project's engineering and planning documentation.

- `change_log.md` is the official chronological record of engineering changes.
- `project_architecture.md` explains components, dependencies, workflows, and technical debt.
- `project_plan.md` defines the project objective, success criteria, and development strategy.
- `repository_structure.md` documents the repository layout.

### `experiments/`

Contains model experiments that are kept separate from the main training and inference paths.

#### `experiments/exp02_efficientnet/`

Contains the EfficientNet-B0 experiment.

- `model_def.py` constructs the EfficientNet-B0 model and replaces its classification head.
- `run_experiment.py` defines experiment transforms, training, evaluation, result writing, and checkpoint saving.
- `results.json` stores the experiment's recorded model name, final accuracy, and epoch count.

### `helpers/`

Contains shared helper functions, primarily for evaluation.

- `__init__.py` marks the directory as a Python package.
- `image_helpers.py` provides OpenCV-based image preprocessing, dataset-statistics loading, and evaluation class names.
- `metrics.py` implements accuracy and confidence helper functions.

### `legacy/`

Contains superseded code retained for reference or compatibility. Despite its location, `cnn_baseline_v2.py` still defines the model used by the active prediction and evaluation scripts.

- `cnn_baseline_v2.py` defines and trains the custom `SmallCNN` model and saves `model_best.pt`.
- `old_train.py` contains video-grouped splitting logic intended to reduce leakage between related surgical frames.

### `notebooks/`

Contains exploratory and data-preparation notebooks. `EDA_final.ipynb` examines class counts and generates the master `labels.csv` file.

### `splits/`

Contains saved dataset-partition metadata. `split_seed42.json` records a deterministic train/validation split, although the current `train.py` recomputes its own split.

### Generated directories

The `.gitignore` also excludes `runs/`, `logs/`, and `outputs/`. These locations are intended for generated experiment logs and outputs, but the current training scripts do not consistently populate them.

## Root Files

### `README.md`

Provides the project name, assessment context, repository status, and basic branch information.

### `config.py`

Defines the settings used by `train.py`, including learning rate, batch size, epoch count, class count, image size, weight decay, and dataset root.

### `contract.py`

Defines the shared submission contract. It locates `predict.py`, runs prediction as a subprocess, discovers expected images and classes, and validates the output CSV schema.

### `check_submission.py`

Runs the local submission preflight check against the validation dataset. It verifies that prediction completes and produces a structurally valid CSV without evaluating model accuracy.

### `data_utils.py`

Defines the prediction class-name list and several dataset implementations from different stages of the project. Only part of this module is currently used by the active pipeline.

### `evaluate_model.py`

Loads the legacy `SmallCNN` checkpoint and evaluates it against the validation folders. It prints overall and per-class accuracy to the console.

### `labels.csv`

Maps image filenames to tool-class labels. It is used by the legacy CNN training pipeline.

### `predict.py`

Provides the required inference command-line interface. It loads `checkpoints/model_best.pt`, predicts each PNG image, and writes a `filename,predicted_class` CSV.

### `run_all.sh`

Runs the EfficientNet experiment, evaluates the legacy production checkpoint, and prints selected values from `config.py`. It currently combines outputs from different model pipelines.

### `stats.json`

Stores dataset channel means, standard deviations, and the number of images used to compute them. `train_v2.py` uses these values for normalization.

### `train.py`

Implements the main ResNet-18 training path. It loads and splits images, trains a new classifier head on a frozen pretrained backbone, prints epoch metrics, and saves `resnet18_final.pt`.

### `train_v2.py`

Wraps `train.py` with dataset-specific normalization and alternate training parameters. It writes to the same ResNet checkpoint path as `train.py`.

### `utils.py`

Provides general helper functions for random seeding, device selection, class-name discovery, and trainable-parameter counting. These functions are not currently integrated consistently into the active scripts.

### `.gitignore`

Excludes datasets, checkpoints, generated outputs, logs, temporary files, Python cache files, notebook checkpoints, and common macOS artifacts from version control.
