# Project Roadmap

This roadmap organizes future engineering improvements into incremental releases. Each planned task should be implemented, reviewed, verified, and logged as a separate logical change whenever practical.

## v1.0.x — Project Foundation

**Goal:** Establish a documented and repeatable development workflow.

- **Completed:** Initialize the Git repository and GitHub synchronization workflow.
- **Completed:** Establish the `main` and `improvement-v1` branch strategy.
- **Completed:** Verify VS Code, GitHub, and Google Colab synchronization.
- **Completed:** Define Mac and Google Drive dataset locations.
- **Completed:** Create project planning and change-log documentation.
- **Completed:** Document repository structure, architecture, and pipeline behavior.
- **Planned:** Add environment and dependency documentation.
- **Planned:** Add clear local and Colab setup instructions to the README.
- **Planned:** Verify every existing entry point against the configured dataset.

## v1.1.x — Reproducibility and Experiment Logging

**Goal:** Make every training run identifiable and reproducible.

- **Planned:** Centralize dataset paths and shared parameters in one configuration source.
- **Planned:** Add complete deterministic seeding for Python, NumPy, and PyTorch.
- **Planned:** Save one configuration snapshot for every run.
- **Planned:** Record the date, Git commit, model architecture, dataset path, and class ordering.
- **Planned:** Write per-epoch training and validation metrics to CSV.
- **Planned:** Create one timestamped directory per experiment.
- **Planned:** Save checkpoint metadata with model weights.
- **Planned:** Save optimizer state, epoch number, and best metric in checkpoints.
- **Planned:** Record a compact JSON summary for each completed experiment.

## v1.2.x — Training and Evaluation Reliability

**Goal:** Correct training and evaluation behavior before introducing more complex models.

- **Planned:** Align training, validation, evaluation, and prediction preprocessing.
- **Planned:** Establish one authoritative class-to-index mapping.
- **Planned:** Correct input scaling and loss usage in the ResNet training loop.
- **Planned:** Ensure validation runs without augmentation or gradient calculation.
- **Planned:** Add model evaluation mode during validation.
- **Planned:** Use the configured weight decay consistently.
- **Planned:** Add a learning-rate scheduler.
- **Planned:** Add validation-based early stopping.
- **Planned:** Save the best validation checkpoint instead of only the final epoch.
- **Planned:** Calculate macro-F1, precision, recall, and per-class support.
- **Planned:** Generate and save a confusion matrix.

## v1.3.x — Data Quality and Model Optimization

**Goal:** Improve generalization through reliable data handling and measured experimentation.

- **Planned:** Validate image readability, dimensions, labels, and duplicate filenames before training.
- **Planned:** Replace silent blank-image fallbacks with explicit data-quality reporting.
- **Planned:** Review and document the label-generation process.
- **Planned:** Remove unintended label corruption or noise from the label-generation workflow.
- **Planned:** Adopt a leakage-resistant video-grouped train/validation split.
- **Planned:** Version and reuse a fixed split manifest.
- **Planned:** Evaluate class-weighted loss or balanced sampling for class imbalance.
- **Planned:** Compare clinically plausible augmentation strategies.
- **Planned:** Run a controlled hyperparameter search.
- **Planned:** Compare experiments using the same split and primary metric.

## v1.4.x — Testing and Pipeline Consolidation

**Goal:** Protect the required interface and reduce duplicated functionality.

- **Planned:** Add unit tests for preprocessing, class mapping, metrics, and split logic.
- **Planned:** Add integration tests for training, prediction, evaluation, and preflight execution.
- **Planned:** Test prediction CSV schema, filename coverage, and duplicate handling.
- **Planned:** Introduce a shared model-artifact contract.
- **Planned:** Connect the selected training checkpoint to prediction and evaluation.
- **Planned:** Consolidate duplicated dataset and preprocessing implementations.
- **Planned:** Connect YAML configuration files to executable code or retire them.
- **Planned:** Separate active code clearly from reference-only legacy code.
- **Planned:** Correct `run_all.sh` so it trains and evaluates the same model pipeline.
- **Planned:** Add lightweight continuous integration for fast tests and static checks.

## v2.0.x — Model and Deployment Improvements

**Goal:** Establish a maintainable production-quality model pipeline after the baseline is trustworthy.

- **Planned:** Select the best validated architecture using reproducible experiment results.
- **Planned:** Evaluate pretrained ResNet and EfficientNet fine-tuning strategies.
- **Planned:** Consider progressive backbone unfreezing and discriminative learning rates.
- **Planned:** Evaluate model calibration and prediction confidence.
- **Planned:** Add inference performance benchmarks for CPU and GPU environments.
- **Planned:** Package the selected checkpoint with preprocessing and label metadata.
- **Planned:** Improve error handling and operational diagnostics in prediction.
- **Planned:** Add a reproducible deployment or submission build process.
- **Planned:** Document model limitations, intended use, and dataset assumptions.
- **Planned:** Preserve backward compatibility with the required prediction CLI and CSV contract.

## Roadmap Principles

- Engineering correctness takes priority over model complexity.
- Validation methodology must be trustworthy before comparing accuracy improvements.
- Every model comparison should use the same data split and primary metric.
- Each logical change should have its own Git commit and change-log entry.
- Existing interfaces should remain compatible unless a breaking change is explicitly justified.
- Future scope may be reordered when experimental evidence or assessment requirements change.
