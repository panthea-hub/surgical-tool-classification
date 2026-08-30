# Project Change Log

This document records significant engineering, workflow, data, modeling, and infrastructure changes made to the surgical tool classification project. Entries are organized chronologically, with the newest entry first.

## Versioning Policy

- **Major version (X.0.0):** Significant milestones or architectural changes.
- **Minor version (X.Y.0):** New features or meaningful improvements.
- **Patch version (X.Y.Z):** Small fixes, documentation updates, refactoring, bug fixes, parameter tuning, or non-breaking improvements.

Each logged change should correspond to a single logical Git commit whenever practical.

## Guidelines

Record changes that affect:

- Model architecture, training, evaluation, or inference
- Data organization, validation, preprocessing, or splitting
- Hyperparameters and experiment configuration
- Reproducibility and experiment tracking
- Dependencies, environments, or execution workflows
- Prediction-interface compatibility
- Testing, documentation, or repository structure
- CI/CD, Git, GitHub, VS Code, or Colab workflows

Each logical improvement should have its own entry and Git commit. Minor spelling or formatting corrections do not require an entry unless they materially change project documentation.

## Change Entry Template

### <date> — <version> — <short-title>

- **Git Commit:** `<commit-hash>`
- **Status:** <status>
- **Files Modified:**
  - `<file-path>`
- **Description:**
  - <description>
- **Reason:** <reason>
- **Verification:**
  - <verification-step>
- **Expected Impact:** <expected-impact>
- **AI Assistance Used:** <none-or-tool-name>
- **Next Step:** <next-step>

## Changes

### 2026-08-30 — v0.2.0 — Automatic experiment logging

- **Git commit:** `<pending>`
- **Status:** Implemented; full training-run verification pending
- **Files modified:**
  - `train.py`
  - `docs/change_log.md`
- **Description:**
  - Added automatic experiment logging to keep track of every training run. Each run is appended to `runs/training_history.csv` and records the training parameters from `config.py`, `train.py`, and `train_v2.py`, along with the final results, start and end time, run duration, and checkpoint location. This provides a single place to compare experiments and makes the training process reproducible.
- **Reason for the change:** Provide one durable, append-only history for comparing completed training runs and tracing their effective inputs and results.
- **Verification:**
  - Verified that `train.py` passes Python syntax parsing.
  - Verified that the change introduces no whitespace errors.
  - Full CSV creation and row-content verification will be completed after the next training run.
- **Expected impact:** Improves experiment traceability, comparison, and reproducibility without changing the existing training logic, hyperparameters, console output, or checkpoint behavior.
- **AI assistance used:** Codex
- **Next step:** Run `train.py` or `train_v2.py` and verify the generated `runs/training_history.csv` row.

### 2026-08-30 — v0.1.0 — Initial project setup

- **Git commit:** `0631344`
- **Status:** Completed
- **Files modified:**
  - `README.md`
  - `docs/project_plan.md`
  - Repository and Git configuration
- **Description:**
  - Created and initialized the Git repository.
  - Established the `main` and `improvement-v1` branch strategy.
  - Configured the VS Code development environment.
  - Established the Google Colab execution workflow.
  - Verified synchronization between the local repository, GitHub, and Colab.
  - Defined dataset locations for Mac and Google Drive environments.
  - Created the README and project planning documentation.
- **Reason for the change:** Establish a consistent, reviewable, and reproducible development workflow before modifying the machine learning solution.
- **Verification:**
  - Verified Git repository initialization.
  - Verified VS Code → GitHub workflow.
  - Verified GitHub → Colab synchronization.
  - Verified Colab → GitHub synchronization.
- **Expected impact:** Provides a reliable foundation for incremental development, Git-based review, local editing, and GPU-backed Colab experiments.
- **AI assistance used:** Codex
- **Next step:** Connect Colab to the Google Drive dataset and run the existing training, prediction, and evaluation scripts.
