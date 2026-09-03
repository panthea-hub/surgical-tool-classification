# Project Plan

## Objective

Improve the performance, robustness, and maintainability of a surgical tool image classification pipeline while preserving the required inference interface.

## Success Criteria

- Improve macro-F1 score on the validation set.
- Preserve compatibility with the required prediction contract.
- Improve code quality and reproducibility.
- Document all experiments and engineering decisions.

## Development Strategy

1. Review the existing implementation.
2. Identify technical issues.
3. Implement one improvement at a time.
4. Validate each change.
5. Commit each logical change separately.
6. Merge stable improvements into the main branch.

## test from codex
## Workflow test completed from VS Code to GitHub to Colab.

## Changes Completed So Far

### BUG FIX

- `train.py` — restricted class discovery to real, non-hidden directories.
- `train.py` — scaled image pixels from `0–255` to `0–1` before normalization.
- `train.py` — disabled augmentation for the internal validation dataset.
- `train.py` — passed raw logits to `CrossEntropyLoss` in training and validation.
- `train.py` — used `model.eval()` and `torch.no_grad()` during validation.
- `train.py` — stopped pooling the held-out `validation/` folder into model development data.
- `evaluate_model.py` — added clear errors for a missing validation directory, missing required class folders, or class folders without `.png` images.

### MODEL IMPROVEMENT

- `train.py` — saved the weights from the epoch with the best validation accuracy.
- `train.py` — added configured weight decay to the SGD optimizer.

### PIPELINE ALIGNMENT

- `evaluate_model.py` — replaced the legacy `SmallCNN` with the current ResNet18 model.
- `evaluate_model.py` — loaded `checkpoints/resnet18_final.pt` and its checkpoint dictionary.
- `evaluate_model.py` — used checkpoint class order, image size, and normalization statistics.
- `evaluate_model.py` — aligned image loading with training by using PIL and RGB input.
- `evaluate_model.py` — built the held-out validation path from `config.DATA_ROOT`.

### REPRODUCIBILITY

- `config.py` — established it as the central configuration source for the ResNet18 pipeline.
- `train.py` — removed the undocumented `lr * 100` multiplier.
- `train.py` — added NumPy and PyTorch seed `42`.
- `train.py` — saved class names, image size, and normalization metadata with the checkpoint.

### EXPERIMENT TRACKING

- `train.py` — appended completed-run configuration and results to `runs/training_history.csv`.
- `evaluate_model.py` — appended evaluation summaries to `runs/evaluation_history.csv`.

### FEATURE ADDITION

- `evaluate_model.py` — added a console confusion matrix in checkpoint class order.

### DOCUMENTATION

- `train.py` — corrected comments to describe the frozen pretrained backbone and trainable classification head.

## Planned Changes

1. [Status: DONE] [Priority: Low] [Category: DOCUMENTATION] train.py — top file docstring
- Correct the description: `build_model()` freezes the ResNet18 backbone with `param.requires_grad = False` and trains only the final `fc` layer.
- Reason: the current claim that the model performs full fine-tuning is inaccurate.

2. [Status: DONE] [Priority: High] [Category: BUG FIX] train.py — ToolDataset.__getitem__(); train_v2.py — normalization override
- Scale image pixels from `0–255` to `0–1` before normalization and keep training and validation preprocessing consistent.
- Reason: `train_v2.py` replaces the ImageNet mean/std with values from `stats.json`, which `ToolDataset.__getitem__()` uses at runtime, but those statistics assume images are already scaled to `0–1`.

3. [Status: PLANNED] [Priority: Medium] [Category: MODEL IMPROVEMENT] train.py / train_v2.py — normalization strategy investigation
- After fixing the normalization scale bug, compare ImageNet mean/std with the dataset-specific values from `stats.json` as a controlled experiment.
- Reason: determine whether ImageNet normalization better matches the pretrained ResNet18 backbone or dataset-specific normalization provides a measurable benefit for the surgical-tool dataset.

4. [Status: IN PROGRESS — training and evaluation histories implemented; run-linked artifacts pending] [Priority: Medium] [Category: EXPERIMENT TRACKING] train.py / train_v2.py / runs/ — experiment tracking
- Keep `runs/training_history.csv` as the run summary; add a unique `run_id`, per-run epoch-metrics CSV, and uniquely named checkpoint linked to that run.
- Save the best validation model and record its `run_id`, best epoch, best validation accuracy, and checkpoint path in `training_history.csv`.
- Reason: preserve lightweight experiment tracking while preventing checkpoint overwrites and retaining epoch-level results.

5. [Status: DONE] [Priority: High] [Category: BUG FIX] train.py / evaluate_model.py — dataset/class loading logic
- Discover only valid class directories and ignore hidden or system files such as `.DS_Store`.
- Reason: ensure the dataset consistently contains only the expected surgical-tool classes and prevent non-class files from entering the class mapping.

6. [Status: DONE] [Priority: High] [Category: REPRODUCIBILITY] config.py / train.py / train_v2.py — learning-rate handling
- Explicitly configure and use the classifier-head learning rate instead of applying the undocumented `lr * 100` multiplier.
- Ensure the configured and effective optimizer learning rates are clear and consistently recorded.
- Reason: the current pipeline exposes `0.0003`, `0.001`, and an actual optimizer learning rate of `0.1`, making runs difficult to interpret and reproduce.

7. ⚠️ [Status: DONE] [Priority: High] [Category: BUG FIX] train.py — training and validation loss calculation
- Change `criterion(probs, lbls)` to `criterion(logits, lbls)` in both training and validation; use softmax separately only when probabilities are needed.
- Reason: `nn.CrossEntropyLoss()` expects raw logits, so passing post-softmax probabilities distorts the training loss, gradients, and validation loss.

8. ⚠️ [Status: DONE] [Priority: High] [Category: MODEL IMPROVEMENT] train.py / train_v2.py — best-model selection
- Save or update the checkpoint whenever validation accuracy improves so its weights correspond to `best_val_accuracy` and `best_epoch`.
- Reason: the current code records the best validation result only for reporting, while `resnet18_final.pt` contains the final epoch's weights and may represent a worse model.

9. [Status: PLANNED] [Priority: Low] [Category: REPRODUCIBILITY] train_v2.py — hyperparameter configuration
- Review the hard-coded `batch_size=8`, `lr=1e-3`, and `epochs=10`, which currently override values from `config.py`.
- Decide whether these overrides are intentional or whether training hyperparameters should come from a single configuration source.
- Reason: avoid conflicting configuration values and make training runs easier to understand and reproduce.

10. [Status: PLANNED] [Priority: High] [Category: PIPELINE ALIGNMENT] predict.py — align image preprocessing with training
- Update inference preprocessing to match the current ResNet18 training pipeline: RGB input, `224 × 224` resizing, correct `0–1` scaling, and the same mean/std normalization used during training.
- Do not copy training-only augmentation such as random flip or color jitter into prediction.
- Reason: `predict.py` currently uses grayscale `128 × 128` images with different scaling and no normalization, so inference inputs do not match the distribution used to train the ResNet18 model.

11. [Status: IN PROGRESS — evaluation complete; prediction pending] [Priority: High] [Category: PIPELINE ALIGNMENT] predict.py / evaluate_model.py — create current ResNet18 inference/evaluation pipeline

- Replace the legacy SmallCNN/model_best.pt path with the current ResNet18 architecture and checkpoint.
- Align evaluation preprocessing with training: RGB input, 224 × 224 resizing, correct 0–1 scaling, and the same normalization.
- Preserve the required prediction CLI and CSV interface.

- Reason: predict.py and evaluate_model.py still use legacy model/preprocessing paths that do not match the current ResNet18 training pipeline.

12. [Status: PLANNED] [Priority: High] [Category: BUG FIX] predict.py — image-loading error handling
- Remove the silent fallback that replaces failed images with `torch.zeros(3, 128, 128)` and report the failed filename and error.
- Decide whether failed images should be skipped or explicitly marked as failed in the output instead of generating a normal prediction from an artificial black image.
- Reason: hidden image-processing failures create normal-looking but unreliable predictions and mask data-quality problems.

13. [Status: DONE] [Priority: High] [Category: BUG FIX] train.py / train_v2.py / evaluate_model.py — prevent train/evaluation data leakage
- Stop pooling the predefined `train/` and `validation/` folders before creating a new random split, and preserve a clean separation between training and validation data.
- Use only the original `train/` folder for model development and hold out a small stratified portion for training-time validation and model selection.
- Initially consider a 5% holdout; increase it to 10–20% if 5% produces too few validation examples per class.
- Keep the original `validation/` folder completely untouched during training and use it only for final evaluation.
- Document that the original `validation/` folder serves as the held-out evaluation set because the project has no separate test set.
- Reason: predefined validation images can currently be used during training and evaluated again, making the reported evaluation accuracy non-independent.

14. [Status: DONE] [Priority: Medium] [Category: PIPELINE ALIGNMENT] evaluate_model.py — use shared dataset configuration- Replace the hardcoded repository-local validation path with a path built from config.DATA_ROOT.
- Avoid maintaining separate dataset paths in training and evaluation.
- Reason: the current path does not exist locally and is not portable to Colab.

15. [Status: PLANNED] [Priority: Medium] [Category: PIPELINE ALIGNMENT] run_all.sh — align end-to-end pipeline- Run one connected ResNet18 workflow: training → evaluation → submission/preflight check.
- Remove the disconnected EfficientNet experiment and unrelated config-printing step.
- Reason: the current script trains EfficientNet, evaluates legacy SmallCNN, and does not run submission checking.

16. [Status: PLANNED] [Priority: Medium] [Category: MODEL IMPROVEMENT] train.py — class-weighted loss (planned)
- Add class-weighted `CrossEntropyLoss` to address class imbalance, especially the underrepresented scissor class.
- Reason: reduce bias toward larger classes and improve learning for the underrepresented scissor class.
