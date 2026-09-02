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

1. [Priority: Low] train.py — top file docstring
- Correct the description: `build_model()` freezes the ResNet18 backbone with `param.requires_grad = False` and trains only the final `fc` layer.
- Reason: the current claim that the model performs full fine-tuning is inaccurate.

2. [Priority: High] train.py — ToolDataset.__getitem__(); train_v2.py — normalization override
- Scale image pixels from `0–255` to `0–1` before normalization and keep training and validation preprocessing consistent.
- Reason: `train_v2.py` replaces the ImageNet mean/std with values from `stats.json`, which `ToolDataset.__getitem__()` uses at runtime, but those statistics assume images are already scaled to `0–1`.

3. [Priority: Medium] train.py / train_v2.py — normalization strategy investigation
- After fixing the normalization scale bug, compare ImageNet mean/std with the dataset-specific values from `stats.json` as a controlled experiment.
- Reason: determine whether ImageNet normalization better matches the pretrained ResNet18 backbone or dataset-specific normalization provides a measurable benefit for the surgical-tool dataset.

4. [Priority: Medium] train.py / train_v2.py / runs/ — experiment tracking
- Keep `runs/training_history.csv` as the run summary; add a unique `run_id`, per-run epoch-metrics CSV, and uniquely named checkpoint linked to that run.
- Save the best validation model and record its `run_id`, best epoch, best validation accuracy, and checkpoint path in `training_history.csv`.
- Reason: preserve lightweight experiment tracking while preventing checkpoint overwrites and retaining epoch-level results.

5. [Priority: High] train.py — dataset/class loading logic
- Discover only valid class directories and ignore hidden or system files such as `.DS_Store`.
- Reason: ensure the dataset consistently contains only the expected surgical-tool classes and prevent non-class files from entering the class mapping.

6. [Priority: High] config.py / train.py / train_v2.py — learning-rate handling
- Explicitly configure and use the classifier-head learning rate instead of applying the undocumented `lr * 100` multiplier.
- Ensure the configured and effective optimizer learning rates are clear and consistently recorded.
- Reason: the current pipeline exposes `0.0003`, `0.001`, and an actual optimizer learning rate of `0.1`, making runs difficult to interpret and reproduce.

7. ⚠️ [Priority: High] train.py — training and validation loss calculation
- Change `criterion(probs, lbls)` to `criterion(logits, lbls)` in both training and validation; use softmax separately only when probabilities are needed.
- Reason: `nn.CrossEntropyLoss()` expects raw logits, so passing post-softmax probabilities distorts the training loss, gradients, and validation loss.

8. ⚠️ [Priority: High] train.py / train_v2.py — best-model selection
- Save or update the checkpoint whenever validation accuracy improves so its weights correspond to `best_val_accuracy` and `best_epoch`.
- Reason: the current code records the best validation result only for reporting, while `resnet18_final.pt` contains the final epoch's weights and may represent a worse model.
