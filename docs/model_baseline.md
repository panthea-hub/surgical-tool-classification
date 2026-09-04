# Model Performance Scorecard

## Original

Held-out accuracy: Not reliably recorded

## Stabilized Baseline

- Held-out accuracy: **83.0%**
- Scissor accuracy: **37.0%**

## Experiment 1 — Class-Weighted Loss

- Overall accuracy: **79.4%** (baseline **83.0%**)
- Scissor accuracy: **48.2%** (baseline **37.0%**)
- Result: scissor improved **+11.1 points**, but overall accuracy dropped **3.6 points**.
- Decision: **Do not adopt**; investigate a less aggressive imbalance strategy next.
- Checkpoint: `resnet18_20260903T210727047388-0700.pt`

## Experiment 2 — Square-Root Class Weighting

- Overall accuracy: **81.6%** (baseline **83.0%**)
- Scissor accuracy: **40.7%** (baseline **37.0%**)
- Decision: **Do not adopt**

## Experiment 3 — Partial ResNet18 Fine-Tuning (`layer4` + `fc`)

- Overall accuracy: **88.5%** (baseline **83.0%**)
- Clipper accuracy: **89.8%**
- Grasper accuracy: **93.1%**
- Hook accuracy: **96.0%**
- Scissor accuracy: **40.7%** (baseline **37.0%**)
- Decision: **ADOPT**
- Checkpoint: `resnet18_20260903T220901549787-0700.pt`

## Progression

**Original** → **Stabilized baseline: 83.0%** → **Partial fine-tuning: 88.5% (adopted)**

**Next goal:** Improve minority-class performance without sacrificing overall accuracy.
