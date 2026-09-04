#!/usr/bin/env bash
# Runs the connected ResNet18 training, evaluation, prediction, and
# submission-validation pipeline.
set -e

DATA_ROOT="$(python -c 'import config; print(config.DATA_ROOT)')"

echo "=== training ResNet18 ==="
python train_v2.py

echo "=== evaluating production checkpoint ==="
python evaluate_model.py

echo "=== generating predictions ==="
python predict.py --data-dir "$DATA_ROOT/validation" --out predictions.csv

echo "=== checking submission ==="
python check_submission.py
