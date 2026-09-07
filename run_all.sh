#!/usr/bin/env bash
# Runs the connected ResNet18 training, evaluation, prediction, and
# submission-validation pipeline.
set -e

DATA_ROOT="$(python -c 'import config; print(config.DATA_ROOT)')"

echo "=== training ResNet18 ==="
python train_v2.py
CHECKPOINT_PATH="$(<runs/latest_checkpoint.txt)"

echo "=== evaluating production checkpoint ==="
python evaluate_model.py --checkpoint "$CHECKPOINT_PATH"

echo "=== generating predictions ==="
python predict.py --checkpoint "$CHECKPOINT_PATH" --data-dir "$DATA_ROOT/validation" --out predictions.csv

echo "=== checking submission ==="
python check_submission.py

echo "=== generating embedding analysis ==="
python analyze_embeddings.py --checkpoint "$CHECKPOINT_PATH"

echo "=== pipeline complete ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Error analysis: runs/error_analysis/"
echo "Embedding plot: runs/embedding_analysis/embedding_umap.png"
echo "Embedding CSV: runs/embedding_analysis/embedding_umap.csv"
echo "Interactive embedding plot: runs/embedding_analysis/embedding_umap.html"