"""Quick accuracy check for the current best checkpoint against the
validation set (we don't have a held-out test set - validation is what we
have, so that's what this reports against).

Usage: python evaluate_model.py
"""
import argparse
import csv
import json
import os
import shutil
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from train import build_model

DATA_ROOT = os.path.join(config.DATA_ROOT, "validation")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "resnet18_final.pt")
EVALUATION_HISTORY_PATH = "runs/evaluation_history.csv"


class EvalDataset(Dataset):
    def __init__(self, root, class_names, image_size, normalization_mean, normalization_std):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Validation directory does not exist: {root}")

        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.image_size = image_size
        self.normalization_mean = np.array(normalization_mean, dtype=np.float32)
        self.normalization_std = np.array(normalization_std, dtype=np.float32)
        self.samples = []
        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Required class directory does not exist: {cls_dir}")
            class_samples = []
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                if not fname.startswith(".") and fname.lower().endswith(".png") and os.path.isfile(path):
                    class_samples.append((path, self.class_to_idx[cls]))
            if not class_samples:
                raise ValueError(f"Required class directory contains no .png images: {cls_dir}")
            self.samples.extend(class_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.array(Image.open(path).convert("RGB"))
        arr = np.array(Image.fromarray(arr).resize((self.image_size, self.image_size)))
        arr = arr.astype(np.float32) / 255.0
        arr = (arr - self.normalization_mean) / self.normalization_std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), label, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    normalization_mean = checkpoint["normalization_mean"]
    normalization_std = checkpoint["normalization_std"]
    dataset = EvalDataset(
        DATA_ROOT, class_names, image_size, normalization_mean, normalization_std
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    per_class_correct = {c: 0 for c in class_names}
    per_class_total = {c: 0 for c in class_names}
    total_correct = 0
    total_samples = 0
    true_labels = []
    predicted_labels = []
    misclassified = []

    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(preds.tolist())
            for p, t, path in zip(preds.tolist(), labels.tolist(), paths):
                cls_name = class_names[t]
                per_class_total[cls_name] += 1
                if p == t:
                    per_class_correct[cls_name] += 1
                else:
                    misclassified.append((path, cls_name, class_names[p]))

    overall_accuracy = total_correct / total_samples
    print(f"overall accuracy: {overall_accuracy:.4f}")
    print("per-class breakdown:")
    per_class_accuracy = {}
    for cls in class_names:
        total = per_class_total[cls]
        acc = per_class_correct[cls] / total if total else float("nan")
        per_class_accuracy[cls] = acc
        print(f"  {cls:10s}  n={total:4d}  acc={acc:.4f}")

    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1
    print("confusion matrix:")
    print(confusion_matrix)

    os.makedirs(os.path.dirname(EVALUATION_HISTORY_PATH), exist_ok=True)
    write_header = not os.path.exists(EVALUATION_HISTORY_PATH)
    with open(EVALUATION_HISTORY_PATH, "a", newline="") as history_file:
        fieldnames = [
            "timestamp",
            "checkpoint_path",
            "overall_accuracy",
            "per_class_accuracy",
            "samples_per_class",
        ]
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().astimezone().isoformat(),
            "checkpoint_path": args.checkpoint,
            "overall_accuracy": overall_accuracy,
            "per_class_accuracy": json.dumps(per_class_accuracy),
            "samples_per_class": json.dumps(per_class_total),
        })

    error_dir = os.path.join("runs", "error_analysis")
    # Remove only this generated directory; never follow a directory symlink.
    if os.path.islink(error_dir):
        os.unlink(error_dir)
    elif os.path.exists(error_dir):
        shutil.rmtree(error_dir)
    os.makedirs(error_dir)

    for path, actual, predicted in misclassified:
        destinations = [
            os.path.join(error_dir, f"missed_{actual}", f"predicted_{predicted}"),
            os.path.join(error_dir, f"predicted_as_{predicted}", f"actual_{actual}"),
        ]
        for destination in destinations:
            os.makedirs(destination, exist_ok=True)
            shutil.copy2(path, os.path.join(destination, os.path.basename(path)))

    print(f"error analysis: {len(misclassified)} misclassified images saved to {error_dir}/")


if __name__ == "__main__":
    main()
