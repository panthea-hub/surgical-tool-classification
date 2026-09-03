"""Quick accuracy check for the current best checkpoint against the
validation set (we don't have a held-out test set - validation is what we
have, so that's what this reports against).

Usage: python evaluate_model.py
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from helpers.metrics import batch_accuracy, epoch_accuracy
from train import build_model

DATA_ROOT = os.path.join(config.DATA_ROOT, "validation")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "resnet18_final.pt")


class EvalDataset(Dataset):
    def __init__(self, root, class_names, image_size, normalization_mean, normalization_std):
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.image_size = image_size
        self.normalization_mean = np.array(normalization_mean, dtype=np.float32)
        self.normalization_std = np.array(normalization_std, dtype=np.float32)
        self.samples = []
        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.array(Image.open(path).convert("RGB"))
        arr = np.array(Image.fromarray(arr).resize((self.image_size, self.image_size)))
        arr = arr.astype(np.float32) / 255.0
        arr = (arr - self.normalization_mean) / self.normalization_std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    normalization_mean = checkpoint["normalization_mean"]
    normalization_std = checkpoint["normalization_std"]
    dataset = EvalDataset(
        DATA_ROOT, class_names, image_size, normalization_mean, normalization_std
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    per_class_correct = {c: 0 for c in class_names}
    per_class_total = {c: 0 for c in class_names}
    batch_accs = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            batch_accs.append(batch_accuracy(logits, labels))
            preds = logits.argmax(dim=1)
            for p, t in zip(preds.tolist(), labels.tolist()):
                cls_name = class_names[t]
                per_class_total[cls_name] += 1
                if p == t:
                    per_class_correct[cls_name] += 1

    print(f"overall accuracy: {epoch_accuracy(batch_accs):.4f}")
    print("per-class breakdown:")
    for cls in class_names:
        total = per_class_total[cls]
        acc = per_class_correct[cls] / total if total else float("nan")
        print(f"  {cls:10s}  n={total:4d}  acc={acc:.4f}")


if __name__ == "__main__":
    main()
