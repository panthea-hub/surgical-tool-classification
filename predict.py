"""Inference CLI - runs the current best checkpoint over a directory of
images and writes out a predictions CSV.

Usage:
  python predict.py --data-dir <DIR> --out <CSV>

<DIR> should look like the data/cholec-tinytools/train (or validation)
folder: one subfolder per class, containing .png frames. A flat folder of
images (no subfolders) also works.
"""
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train import build_model

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "resnet18_final.pt")


class InferenceDataset(Dataset):
    def __init__(self, data_dir, image_size, normalization_mean, normalization_std):
        self.image_size = image_size
        self.normalization_mean = torch.tensor(normalization_mean).view(3, 1, 1)
        self.normalization_std = torch.tensor(normalization_std).view(3, 1, 1)
        subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
        self.paths = []
        if subdirs:
            for d in subdirs:
                self.paths.extend(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))
        else:
            self.paths.extend(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
            tensor = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
            tensor = (tensor - self.normalization_mean) / self.normalization_std
        except Exception:
            # corrupt or unreadable frame - fall back to a blank image so
            # the batch shapes stay consistent and inference doesn't stop
            tensor = torch.zeros(3, self.image_size, self.image_size)
        return tensor, os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    normalization_mean = checkpoint["normalization_mean"]
    normalization_std = checkpoint["normalization_std"]
    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ds = InferenceDataset(
        args.data_dir, image_size, normalization_mean, normalization_std
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    rows = []
    with torch.no_grad():
        for imgs, names in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            for name, p in zip(names, preds):
                rows.append((name, class_names[p]))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_class"])
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
