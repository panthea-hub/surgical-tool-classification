"""Main training entry point - ResNet-18 finetune on the tool classifier.

This is the current best model. We do a full finetune of the backbone with
a fresh classification head on top; see the model setup below.

Usage: python train.py
"""
import csv
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])

TRAINING_HISTORY_PATH = "runs/training_history.csv"
TRAINING_HISTORY_FIELDS = [
    "run_timestamp",
    "script_name",
    "dataset_path",
    "model_name",
    "class_names",
    "image_size",
    "configured_batch_size",
    "effective_batch_size",
    "configured_epochs",
    "effective_epochs",
    "configured_learning_rate",
    "effective_learning_rate",
    "effective_optimizer_learning_rate",
    "configured_weight_decay",
    "effective_weight_decay",
    "optimizer",
    "momentum",
    "validation_split",
    "random_seed",
    "normalization_mean",
    "normalization_std",
    "augmentation_settings",
    "device",
    "start_time",
    "end_time",
    "duration_seconds",
    "final_train_loss",
    "final_validation_loss",
    "final_validation_accuracy",
    "best_validation_accuracy",
    "best_epoch",
    "checkpoint_path",
]


class ToolDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        arr = np.array(img)  # HWC, uint8, 0-255
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        if self.augment:
            tensor = AUGMENT(tensor)
        # Normalize with standard ImageNet stats so the pretrained backbone
        # sees inputs on the distribution it expects.
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, self.labels[idx]


def load_all_paths_and_labels():
    """Pull together every image we have - train and validation folders both
    get pooled here and we carve out our own split below, since the
    validation folder alone is too small to get a stable estimate from."""
    classes = sorted(os.listdir(config.DATA_ROOT + "/train"))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for split in ("train", "validation"):
        for cls in classes:
            for p in glob.glob(f"{config.DATA_ROOT}/{split}/{cls}/*.png"):
                paths.append(p)
                labels.append(class_to_idx[cls])
    return paths, labels, classes


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Full finetune: unfreeze everything and let the whole network adapt to
    # the surgical domain, then swap in our classification head.
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def append_training_history(row):
    os.makedirs(os.path.dirname(TRAINING_HISTORY_PATH), exist_ok=True)
    write_header = not os.path.exists(TRAINING_HISTORY_PATH)
    with open(TRAINING_HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(batch_size=16, lr=config.LEARNING_RATE, epochs=config.NUM_EPOCHS):
    start_time = datetime.now().astimezone()
    start_counter = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels, classes = load_all_paths_and_labels()
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.15, shuffle=True, random_state=42, stratify=labels,
    )

    train_ds = ToolDataset(train_paths, train_labels, augment=True)
    val_ds = ToolDataset(val_paths, val_labels, augment=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(len(classes)).to(device)
    # Only the head has requires_grad=True at this point, so this optimizes
    # the classifier on top of frozen pretrained features - much faster to
    # converge than updating the whole backbone.
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr * 100, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = float("-inf")
    best_epoch = None
    final_train_loss = None
    final_val_loss = None
    final_val_accuracy = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            loss = criterion(probs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        val_loss = 0.0
        val_correct = 0
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            loss = criterion(probs, lbls)
            val_loss += loss.item() * imgs.size(0)
            val_correct += (probs.argmax(dim=1) == lbls).sum().item()

        final_train_loss = total_loss / len(train_ds)
        final_val_loss = val_loss / len(val_ds)
        final_val_accuracy = val_correct / len(val_ds)
        if final_val_accuracy > best_val_accuracy:
            best_val_accuracy = final_val_accuracy
            best_epoch = epoch

        print(
            f"epoch {epoch:02d}  train_loss={final_train_loss:.4f}  "
            f"val_loss={final_val_loss:.4f}  val_acc={final_val_accuracy:.4f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/resnet18_final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print("saved checkpoints/resnet18_final.pt")

    end_time = datetime.now().astimezone()
    duration_seconds = time.perf_counter() - start_counter
    optimizer_group = optimizer.param_groups[0]
    append_training_history({
        "run_timestamp": start_time.isoformat(),
        "script_name": os.path.basename(sys.argv[0]),
        "dataset_path": config.DATA_ROOT,
        "model_name": "resnet18",
        "class_names": "|".join(classes),
        "image_size": config.IMAGE_SIZE,
        "configured_batch_size": config.BATCH_SIZE,
        "effective_batch_size": batch_size,
        "configured_epochs": config.NUM_EPOCHS,
        "effective_epochs": epochs,
        "configured_learning_rate": config.LEARNING_RATE,
        "effective_learning_rate": lr,
        "effective_optimizer_learning_rate": optimizer_group["lr"],
        "configured_weight_decay": config.WEIGHT_DECAY,
        "effective_weight_decay": optimizer_group["weight_decay"],
        "optimizer": optimizer.__class__.__name__,
        "momentum": optimizer_group["momentum"],
        "validation_split": 0.15,
        "random_seed": 42,
        "normalization_mean": "|".join(map(str, IMAGENET_MEAN)),
        "normalization_std": "|".join(map(str, IMAGENET_STD)),
        "augmentation_settings": " ".join(str(AUGMENT).split()),
        "device": str(device),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration_seconds,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_val_loss,
        "final_validation_accuracy": final_val_accuracy,
        "best_validation_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
    })


if __name__ == "__main__":
    main()
