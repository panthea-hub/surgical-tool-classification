"""Main training entry point - ResNet-18 tool classifier.

The model uses a pretrained frozen backbone and trains a new classification
head; see the model setup below.

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
    "run_id",
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
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        if self.augment:
            tensor = AUGMENT(tensor)
        # Normalize with standard ImageNet stats so the pretrained backbone
        # sees inputs on the distribution it expects.
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, self.labels[idx]


def load_all_paths_and_labels():
    """Load images from the training folder for the internal split below."""
    train_root = os.path.join(config.DATA_ROOT, "train")
    classes = sorted(
        entry.name
        for entry in os.scandir(train_root)
        if entry.is_dir() and not entry.name.startswith(".")
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for cls in classes:
        for p in glob.glob(f"{train_root}/{cls}/*.png"):
            paths.append(p)
            labels.append(class_to_idx[cls])
    return paths, labels, classes


def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    # Keep the pretrained backbone frozen and train a new classification head.
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def append_training_history(row):
    os.makedirs(os.path.dirname(TRAINING_HISTORY_PATH), exist_ok=True)
    write_header = not os.path.exists(TRAINING_HISTORY_PATH)
    if not write_header:
        with open(TRAINING_HISTORY_PATH, newline="") as history_file:
            reader = csv.DictReader(history_file)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames
        if existing_fields != TRAINING_HISTORY_FIELDS:
            with open(TRAINING_HISTORY_PATH, "w", newline="") as history_file:
                writer = csv.DictWriter(history_file, fieldnames=TRAINING_HISTORY_FIELDS)
                writer.writeheader()
                writer.writerows(existing_rows)
    with open(TRAINING_HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(batch_size=16, lr=config.LEARNING_RATE, epochs=config.NUM_EPOCHS):
    np.random.seed(42)
    torch.manual_seed(42)
    start_time = datetime.now().astimezone()
    run_id = start_time.strftime("%Y%m%dT%H%M%S%f%z")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as metrics_file:
        csv.DictWriter(
            metrics_file,
            fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"],
        ).writeheader()
    start_counter = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels, classes = load_all_paths_and_labels()
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.15, shuffle=True, random_state=42, stratify=labels,
    )

    train_ds = ToolDataset(train_paths, train_labels, augment=True)
    val_ds = ToolDataset(val_paths, val_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(len(classes)).to(device)
    # Only the head has requires_grad=True at this point, so this optimizes
    # the classifier on top of frozen pretrained features - much faster to
    # converge than updating the whole backbone.
    optimizer = torch.optim.SGD(
        model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = float("-inf")
    best_epoch = None
    best_model_state = None
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
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(dim=1) == lbls).sum().item()

        final_train_loss = total_loss / len(train_ds)
        final_val_loss = val_loss / len(val_ds)
        final_val_accuracy = val_correct / len(val_ds)
        if final_val_accuracy > best_val_accuracy:
            best_val_accuracy = final_val_accuracy
            best_epoch = epoch
            best_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        print(
            f"epoch {epoch:02d}  train_loss={final_train_loss:.4f}  "
            f"val_loss={final_val_loss:.4f}  val_acc={final_val_accuracy:.4f}"
        )
        with open(metrics_path, "a", newline="") as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"],
            )
            writer.writerow({
                "epoch": epoch,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "val_accuracy": final_val_accuracy,
            })

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/resnet18_{run_id}.pt"
    torch.save(
        {
            "model_state_dict": best_model_state,
            "class_names": classes,
            "image_size": config.IMAGE_SIZE,
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        checkpoint_path,
    )
    print(f"saved {checkpoint_path}")

    end_time = datetime.now().astimezone()
    duration_seconds = time.perf_counter() - start_counter
    optimizer_group = optimizer.param_groups[0]
    append_training_history({
        "run_id": run_id,
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
