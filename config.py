"""Central configuration for the main ResNet18 training pipeline."""

import os

LEARNING_RATE = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_CLASSES = 4
IMAGE_SIZE = 224
WEIGHT_DECAY = 1e-5

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    "/Users/panthea/Documents/Datasets/cholec-tinytools/data/cholec-tinytools"
)
