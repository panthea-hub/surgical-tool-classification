"""Central configuration for the main ResNet18 training pipeline."""

import os

LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = 15
NUM_CLASSES = 4
IMAGE_SIZE = 224
WEIGHT_DECAY = 1e-5
DATA_ROOT = os.environ.get("DATA_ROOT", "data/cholec-tinytools")
