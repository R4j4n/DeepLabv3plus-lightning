import torch

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "person-segmentation/train/image"
TRAIN_MASK_DIR = "person-segmentation/train/mask"
VAL_IMG_DIR = "person-segmentation/test/image"
VAL_MASK_DIR = "person-segmentation/test/mask"