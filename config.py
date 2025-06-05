import torch

BATCH_SIZE = 16
LR = 1e-3
LAMBDA = 1e-4
EPOCHS = 100
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False
SAVE_MODEL = True