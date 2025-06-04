import torch

BATCH_SIZE = 16
LR = 1e-4
LAMBDA = 1e-2
EPOCHS = 20
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False
SAVE_MODEL = True