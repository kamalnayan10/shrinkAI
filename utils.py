import torch
import os
from torchvision.utils import save_image
import torch.nn.functional as F

def save_input_output_images(input_tensor, output_tensor, epoch, save_dir="comparison_outputs"):

    os.makedirs(save_dir, exist_ok=True)

    input_img = input_tensor[0].detach().cpu()
    output_img = output_tensor[0].detach().cpu()

    # denormalise
    input_img = (input_img + 1) / 2
    output_img = (output_img + 1) / 2

    output_img = torch.clamp(output_img, 0, 1)

    save_image(input_img, os.path.join(save_dir, f"input_epoch_{epoch}.png"))
    save_image(output_img, os.path.join(save_dir, f"output_epoch_{epoch}.png"))

def pad_to_multiple(x, multiple=64):
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h)), (h, w)

def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print("Saved Model and Optimizer state dicts successfully!")

def load_checkpoint(model, optimizer, checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("✅ Loaded Model and Optimizer state dicts successfully!")