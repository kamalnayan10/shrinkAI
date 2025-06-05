import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import pickle
import math

from compression_model import Compression
from utils import load_checkpoint, pad_to_multiple
from config import *


def preprocess_and_pad(image_path: str, device: torch.device):

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = to_tensor(img).unsqueeze(0).to(device)
    x_padded, orig_shape = pad_to_multiple(x, multiple=64)
    return x_padded, orig_shape


def generate_scale_table(min_scale=0.11, max_scale=256, levels=64):
    """Generate a logarithmically spaced scale table (like CompressAI)"""
    return torch.exp(torch.linspace(math.log(min_scale), math.log(max_scale), levels))

def compress_image(model, x: torch.Tensor, orig_shape: tuple, output_path: str):

    model.entropy_bottleneck.update()

    with torch.no_grad():
        y = model.encoder(x)
        z = model.h_enc(y)

        z_strings = model.entropy_bottleneck.compress(z)

        mu, sigma = model.h_dec(z)


        sigma_min = sigma.min().item()
        sigma_max = sigma.max().item()
        scale_table = generate_scale_table(min_scale=max(0.01, sigma_min * 0.8),
                                        max_scale=sigma_max * 1.2)
        model.gaussian_cond.scale_table = scale_table
        model.gaussian_cond.update()

        y_strings = model.gaussian_cond.compress(y, sigma)

    data = {
        "z_strings":  z_strings,
        "y_strings":  y_strings,
        "z_shape":    z.shape,
        "orig_shape": orig_shape
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="ShrinkAI: compress a single image into a .pth bitstream"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input image (e.g. input.jpg)"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Path to save compressed file (e.g. compressed.pth)"
    )

    args = parser.parse_args()

    device = DEVICE

    model = Compression(3,32,192,3, device = device).to(device)
    load_checkpoint(model, None, "good_models/full_ffct_32in.pth", device)

    x_padded, orig_shape = preprocess_and_pad(args.input, device)

    compress_image(model, x_padded, orig_shape, args.output)

    print(f"\n✅ Compressed file saved at: {args.output}")
    print(f"   • Original image shape: {orig_shape[0]}×{orig_shape[1]}")
    print(f"   • Padded image shape:   {x_padded.shape[-2]}×{x_padded.shape[-1]}\n")


if __name__ == "__main__":
    main()
