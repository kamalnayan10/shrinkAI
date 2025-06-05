import argparse
import torch
from torchvision.utils import save_image
import pickle

from compression_model import Compression
from utils import load_checkpoint
import math
from config import *

def generate_scale_table(min_scale=0.11, max_scale=256, levels=64):
    return torch.exp(torch.linspace(math.log(min_scale), math.log(max_scale), levels))


def decompress_image(model, compressed, device):
    z_strings = compressed["z_strings"]
    y_strings = compressed["y_strings"]
    z_shape = compressed["z_shape"]
    orig_shape = compressed["orig_shape"]


    model.entropy_bottleneck.update()
    with torch.no_grad():
        z_hat = model.entropy_bottleneck.decompress(
            z_strings, size = z_shape
        ).squeeze(0).to(device)
        z_hat = z_hat[0, 0, :, :, :].unsqueeze(0)

        mu, sigma = model.h_dec(z_hat)

        sigma_min = sigma.min().item()
        sigma_max = sigma.max().item()
        scale_table = generate_scale_table(min_scale=max(0.01, sigma_min * 0.8),
                                        max_scale=sigma_max * 1.2)
        model.gaussian_cond.scale_table = scale_table
        model.gaussian_cond.update()
        y_hat = model.gaussian_cond.decompress(y_strings, sigma).to(device)

        x_hat = model.decoder(y_hat)
        x_hat = x_hat.clamp(-1, 1)

        # denormalize image
        x_hat = (x_hat + 1) / 2

        # crop to original size
        h_orig, w_orig = orig_shape
        x_hat = x_hat[:, :, :h_orig, :w_orig]

    return x_hat



def main():
    parser = argparse.ArgumentParser(description="Decompress an image from .bin file")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to .bin file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output PNG")
    args = parser.parse_args()

    device = DEVICE 

    model = Compression(3,32,192,3,device = device).to(device)
    load_checkpoint(model, None, "good_models/full_ffct_32in.pth", device)

    with open(args.input, "rb") as f:
        compressed = pickle.load(f)

    x_hat = decompress_image(model, compressed, device)

    save_image(x_hat, args.output)
    print(f"✅ Image decompressed and saved to: {args.output}")


if __name__ == "__main__":
    main()
