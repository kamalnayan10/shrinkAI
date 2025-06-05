import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from compressai.losses.rate_distortion import RateDistortionLoss
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import StepLR

from dataset import DIV2KDataset
from compression_model import Compression

from utils import save_checkpoint, save_input_output_images, load_checkpoint

import torch
from compressai.losses.rate_distortion import RateDistortionLoss
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import os
import math

from config import *

def lambda_schedule(epoch, start=0.001, end=0.03, total_epochs=100):
    cos_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    return end + (start - end) * cos_decay

def train(model, train_loader, device, epochs=20, lr=1e-4, lambda_rd=0.01,
          checkpoint_dir="checkpoints", image_dir="comparison_outputs"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    model.to(device)
    criterion = RateDistortionLoss(lmbda=lambda_rd)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # halves every 10 epochs

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    if LOAD_MODEL:
        load_checkpoint(model, optimizer, "checkpoints_ffct/compression_epoch30.pth", DEVICE)

    for epoch in range(epochs):
        model.train()
        total_loss, total_psnr, total_ssim, total_bpp = 0, 0, 0, 0

        loop = tqdm(train_loader, leave=False)
        criterion.lmbda = lambda_schedule(epoch, lambda_rd, 3e-2, epochs)
        for i, batch in enumerate(loop):
            batch = batch.to(device)

            output = model(batch)
            loss_dict = criterion(output, batch)

            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            

            total_loss += loss_dict["loss"].item()

            y_likelihoods = output["likelihoods"]["y"]
            z_likelihoods = output["likelihoods"]["z"]

            N, _, H, W = batch.shape
            num_pixels = N * H * W

            y_bits = torch.sum(-torch.log2(y_likelihoods + 1e-9))
            z_bits = torch.sum(-torch.log2(z_likelihoods + 1e-9))
            bpp = (y_bits + z_bits) / num_pixels

            total_bpp += bpp

            # Clamp and denormalize for metrics
            x_hat = output['x_hat'].clamp(-1, 1)
            target = batch.clamp(-1, 1)
            x_hat_denorm = (x_hat + 1) / 2
            target_denorm = (target + 1) / 2

            total_psnr += psnr_metric(x_hat_denorm, target_denorm).item()
            total_ssim += ssim_metric(x_hat_denorm, target_denorm).item()

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=loss_dict["loss"].item())

            if i == 0 and epoch % 5 == 0:
                save_input_output_images(target, x_hat, epoch, save_dir=image_dir)

        lr_scheduler.step()

        N = len(train_loader)
        avg_loss = total_loss / N
        avg_psnr = total_psnr / N
        avg_ssim = total_ssim / N
        avg_bpp = total_bpp / N

        current_lr = optimizer.param_groups[0]['lr']

        print(f"""Epoch [{epoch}/{epochs}] — Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | BPP: {avg_bpp:.4f}
              | cur_LR: {current_lr:.6f} | cur_lambda: {criterion.lmbda:.4f}""")

        if epoch % 5 == 0 and SAVE_MODEL:
            save_checkpoint(model, optimizer, f"{checkpoint_dir}/compression_epoch{epoch}.pth")



if __name__ == "__main__":

    train_dataset = DIV2KDataset("DIV2K_train_HR", crop_size=IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = Compression(in_channels=3, dim=32, latent_channels=192, z_channels=96, out_channels=3)

    os.makedirs("checkpoints", exist_ok=True)
    train(model, train_loader, DEVICE, epochs=EPOCHS, lr=LR, lambda_rd=LAMBDA, checkpoint_dir="checkpoints_ffct", image_dir="comparison_ffct_output")
