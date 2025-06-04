import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from compressai.losses.rate_distortion import RateDistortionLoss
from tqdm import tqdm
import os

from dataset import DIV2KDataset
from compression_model import Compression

from utils import save_checkpoint, save_input_output_images, load_checkpoint

import torch
from compressai.losses.rate_distortion import RateDistortionLoss
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import os

from config import *

def train(model, train_loader, device, epochs=20, lr=1e-4, lambda_rd=0.01, checkpoint_dir="checkpoints", image_dir="comparison_outputs"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    model.to(device)
    criterion = RateDistortionLoss(lmbda=lambda_rd)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    if LOAD_MODEL:
        model, optimizer = load_checkpoint(model, optimizer, "checkpoints/compression_epoch0.pth", DEVICE)

    for epoch in range(0, epochs):
        model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        loop = tqdm(train_loader, leave=False)
        for i, batch in enumerate(loop):
            batch = batch.to(device)

            output = model(batch)
            loss_dict = criterion(output, batch)

            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()

            total_loss += loss_dict["loss"].item()

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

        N = len(train_loader)
        avg_loss = total_loss / N
        avg_psnr = total_psnr / N
        avg_ssim = total_ssim / N

        print(f"Epoch [{epoch}/{epochs}] — Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

        if epoch % 5 == 0 and SAVE_MODEL:
            save_checkpoint(model, optimizer, epoch, f"{checkpoint_dir}/compression_epoch{epoch}.pth")



if __name__ == "__main__":
    device = DEVICE

    # Dataset and DataLoader
    train_dataset = DIV2KDataset("DIV2K_train_HR", crop_size=IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Model
    model = Compression(in_channels=3, dim=64, latent_channels=192, out_channels=3)

    # Training
    os.makedirs("checkpoints", exist_ok=True)
    train(model, train_loader, device, epochs=EPOCHS, lr=LR, lambda_rd=LAMBDA)
