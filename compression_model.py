import torch
import torch.nn as nn
from model import Encoder, Decoder, HyperEncoder, HyperDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class Compression(nn.Module):
    def __init__(self, in_channels, dim, latent_channels, out_channels, z_channels= 96, device = "cpu"):
        super().__init__()

        self.encoder = Encoder(in_channels, dim, latent_channels, device = device)
        self.h_enc = HyperEncoder(latent_channels, z_channels)
        self.h_dec = HyperDecoder(latent_channels, z_channels)
        self.decoder = Decoder(latent_channels, dim, out_channels, device = device)
        self.entropy_bottleneck = EntropyBottleneck(z_channels)
        self.gaussian_cond = GaussianConditional(None)

    def forward(self, x):
        y = self.encoder(x)

        z = self.h_enc(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        mu, sigma = self.h_dec(z_hat)
        y_hat, y_likelihoods = self.gaussian_cond(y, sigma)

        x_hat = self.decoder(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
            }
        }
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    compression_model = Compression(3, 64, 192, 3).to(device)

    x_hat, y_hat, y_likelihoods, z_likelihoods = compression_model(input_tensor)

    print(x_hat.shape, end="\n")
    print(y_hat.shape, end="\n")
    print(y_likelihoods.shape, end="\n")
    print(z_likelihoods.shape, end="\n")
