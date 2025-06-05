import torch
import torch.nn.functional as F
import torch.nn as nn
from gdn import GDN
from ffct import FFCTranspose

class Encoder(nn.Module):
    def __init__(self, in_channels, dim, latent_channels = 192, alpha = 0.5, device = "cpu"):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1), # 128x128
            GDN(dim, device = device)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1), # 64x64
            GDN(dim*2, device = device)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(dim*2, dim*2*2, kernel_size=3, stride=2, padding=1), # 32x32
            GDN(dim*2*2, device = device)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(dim*2*2, latent_channels, kernel_size=3, stride = 2,padding=1), # 16x16
            nn.Tanh()
        )
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        z = self.enc4(x3)

        return z

class Decoder(nn.Module):
    def __init__(self, latent_channels, dim, out_channels, alpha = 0.5, device = "cpu"):
        super().__init__()
        # self.dec1 = nn.Sequential(
        #     nn.ConvTranspose2d(latent_channels, dim*2*2, kernel_size = 3, stride = 2, padding = 1, output_padding=1), # 32x32
        #     GDN(dim*2*2, inverse=True)
        # )
        # self.dec2 = nn.Sequential(
        #     nn.ConvTranspose2d(dim*2*2, dim*2, kernel_size = 3, stride = 2, padding = 1, output_padding=1), # 64x64
        #     GDN(dim*2, inverse=True)
        # )
        # self.dec3 = nn.Sequential(
        #     nn.ConvTranspose2d(dim*2, dim, kernel_size = 3, stride = 2, padding = 1, output_padding=1), # 128x128
        #     GDN(dim, inverse=True)
        # )
        # self.dec4 = nn.Sequential(
        #     nn.ConvTranspose2d(dim, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding=1), # 256x256
        #     nn.Tanh()
        # )
        self.dec1 = nn.Sequential(
            FFCTranspose(latent_channels, dim*2*2,alpha = alpha), # 32x32
            GDN(dim*2*2, inverse=True, device = device)
        )
        self.dec2 = nn.Sequential(
            FFCTranspose(dim*2*2, dim*2,alpha = alpha), # 64x64
            GDN(dim*2, inverse=True, device = device)
        )
        self.dec3 = nn.Sequential(
            FFCTranspose(dim*2, dim,alpha = alpha), # 128x128
            GDN(dim, inverse=True, device = device)
        )
        self.dec4 = nn.Sequential(
            FFCTranspose(dim, out_channels,alpha = alpha), # 256x256
            nn.Tanh()
        )

    def forward(self, y_hat):
        d1 = self.dec1(y_hat)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        x_hat = self.dec4(d3)

        return x_hat

class HyperEncoder(nn.Module):
    def __init__(self, latent_channels=192, z_channels=96, alpha = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, z_channels, kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, y):
        z = self.net(y)
        return z

class HyperDecoder(nn.Module):
    def __init__(self, latent_channels=192, z_channels=96, alpha = 0.5):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.ConvTranspose2d(z_channels, latent_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(latent_channels, latent_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        # )
        self.net = nn.Sequential(
            FFCTranspose(z_channels, latent_channels, alpha),
            nn.ReLU(inplace=True),
            FFCTranspose(latent_channels, latent_channels, alpha),
            nn.ReLU(inplace=True),
            FFCTranspose(latent_channels, latent_channels * 2, alpha)
        )
    
    def forward(self, z_hat):
        gaussian_params = self.net(z_hat)
        mu, sigma = torch.chunk(gaussian_params, chunks=2, dim=1)
        return mu, F.softplus(sigma) + 1e-6 

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    encoder = Encoder(3, 64, 192).to(device)
    h_enc = HyperEncoder(192).to(device)
    h_dec = HyperDecoder(192).to(device)
    decoder = Decoder(192, 64, 3).to(device)

    # Encode image
    y = encoder(input_tensor)

    # HyperEncoder
    z = h_enc(y)

    # HyperDecoder
    mu, sigma = h_dec(z)

    # Decode
    x_hat = decoder(y)

    print(f"Input: {input_tensor.shape}")
    print(f"Latent y: {y.shape}")
    print(f"Hyperlatent z: {z.shape}")
    print(f"Reconstructed x̂: {x_hat.shape}")
