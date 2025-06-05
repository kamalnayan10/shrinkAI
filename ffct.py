import torch
import torch.nn as nn


class FFCTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, alpha):
        super().__init__()

        self.c_g_in = max(1, int(in_channels * alpha)) if alpha > 0 else 0
        self.c_l_in = in_channels - self.c_g_in

        self.c_g_out = int(out_channels * alpha)
        self.c_l_out = out_channels - self.c_g_out

        # Local → Local Transpose Conv
        self.local_conv = nn.ConvTranspose2d(self.c_l_in, self.c_l_out, kernel_size=4, stride=2, padding=1)

        # Global → Global path (via FFT)
        self.global_conv = nn.ConvTranspose2d(self.c_g_in * 2, self.c_g_out * 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.c_g_out * 2)

        # Cross-talk
        self.g2l_conv = nn.ConvTranspose2d(self.c_g_in, self.c_l_out, kernel_size=4, stride=2, padding=1)
        self.l2g_conv = nn.ConvTranspose2d(self.c_l_in, self.c_g_out, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_l = x[:, :self.c_l_in, :, :]
        x_g = x[:, self.c_l_in:, :, :]

        y_conv_l = self.local_conv(x_l)

        if self.c_g_in > 0:
            y = torch.fft.rfft2(x_g, norm="ortho")
            y_r = y.real
            y_i = y.imag
            y_cat = torch.cat([y_r, y_i], dim=1)

            y_conv_g = self.relu(self.bn(self.global_conv(y_cat)))
            y_r_out, y_i_out = torch.chunk(y_conv_g, 2, dim=1)
            y_complex = torch.complex(y_r_out, y_i_out)
            y_g = torch.fft.irfft2(y_complex, s=(x.shape[-2] * 2, x.shape[-1] * 2), norm="ortho")

            y_conv_l += self.g2l_conv(x_g)
            y_g += self.l2g_conv(x_l)

            return torch.cat([y_conv_l, y_g], dim=1)
        else:
            return y_conv_l

def test_ffc_transpose():
    B = 2                # Batch size
    C_in = 64            # Input channels
    C_out = 128          # Output channels
    H, W = 16, 16        # Input size
    alpha = 0.5

    model = FFCTranspose(C_in, C_out, alpha)
    x = torch.randn(B, C_in, H, W)
    y = model(x)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (B, C_out, H * 2, W * 2), "❌ Upsampled output shape is incorrect!"
    print("✅ FFCTranspose test passed.")


if __name__ == "__main__":
    test_ffc_transpose()
