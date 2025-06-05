# 📦 ShrinkAI: Learned Image Compression with FFC + Hyperprior

**ShrinkAI** is a deep image compression model inspired by Ballé et al. (2018), enhanced with **Fast Fourier Convolutions (FFC)** in the decoder and hyperdecoder for better global feature reconstruction. It uses a **hyperprior-based entropy model** to achieve competitive compression rates while maintaining visual quality.

---

## 🧠 Model Overview

ShrinkAI compresses images using the following pipeline:

Input Image
↓
Encoder (Conv + GDN)
↓
Latent y
↓
HyperEncoder
↓
Latent z
↓
EntropyBottleneck → z_strings
↓
HyperDecoder → μ, σ
↓
GaussianConditional → y_strings
↓
Decoder (FFCTranspose + GDN⁻¹)
↓
Reconstructed Image

---

## 🧪 Results

| Metric        | Value                   |
| ------------- | ----------------------- |
| **BPP**       | 0.83                    |
| **PSNR**      | 26.01 dB                |
| **Dataset**   | DIV2K (256×256 patches) |
| **Framework** | PyTorch + CompressAI    |

> ✅ Achieves good rate–distortion balance on natural images with learned entropy coding.

---

## 🏗️ Architecture Components

### 🔹 Encoder

- 4-stage convolutional layers with GDN activation
- Compresses input to latent space `y`

### 🔹 Hyperprior

- `HyperEncoder` processes `y` into hyper-latents `z`
- `EntropyBottleneck` compresses `z`
- `HyperDecoder` with **FFCTranspose** layers produces Gaussian parameters `μ, σ` for modeling `y`

### 🔹 Decoder

- 4-stage **FFCTranspose** layers with inverse GDN
- Reconstructs the image from `y_hat`

---

## 🏋️ Training Setup

| Setting           | Value                           |
| ----------------- | ------------------------------- |
| **Loss**          | Rate–Distortion Loss (λ = 1e-4) |
| **Optimizer**     | Adam                            |
| **Learning Rate** | 1e-3                            |
| **Batch Size**    | 16                              |
| **Epochs**        | 100                             |
| **Patch Size**    | 256×256                         |
| **Data Aug**      | Random crop                     |

---

## 🧰 Usage

### 🔧 Installation

```bash
git clone https://github.com/kamalnayan10/shrinkAI
cd shrinkAI
```

### CLI Utility

compression

```bash
python compress.py --input test.png --output compressed.bin
```

decompression

```bash
python decompress.py --input compressed.bin --output test_output.png
```

### 🙌 Acknowledgements

- Based on [Ballé et al. (2018)](https://arxiv.org/abs/1802.01436)

- GDN Module code taken from [jorge-pessoa](https://github.com/jorge-pessoa/pytorch-gdn)

- FFC blocks adapted from [Fast Fourier Convolution](https://papers.nips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf)
