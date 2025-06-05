# 📦 ShrinkAI: Learned Image Compression with FFC + Hyperprior

**ShrinkAI** is a deep image compression model inspired by Ballé et al. (2018), enhanced with **Fast Fourier Convolutions (FFC)** in the decoder and hyperdecoder for better global feature reconstruction. It uses a **hyperprior-based entropy model** to achieve competitive compression rates while maintaining visual quality.

---

## 🧠 Model Overview

ShrinkAI compresses images using the following pipeline:

```
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
Decoder (FFCTranspose + Inverse GDN)
   ↓
Reconstructed Image
```

---

## 🧪 Results

| Metric        | Value                   |
| ------------- | ----------------------- |
| **BPP**       | 0.83                    |
| **PSNR**      | 26.01 dB                |
| **Dataset**   | DIV2K (256×256 patches) |
| **Framework** | PyTorch + CompressAI    |

> ✅ Achieves a strong rate–distortion trade-off on natural images with learned entropy coding.

---

## 🏗️ Architecture Components

### 🔹 Encoder
- 4-stage convolutional encoder
- GDN (Generalized Divisive Normalization) activation
- Produces latent space `y`

### 🔹 Hyperprior
- `HyperEncoder`: encodes `y` into `z`
- `EntropyBottleneck`: compresses `z` into bitstream
- `HyperDecoder`: reconstructs distribution parameters (μ, σ) using FFCTranspose layers

### 🔹 Decoder
- 4-stage **FFCTranspose** decoder
- Inverse GDN
- Reconstructs image from quantized `y`

---

## 🏋️ Training Configuration

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

## ⚙️ Setting Up Environment

### ✅ Step-by-step Setup

```bash
# Clone the repository
git clone https://github.com/kamalnayan10/shrinkAI.git
cd shrinkAI

# Create a virtual environment (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# For Windows:
# python -m venv venv
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 CLI Usage

### 🔐 Compression

```bash
python compress.py --input test.png --output compressed.bin
```

### 🖼️ Decompression

```bash
python decompress.py --input compressed.bin --output test_output.png
```

---

### 📸 Compression Example

<div align="center">
  <table>
    <tr>
      <td align="center"><strong>Original Image</strong></td>
      <td align="center"><strong>Reconstructed Image</strong></td>
    </tr>
    <tr>
      <td><img src="test.png" alt="Original Image" width="300"/></td>
      <td><img src="test_output.png" alt="Reconstructed Image" width="300"/></td>
    </tr>
  </table>
</div>

<p align="center">
  <strong>🧾 Compression Stats:</strong><br>
  <code>Original Image:</code> 5.0 MB → <code>Compressed File:</code> 1.5 MB → <code>Reconstructed Image:</code> 4.7 MB<br>
  <em>~70% reduction in transmission/storage size with near-lossless reconstruction.</em>
</p>

---

## 🙌 Acknowledgements

- 📜 Based on [Ballé et al. (2018)](https://arxiv.org/abs/1802.01436)
- 🔧 GDN module from [jorge-pessoa/pytorch-gdn](https://github.com/jorge-pessoa/pytorch-gdn)
- 🌐 FFC layers adapted from [Fast Fourier Convolution (NIPS 2020)](https://papers.nips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this software for both personal and commercial purposes, provided that proper attribution is given.
