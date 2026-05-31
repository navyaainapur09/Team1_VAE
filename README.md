# Reproduction and Extension of Variational Autoencoders using PyTorch-VAE

## Overview

This project reproduces and extends the PyTorch-VAE repository by implementing and evaluating multiple Variational Autoencoder (VAE) architectures on the CelebA dataset. In addition to reproducing existing models, a novel **PerceptualVAE** model was developed using VGG16-based perceptual loss to improve image reconstruction quality.

This project was completed as part of the **Deep Learning Course Project (2025–2026)**.

---

## Project Details

| Attribute | Details |
|------------|----------|
| Course | Deep Learning |
| Academic Year | 2025–2026 |
| Framework | PyTorch 2.x, PyTorch Lightning 1.9.5 |
| Platform | Kaggle (Tesla P100 GPU, 16GB VRAM) |
| Dataset | CelebA + dSprites |
| Base Repository | PyTorch-VAE |

---

## Objectives

- Reproduce baseline VAE models from the PyTorch-VAE repository.
- Compare reconstruction quality across different VAE variants.
- Implement a new PerceptualVAE using VGG16 perceptual loss.
- Evaluate disentanglement using MIG score.
- Perform hyperparameter tuning and ablation studies.

---

## Models Implemented

### Existing Models

- VanillaVAE
- BetaVAE
- BetaTCVAE
- MSSIMVAE

### Proposed Model

- **PerceptualVAE** ⭐

PerceptualVAE extends VanillaVAE by incorporating VGG16 feature-based perceptual loss to generate sharper and more realistic reconstructions.

---

## Datasets

### CelebA Dataset

- Total Images: 202,599
- Resolution: 64 × 64
- Training Split: 80%
- Validation Split: 10%
- Test Split: 10%

### dSprites Dataset

- Total Images: 737,280
- Resolution: 64 × 64
- Used for disentanglement evaluation
- Ground Truth Factors:
  - Shape
  - Scale
  - Rotation
  - X Position
  - Y Position

---

## Model Architecture

```text
Input Image (64×64×3)
          ↓
Encoder (Conv2D ×4)
          ↓
Latent Space (128-Dim)
          ↓
Reparameterization
          ↓
Decoder (ConvTranspose2D ×4)
          ↓
Reconstructed Image
```

---

## Loss Functions

### VanillaVAE

```text
Loss = MSE + KL Divergence
```

### BetaVAE

```text
Loss = MSE + β × KL Divergence
```

### BetaTCVAE

```text
Loss = MSE + MI + TC + DWKL
```

### MSSIMVAE

```text
Loss = (1 − SSIM) + KL Divergence
```

### PerceptualVAE

```text
Loss = MSE + KL Divergence + 0.1 × Perceptual Loss
```

---

## Hyperparameters

| Parameter | Value |
|------------|--------|
| Batch Size | 64 |
| Latent Dimension | 128 |
| Optimizer | Adam |
| Learning Rate | 0.005 |
| Epochs | 5 |

---

## Evaluation Metrics

| Metric | Description |
|----------|-------------|
| MSE | Mean Squared Error |
| SSIM | Structural Similarity Index |
| PSNR | Peak Signal-to-Noise Ratio |
| MIG | Mutual Information Gap |

---

## Results

| Model | MSE ↓ | SSIM ↑ | PSNR ↑ | MIG ↑ |
|---------|---------|---------|---------|---------|
| VanillaVAE | ~0.020 | ~0.55 | ~17 dB | ~0.06 |
| BetaVAE | ~0.047 | ~0.48 | ~14 dB | ~0.12 |
| BetaTCVAE | ~0.055 | ~0.45 | ~13 dB | N/A |
| MSSIMVAE | ~0.055 | ~0.52 | ~13 dB | N/A |
| PerceptualVAE ⭐ | ~0.018 | ~0.58 | ~18 dB | N/A |

---

## Key Findings

- PerceptualVAE achieved the best reconstruction quality.
- BetaVAE improved latent disentanglement compared to VanillaVAE.
- β = 4 provided the best balance between reconstruction and disentanglement.
- Learning rate 0.005 achieved the most stable convergence.
- All models showed consistent training loss reduction.

---

## Generated Outputs

The project generates:

- Reconstruction comparison images
- PerceptualVAE analysis plots
- Hyperparameter tuning visualizations
- Latent traversal visualizations
- t-SNE latent space plots
- Metrics CSV reports

### Output Files

```text
phase2_comparison.png
phase3_perceptual_comparison.png
phase3_analysis.png
phase4_metrics.csv
phase4_reconstruction.png
phase4_error_analysis.png
phase4_tsne.png
latent_traversal_dsprites.png
```

---

## Installation

```bash
git clone https://github.com/AntixK/PyTorch-VAE.git
cd PyTorch-VAE
pip install -r requirements.txt
```

---

## Running the Project

```bash
python run.py
```

or

```bash
python experiment.py
```

---

## Novel Contribution: PerceptualVAE

PerceptualVAE is the primary contribution of this project.

Features:

- Uses pretrained VGG16 features.
- Computes perceptual similarity instead of relying only on pixel-wise loss.
- Produces sharper facial reconstructions.
- Improves SSIM and PSNR scores.
- Reduces reconstruction error.

---

## Future Work

- Train on CelebA-HQ (256×256 resolution).
- Increase training to 100+ epochs.
- Implement FactorVAE and DIP-VAE.
- Explore Vision Transformer based VAEs.
- Deploy using ONNX for real-time inference.
- Extend to Conditional Variational Autoencoders (CVAE).

---

## References

1. Kingma & Welling, Auto-Encoding Variational Bayes, ICLR 2014.
2. Higgins et al., β-VAE, ICLR 2017.
3. Chen et al., Beta-TCVAE, NeurIPS 2018.
4. Zhao et al., InfoVAE, AAAI 2019.
5. AntixK, PyTorch-VAE Repository.
6. CelebA Dataset.
7. dSprites Dataset.
8. Simonyan & Zisserman, VGG16.
9. Johnson et al., Perceptual Losses.
10. PyTorch Lightning.

---

## Acknowledgements

This work reproduces and extends the PyTorch-VAE repository and introduces PerceptualVAE as a novel enhancement for improving image reconstruction quality using perceptual loss.
