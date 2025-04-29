# VAE-Based Latent Code Augmentation for Data-Free Black-Box Substitute Attacks

This repository presents an enhanced framework for data-free black-box adversarial attacks. Building upon the Latent Code Augmentation (LCA) pipeline, we replace Stable Diffusion’s encoder and decoder with a lightweight Variational Autoencoder (VAE), enabling efficient and controllable synthetic data generation for training substitute models in the absence of real training data.

## Overview

In substitute model attacks, the attacker trains a surrogate model to mimic the decision boundary of a black-box target by using synthetic data. Traditional methods used GANs or diffusion models, but these approaches either suffer from poor diversity or are computationally expensive. This project introduces a modular VAE-based latent code augmentation pipeline that achieves high-quality image synthesis, effective membership inference, and strong adversarial transferability.

## Key Contributions

- Replaced the Stable Diffusion encoder and decoder with a custom-trained VAE architecture.
- Enabled efficient latent code augmentation using Gaussian perturbation and code mixing.
- Integrated a suite of handcrafted pixel-space transformations (CutMix, RICAP, Gaussian and salt-and-pepper noise, translation).
- Achieved high structural and perceptual similarity with real CIFAR images using LPIPS and L2 metrics.
- Trained a ResNet-34 substitute model using pseudo-labeled VAE-generated data under full data-free constraints.

## Repository Structure

AdvMLProject/ ├── vae.py # VAE architecture with encoder and decoder ├── resnet.py # ResNet-34 model used for target and substitute training ├── stage1.py # Encodes real images using the VAE encoder, saves latent vectors ├── stage2.py # Augments latent codes and decodes them back into images using VAE ├── compute_loss.py # Computes LPIPS and L2 loss between real and generated images ├── my_transform.py # Pixel-space augmentations (e.g., noise, CutMix, RICAP) ├── output/ │ └── checkpoints/ # Contains trained VAE model weights (.pt) ├── result1/ # Stores encoded latent code files (.pt format) ├── result2/ # Stores generated images from augmented latent codes (.png) └── README.md # Project documentation (this



- `stage1.py`: Run this first to obtain the latent representations of real images.
- `stage2.py`: Uses saved latent codes to generate new images for training substitute models.
- `compute_loss.py`: Helps in evaluating how realistic or close the generated images are to the original data.
- `my_transform.py`: Applies various augmentations in pixel space.





## Methodology

### Stage 1: Latent Code Extraction

- Synthetic seed images are encoded using a trained VAE.
- Only the mean vector from the latent distribution is retained for stability.
- Latent codes are saved and filtered using membership inference based on black-box target model confidence.

### Stage 2: Latent Code Augmentation and Decoding

- Latent codes are perturbed using Gaussian noise and mixed across samples.
- The VAE decoder reconstructs images from augmented latent codes.
- Pixel-level augmentations (CutMix, RICAP, spatial translation, noise injection) further enrich the dataset.

### Substitute Training

- Pseudo-labels for generated images are obtained by querying the target model.
- A ResNet-34 substitute model is trained using cross-entropy loss on this labeled synthetic dataset.

## Results

### Quantitative Evaluation

| Metric              | Value                      |
|---------------------|----------------------------|
| LPIPS               | 0.1376                     |
| L2 Loss             | 0.0955                     |
| Membership Recall   | 100% (across 10 classes)   |
| Evaluated Pairs     | 10,000                     |



## Reproducibility

python train_vae.py
python stage1.py
python compute_loss.py
python stage2.py

---
