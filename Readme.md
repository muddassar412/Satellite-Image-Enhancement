# 🌍 Satellite Image Enhancement with UNet + Transformer + GAN

This repository implements a **GAN-based model with UNet + Transformer generator** for satellite image enhancement, denoising, and reconstruction.  
The project combines **transformers, perceptual loss, and adversarial training** to improve image quality while preserving structural details.

---

## 🚀 Features
- **UNet + Transformer Generator**  
  - Encoder-decoder UNet with a Transformer bottleneck for global context.
- **PatchGAN Discriminator**  
  - Spectrally normalized patch-based discriminator for stable adversarial training.
- **Hybrid Loss Function**  
  - Adversarial loss (BCE)  
  - L1 reconstruction loss  
  - VGG-based perceptual loss  
  - SSIM & PSNR for structural similarity and sharpness  
- **Instance Noise**  
  - Improves discriminator generalization and prevents overfitting.
- **Training Logs & Outputs**  
  - Saves training curves, before/after images, and models.  
  - Training metrics (D Loss, G Loss, D Accuracy) logged into CSV.

---

## 📂 Project Structure
