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
├── app.py # Training script

├── README.md # Project documentation

├── requirements.txt # Dependencies

├── /result # Training logs and saved models

│ ├── before/ # Input images (sampled during training)

│ ├── after/ # Generated images (sampled during training)

│ ├── training_log.csv # CSV with D/G losses and accuracy

│ ├── generator.pth # Saved Generator model

│ └── discriminator.pth # Saved Discriminator model



---

## ⚙️ Requirements
Install dependencies before running:
```bash
pip install torch torchvision piq matplotlib Pillow   ```



If using Google Colab:
```bash
from google.colab import drive
drive.mount('/content/drive') ```



##**🏋️ Training**

Run training with:

python app.py

## **Default arguments:**

epochs=150

batch_size=16

lr_g=3e-6 (Generator learning rate)

lr_d=3e-5 (Discriminator learning rate)

**Change dataset path inside train():**

train(data_root='/path/to/your/images', epochs=150, batch_size=16, device='cuda')

##**📊 Results**

**During training:**

Saves before/after images every 10 epochs.

Logs D Loss, G Loss, and Discriminator Accuracy to training_log.csv.

At the end, training progress plot is displayed with loss curves.

**Example outputs:**

Input (before) → Enhanced output (after).

##**🧠 Model Architecture**
Generator (UNet + Transformer)

Encoder: 3 convolutional down-sampling blocks.

Bottleneck: TransformerBlock (multi-head self-attention + feed-forward).

Decoder: Upsampling with skip connections + convolutional refinement.

Output: 3-channel RGB image with sigmoid.

Discriminator (PatchGAN)

Patch-level classification with spectral normalization.

Outputs real/fake probability maps.

**📈 Loss Functions**

Total Generator Loss =
Adversarial Loss + L1 Loss (×10) + Perceptual Loss (×5) + SSIM Loss (×2) + PSNR Loss (×0.1)

💾 Saving & Checkpoints

Models saved automatically after training in result/:

generator.pth

discriminator.pth

📌 Future Work

Multi-scale transformer blocks.

Attention-based discriminator.

Support for multispectral (beyond RGB) satellite images.

✨ Citation

If you use this code in your research, please cite the repository.

📜 License

MIT License. You are free to use and modify with proper attribution.

