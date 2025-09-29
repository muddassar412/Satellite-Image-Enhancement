# -*- coding: utf-8 -*-


# ======================= Imports =======================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import csv
from piq import ssim, psnr  # <-- Added for SSIM and PSNR

# ======================= Dataset =======================
class SatelliteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image

# ======================= Instance Noise =======================
def add_instance_noise(images, stddev=0.05):
    noise = torch.randn_like(images) * stddev
    return images + noise

# ======================= Transformer Block =======================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(inplace=True), nn.Linear(ff_dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(2, 0, 1)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x2 = self.norm1(x_flat + attn_out)
        ff_out = self.ff(x2)
        x3 = self.norm2(x2 + ff_out)
        return x3.permute(1, 2, 0).view(B, C, H, W)

# ======================= Generator =======================
class UNetTransformerGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(2)

        self.transformer = TransformerBlock(dim=256, num_heads=8, ff_dim=512)
        self.conv_hidden = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(384, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.conv1(x)
        p1 = self.pool1(e1)
        e2 = self.conv2(p1)
        p2 = self.pool2(e2)
        e3 = self.conv3(p2)
        p3 = self.pool3(e3)

        t = self.transformer(p3)
        h = self.conv_hidden(t)

        u1 = self.up1(h)
        d1 = self.dec1(torch.cat([u1, e3], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u3 = self.up3(d2)
        d3 = self.dec3(torch.cat([u3, e1], dim=1))
        return torch.sigmoid(self.final(d3))

# ======================= Discriminator =======================
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def sn_conv(in_channels, out_channels, stride=2):
            return nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1))

        self.net = nn.Sequential(
            sn_conv(3, 64), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(64, 128), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(128, 256), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(256, 512), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(512, 512), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ======================= Perceptual Loss =======================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return nn.functional.l1_loss(input_features, target_features)

# ======================= Training Function =======================
def train(data_root, epochs=20, batch_size=20, lr_g=3e-6, lr_d=3e-5, device='cuda'):
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = SatelliteDataset(data_root, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = UNetTransformerGenerator().to(device)
    D = PatchDiscriminator().to(device)
    perceptual_loss = VGGPerceptualLoss().to(device)

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    G_losses, D_losses, acc_list = [], [], []
    total_correct_all_epochs = 0
    total_samples_all_epochs = 0

    result_dir = os.path.join(data_root, 'result')
    before_dir = os.path.join(result_dir, 'before')
    after_dir = os.path.join(result_dir, 'after')

    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)

    with open(os.path.join(result_dir, 'training_log.csv'), mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(['Epoch', 'D Loss', 'G Loss', 'D Accuracy'])

        for epoch in range(epochs):
            running_d_loss, running_g_loss = 0.0, 0.0
            correct, total = 0, 0

            for real, target in loader:
                real, target = real.to(device), target.to(device)
                fake = G(real).detach()

                for _ in range(3):
                    opt_D.zero_grad()
                    real_noisy = add_instance_noise(target, 0.05)
                    fake_noisy = add_instance_noise(fake, 0.05)

                    real_pred = D(real_noisy)
                    fake_pred = D(fake_noisy)

                    real_labels = torch.full_like(real_pred, 0.9, device=device)
                    fake_labels = torch.full_like(fake_pred, 0.1, device=device)

                    real_loss = adv_loss(real_pred, real_labels)
                    fake_loss = adv_loss(fake_pred, fake_labels)

                    d_loss = real_loss + fake_loss
                    d_loss.backward()
                    opt_D.step()

                    real_preds_sigmoid = torch.sigmoid(real_pred)
                    fake_preds_sigmoid = torch.sigmoid(fake_pred)
                    correct_real = (real_preds_sigmoid > 0.5).sum().item()
                    correct_fake = (fake_preds_sigmoid < 0.5).sum().item()
                    correct += correct_real + correct_fake
                    total += real_pred.numel() + fake_pred.numel()

                opt_G.zero_grad()
                fake = G(real)
                fake_pred2 = D(fake)

                g_adv = adv_loss(fake_pred2, real_labels)
                g_l1 = l1_loss(fake, target) * 10
                g_perc = perceptual_loss(fake, target) * 5
                g_ssim = (1 - ssim(fake, target, data_range=1.0)) * 2
                g_psnr = (1 / (psnr(fake, target, data_range=1.0) + 1e-8)) * 0.1

                g_loss = g_adv + g_l1 + g_perc + g_ssim + g_psnr
                g_loss.backward()
                opt_G.step()

                running_d_loss += d_loss.item()
                running_g_loss += g_loss.item()

                if epoch % 10 == 0:
                    before_image = transforms.ToPILImage()(real[0].cpu())
                    before_image.save(os.path.join(before_dir, f'{epoch}_{real.shape[0]}.png'))
                    after_image = transforms.ToPILImage()(fake[0].cpu())
                    after_image.save(os.path.join(after_dir, f'{epoch}_{real.shape[0]}.png'))

            D_losses.append(running_d_loss / len(loader))
            G_losses.append(running_g_loss / len(loader))
            accuracy = 100 * correct / total
            acc_list.append(accuracy)

            total_correct_all_epochs += correct
            total_samples_all_epochs += total

            log_writer.writerow([epoch + 1, D_losses[-1], G_losses[-1], accuracy])

            print(f"Epoch [{epoch + 1}/{epochs}] | D Loss: {D_losses[-1]:.4f} | G Loss: {G_losses[-1]:.4f} | D Acc: {accuracy:.2f}%")

        overall_accuracy = 100 * total_correct_all_epochs / total_samples_all_epochs
        print(f"\nOverall Model Accuracy across all epochs: {overall_accuracy:.2f}%")

        plt.plot(D_losses, label='D Loss')
        plt.plot(G_losses, label='G Loss')
        plt.plot(acc_list, label='D Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend()
        plt.title('Training Progress')
        plt.show()

        # ================== SAVE THE MODELS ==================
        torch.save(G.state_dict(), os.path.join(result_dir, 'generator.pth'))
        torch.save(D.state_dict(), os.path.join(result_dir, 'discriminator.pth'))
        print("Models saved successfully!")

# ======================= Run Training =======================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train(data_root='/content/drive/MyDrive/Colab_Datasets/Images', epochs=150, batch_size=16, device=device)
