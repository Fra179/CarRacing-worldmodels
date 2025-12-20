import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob


# --- 1. Residual Block Definition ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First Conv Layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second Conv Layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (if dimensions change, we project x)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# --- 2. The Residual VAE Module ---
class ResidualVAE(pl.LightningModule):
    def __init__(self, latent_size=32, lr=1e-3, kl_weight=0.0001):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            # Input: (3, 64, 64)
            ResidualBlock(3, 32, stride=2),  # -> (32, 32, 32)
            ResidualBlock(32, 64, stride=2),  # -> (64, 16, 16)
            ResidualBlock(64, 128, stride=2),  # -> (128, 8, 8)
            ResidualBlock(128, 256, stride=2)  # -> (256, 4, 4)
        )

        # Flatten size: 256 channels * 4 * 4
        self.flat_size = 256 * 4 * 4

        # Latent Projections
        self.fc_mu = nn.Linear(self.flat_size, latent_size)
        self.fc_logvar = nn.Linear(self.flat_size, latent_size)

        # Decoder Input
        self.decoder_input = nn.Linear(latent_size, self.flat_size)

        # Decoder: Mirror of Encoder (using Upsample + ResBlock)
        self.decoder = nn.Sequential(
            ResidualBlock(256, 128, stride=1),
            nn.Upsample(scale_factor=2),  # -> 8x8

            ResidualBlock(128, 64, stride=1),
            nn.Upsample(scale_factor=2),  # -> 16x16

            ResidualBlock(64, 32, stride=1),
            nn.Upsample(scale_factor=2),  # -> 32x32

            ResidualBlock(32, 32, stride=1),  # Extra block to smooth features
            nn.Upsample(scale_factor=2),  # -> 64x64

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output 0-1
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def training_step(self, batch, batch_idx):
        # Unwrap batch if it's a list (some dataloaders do this)
        x = batch[0] if isinstance(batch, list) else batch

        recon, mu, logvar = self(x)

        # 1. Reconstruction Loss (MSE)
        # Using reduction='mean' makes loss independent of image size
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # 2. KL Divergence
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch

        # Total Loss
        loss = recon_loss + (self.hparams.kl_weight * kl_loss)

        self.log("train_loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("kl_loss", kl_loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(f"{data_dir}/*.npz")  # Assuming .npz from previous script
        if not self.files:
            # Fallback for .npy if you used a different script
            self.files = glob.glob(f"{data_dir}/*.npy")
            self.mode = 'npy'
        else:
            self.mode = 'npz'

        print(f"Found {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        if self.mode == 'npz':
            # Load full episode -> Random Frame
            data = np.load(file_path)
            obs = data['obs']  # Shape (Seq_Len, 64, 64, 3)
            # Pick one random frame from the episode to train on
            # This avoids loading the whole episode into VRAM
            frame_idx = np.random.randint(0, len(obs))
            img = obs[frame_idx]
        else:
            # Load single frame .npy
            img = np.load(file_path)

        # Preprocess
        # Convert to Tensor (C, H, W) and Float 0-1
        img_tensor = torch.from_numpy(img).float()
        if img_tensor.max() > 1.0:
            img_tensor /= 255.0

        # Ensure channel first: (H, W, C) -> (C, H, W)
        if img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor


def main():
    # --- Configuration ---
    DATA_DIR = "../car_racing_data"  # Path to your .npz or .npy files
    BATCH_SIZE = 64
    MAX_EPOCHS = 20
    LATENT_SIZE = 32

    # 1. Setup Data
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found. Please run the data collection script first.")
        return

    dataset = CarRacingDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. Setup Model
    model = ResidualVAE(latent_size=LATENT_SIZE, lr=1e-3, kl_weight=0.0001)

    # 3. Trainer
    # Accelerator='auto' will automatically pick GPU if available
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True
    )

    print("Starting training...")
    trainer.fit(model, dataloader)

    # 4. Save
    model_path = "residual_vae_final.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Training complete. Model saved to {model_path}")


if __name__ == '__main__':
    # Fix for Windows DataLoaders performance
    torch.set_float32_matmul_precision('medium')
    main()