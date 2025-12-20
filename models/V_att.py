import torch

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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

class Encoder(nn.Module):
    def __init__(self, latent_size=32):
        super().__init__()

        self.enc = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            ResidualBlock(3, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
        )

        self.multihead = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_size)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_size)

    def forward(self, x):
        x = self.enc(x)
        bs = x.size(0)
        x_flat = x.view(bs, 256, -1).transpose(1, 2)  # (B, Seq, Embedding)
        x, _ = self.multihead(x_flat, x_flat, x_flat)  # Apply MHA
        x = x.transpose(1, 2).reshape(bs, -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var

class LightningVModel(pl.LightningModule):
    def __init__(self, latent_size=32, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # --- Encoder ---
        self.enc = Encoder(latent_size=latent_size)
        self.dec_fc = nn.Linear(latent_size, 256 * 4 * 4)

        self.lr = lr

        # --- Decoder ---
        self.multihead = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.dec = nn.Sequential(
            ResidualBlock(256, 128, stride=1),
            nn.Upsample(scale_factor=2),
            ResidualBlock(128, 64, stride=1),
            nn.Upsample(scale_factor=2),
            ResidualBlock(64, 32, stride=1),
            nn.Upsample(scale_factor=2),
        )

        self.dec_out = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(x.size(0), 256, 4, 4)  # (B, 256, 4, 4)

        # Reshape per MHA: (B, seq_len, embed_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, 256, -1).transpose(1, 2)  # (B, 16, 256)
        x, _ = self.multihead(x, x, x)  # Apply MHA

        # Reshape back: (B, 256, 4, 4)
        x = x.transpose(1, 2).view(batch_size, 256, 4, 4)

        x = self.dec(x)
        return self.dec_out(x)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, list) else batch
        recon, mu, logvar = self(x)

        # MSE + KL
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.data_dir, self.files[idx]))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img  # Add batch dimension

def main():
    # train the model, images are in the car_racing_data folder in npy format

    dataset = CarRacingDataset('../car_racing_data')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LightningVModel()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, dataloader)
    model_path = 'vae_model.ckpt'
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()