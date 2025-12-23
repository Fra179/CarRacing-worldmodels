import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob



class VAE(pl.LightningModule):
    def __init__(self, latent_size: int, lr: float = 1e-3, kl_weight: float = 0.0001):
        super().__init__()
        self.latent_size = latent_size
        self.lr = lr
        self.kl_weight = kl_weight
        
        # encoder
        self.enc_conv1 = nn.Conv2d(3,32,kernel_size=4,stride=2, padding=0)
        self.enc_conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2, padding=0)
        self.enc_conv3 = nn.Conv2d(64,128,kernel_size=4,stride=2, padding=0)
        self.enc_conv4 = nn.Conv2d(128,256,kernel_size=4,stride=2, padding=0)
        
        # z
        self.mu = nn.Linear(1024, latent_size)
        self.logvar = nn.Linear(1024, latent_size)
        
        # decoder
        self.dec_conv1 = nn.ConvTranspose2d(latent_size, 128, kernel_size=5, stride=2, padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        out = self.decode(z)
        
        return out, mu, logvar   
        
    def encode(self, x):
        batch_size = x.shape[0]
        
        out = F.relu(self.enc_conv1(x))
        out = F.relu(self.enc_conv2(out))
        out = F.relu(self.enc_conv3(out))
        out = F.relu(self.enc_conv4(out))
        out = out.reshape(batch_size, 1024)
        
        mu = self.mu(out)
        logvar = self.logvar(out)
        
        return mu, logvar
        
    def decode(self, z):
        batch_size = z.shape[0]
        
        out = z.view(batch_size, self.latent_size, 1, 1)
        out = F.relu(self.dec_conv1(out))
        out = F.relu(self.dec_conv2(out))
        out = F.relu(self.dec_conv3(out))
        out = torch.sigmoid(self.dec_conv4(out))
        
        return out
        
        
    def latent(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar).to(self.device)
        z = mu + eps*sigma
        
        return z
    
    def obs_to_z(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        
        return z

    def sample(self, z):
        out = self.decode(z)
        
        return out
    
    def vae_loss(self, out, y, mu, logvar):
        BCE = F.mse_loss(out, y, reduction="sum")
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KL
        return loss, BCE, KL

    def training_step(self, batch, batch_idx):
        x = batch
        recon, mu, logvar = self(x)

        loss, bce, kl_loss = self.vae_loss(recon, x, mu, logvar)

        # Log per-epoch to enable ModelCheckpoint monitoring
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("recon_loss", bce, prog_bar=True, on_step=False, on_epoch=True)
        self.log("kl_loss", kl_loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(f"{data_dir}/*.npy")

        print(f"Found {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

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
    DATA_DIR = "car_racing_data" 
    BATCH_SIZE = 64
    MAX_EPOCHS = 20
    LATENT_SIZE = 32

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found. Please run the data collection script first.")
        return

    dataset = CarRacingDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. Setup Model
    model = VAE(latent_size=LATENT_SIZE) #, lr=1e-3, kl_weight=0.0001)

    # 3. Trainer
    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="vae-{epoch:02d}-{train_loss:.4f}",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_cb],
    )

    print("Starting training...")
    trainer.fit(model, dataloader)

    # 4. Save
    model_path = "residual_vae_final.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Training complete. Final model saved to {model_path}")
    print("Best checkpoint saved by Lightning under lightning_logs/.../checkpoints/")

    # Visualize with matplotlib a couple of examples
    import matplotlib.pyplot as plt
    model.eval()
    sample_batch = next(iter(dataloader))
    with torch.no_grad():
        recon_batch, _, _ = model(sample_batch.to(model.device))
    sample_batch = sample_batch.cpu()
    recon_batch = recon_batch.cpu()
    for i in range(5):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(sample_batch[i].permute(1, 2, 0))
        axes[0].set_title("Original")
        axes[1].imshow(recon_batch[i].permute(1, 2, 0))
        axes[1].set_title("Reconstructed")
        plt.show()
        plt.imsave(f"reconstruction_{i}.png", recon_batch[i].permute(1, 2, 0).numpy())



if __name__ == '__main__':
    # Fix for Windows DataLoaders performance
    torch.set_float32_matmul_precision('medium')
    main()