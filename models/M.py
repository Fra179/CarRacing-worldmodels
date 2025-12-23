import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Allow running both as a module (python -m models.M) and as a script (python models/M.py)
try:
    from models.V import VAE
except ImportError:  # fallback when executed directly from repo root
    from V import VAE

class MDN(nn.Module):
    
    def __init__(self, input_size, output_size, K, units=512):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.K = K
        
        self.l1 = nn.Linear(input_size, 3 * K * output_size)
        
        self.oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi)
        
    def forward(self, x):
        batch_size, seq_len = x.shape[0],x.shape[1]

        out = self.l1(x)
        pi, sigma, mu  = torch.split(out, (self.K * self.output_size , self.K * self.output_size, self.K * self.output_size), dim=2)
        
         
        sigma = sigma.view(batch_size, seq_len, self.K, self.output_size)
        sigma = torch.exp(sigma)
        
        mu = mu.view(batch_size, seq_len, self.K, self.output_size)

        pi = pi.view(batch_size, seq_len, self.K, self.output_size)
        pi = F.softmax(pi, dim=2)
        
        return pi, sigma, mu
    
    def gaussian_distribution(self, y, mu, sigma):
        y = y.unsqueeze(2).expand_as(sigma)
        
        out = (y - mu) / sigma
        out = -0.5 * (out * out)
        out = (torch.exp(out) / sigma) * self.oneDivSqrtTwoPI

        return out
    
    def loss(self, y, pi, mu, sigma):

        out = self.gaussian_distribution(y, mu, sigma)
        out = out * pi
        out = torch.sum(out, dim=2)
        
        # kill (inf) nan loss
        out[out <= float(1e-24)] = 1
        
        out = -torch.log(out)
        out = torch.mean(out)
        
        return out

class MDN_LSTM_DataSet(Dataset):
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = filter(lambda x: x.endswith('.npz'), os.listdir(data_dir))
        self.data_files = map(lambda x: os.path.join(data_dir, x), self.data_files)
        self.data_files = list(self.data_files)

        model = VAE.load_from_checkpoint('residual_vae_final.ckpt', latent_size=32).to('cpu')
        model.eval()
        self.vae = model

    def __len__(self):
        return len(self.data_files) 
    
    def __getitem__(self, index):
        data = np.load(self.data_files[index], allow_pickle=True)
        obs = torch.from_numpy(data["obs"]).permute(0,3,1,2).float()
        actions = data["action"]

        if obs.max() > 1.0:
            obs = obs / 255.0

        with torch.no_grad():
            mu, logvar = self.vae.encode(obs)
            z = self.vae.latent(mu, logvar).numpy()  # (episode_size, latent_size)

        combined = np.concatenate([z, actions], axis=1)  # (episode_size, latent_size + action_dim)
        x = combined[:-1].astype(np.float32)  # inputs for 0..episode_size-2
        y = z[1:].astype(np.float32)   # targets for 1...episode_size-1
        
        return torch.from_numpy(x), torch.from_numpy(y)
    
class MDN_LSTM(pl.LightningModule):

    def __init__(self, input_size, output_size, mdn_units=512, hidden_size=256, num_mixs=5):
        super(MDN_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_mixs = num_mixs
        self.input_size = input_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.mdn = MDN(hidden_size, output_size, num_mixs, mdn_units)

    def forward(self, x, state=None):
        
        y = None
        if state is None:
            y, state = self.lstm(x)
        else:
            y, state = self.lstm(x, state)
        
        pi, sigma, mu = self.mdn(y)
        
        return pi, sigma, mu, state
            
    def forward_lstm(self, x, state=None):
        
        y = None
        x = x.unsqueeze(0) # batch first
        if state is None:
            y, state = self.lstm(x)
        else:
            y, state = self.lstm(x, state)

        return y, state

    def loss(self, y, pi, mu, sigma):
        loss = self.mdn.loss(y, pi, mu, sigma)
        return loss

    def get_hidden_size(self):
        return self.hidden_size
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-3)
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        # Concatenate z and actions as input
        pi, sigma, mu, _ = self.forward(x)
        loss = self.loss(y, pi, mu, sigma)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    

def main():
    dataset = MDN_LSTM_DataSet(data_dir="car_racing_data")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)
    model = MDN_LSTM(input_size=32+3, output_size=32, mdn_units=512, hidden_size=512, num_mixs=5)
    
    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="mdn-{epoch:02d}-{train_loss:.4f}",
    )
    
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb],
    )
    trainer.fit(model, dataloader)
    print("Training complete. Saving final checkpoint...")
    trainer.save_checkpoint("MDN_LSTM_checkpoint.ckpt")
    print("Final checkpoint saved as MDN_LSTM_checkpoint.ckpt")
    print("Best checkpoint saved by Lightning under lightning_logs/.../checkpoints/")

if __name__ == "__main__":
    main()