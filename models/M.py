from torch import nn
import pytorch_lightning as pl

class M(pl.LightningModule):
    def __init__(self, latent_size=32, action_size=3, hidden_size=256, num_gaussians=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(latent_size + action_size, hidden_size, batch_first=True)
        self.fc_pi = nn.Linear(hidden_size, num_gaussians)
        self.fc_mu = nn.Linear(hidden_size, num_gaussians * latent_size)
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians * latent_size)

    def forward(self, z, action, hidden_state=None):
        # z: (Batch, Seq, Latent)
        # action: (Batch, Seq, Action)
        x = torch.cat([z, action], dim=2)
        out, hidden = self.lstm(x, hidden_state)

        # MDN heads
        pi = F.softmax(self.fc_pi(out), dim=2)
        mu = self.fc_mu(out).view(-1, out.size(1), self.hparams.num_gaussians, self.hparams.latent_size)
        sigma = torch.exp(self.fc_sigma(out)).view(-1, out.size(1), self.hparams.num_gaussians,
                                                   self.hparams.latent_size)

        return pi, mu, sigma, hidden

    def mdn_loss_function(self, pi, mu, sigma, target_z):
        """
        Calculates negative log-likelihood of target_z under the Mixture of Gaussians.
        target_z: (Batch, Seq, Latent)
        """
        # Expand target for broadcasting against Num_Gaussians
        # Target shape: (Batch, Seq, 1, Latent)
        target = target_z.unsqueeze(2)

        # Calculate Normal Probability: N(target | mu, sigma)
        # Note: We use log_prob for numerical stability
        dist = torch.distributions.Normal(mu, sigma)
        log_probs = dist.log_prob(target)  # Shape: (Batch, Seq, G, Latent)

        # Sum over latent dimensions (assuming diagonal covariance)
        log_probs = torch.sum(log_probs, dim=3)  # Shape: (Batch, Seq, G)

        # Combine with Mixing Coefficients (pi) using LogSumExp trick
        # We want log( sum( pi * N(target) ) )
        # = log_sum_exp( log(pi) + log_probs )
        weighted_log_probs = torch.log(pi + 1e-8) + log_probs
        log_likelihood = torch.logsumexp(weighted_log_probs, dim=2)

        return -torch.mean(log_likelihood)

    def training_step(self, batch, batch_idx):
        # Batch should contain sequences: (z_seq, action_seq, target_z_seq)
        # target_z_seq is essentially z_seq shifted by 1
        z, action, target_z = batch

        pi, mu, sigma, _ = self(z, action)
        loss = self.mdn_loss_function(pi, mu, sigma, target_z)

        self.log("mdn_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)