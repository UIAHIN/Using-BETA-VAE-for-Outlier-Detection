from torch import nn
import torch

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=1.0):
        super(BetaVAE, self).__init__()
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # Output mean and log variance
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Assuming input is normalized between 0 and 1
        )

    def encode(self, x):
        params = self.encoder(x)
        mean = params[:, :params.size(1) // 2]
        log_var = params[:, params.size(1) // 2:]
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mean, log_var

    def loss_function(self, recon_x, x, mean, log_var):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return BCE + self.beta * KLD  # Weighted by beta for the VAE loss

def train_beta_vae(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0]
            recon_x, mean, log_var = model(x)
            loss = model.loss_function(recon_x, x, mean, log_var)
            loss.backward()
            optimizer.step()