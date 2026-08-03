import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim=30, latent_dim=8, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(32, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(20, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(20, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    model = Autoencoder(input_dim=30, latent_dim=10)
    dummy_input = torch.randn(8, 30)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")