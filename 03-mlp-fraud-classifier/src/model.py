import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    def __init__(self, input_dim=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = FraudMLP(input_dim=29)
    dummy = torch.randn(8, 29)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(model)