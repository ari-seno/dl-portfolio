import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    PyTorch reimplementation of Project 1's NumPy neural network.
    Architecture: Input -> Linear -> ReLU -> Linear -> Sigmoid -> Output
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            y_pred = self.forward(x)
            return (y_pred >= threshold).float()