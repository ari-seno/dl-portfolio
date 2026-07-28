import numpy as np
from src.layers import Dense
from src.activations import ReLU, Sigmoid
from src.losses import BinaryCrossEntropy


class NeuralNetwork:
    """
    Simple feedforward neural network for binary classification.
    Architecture (default): Input -> Dense -> ReLU -> Dense -> Sigmoid -> Output
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        self.dense1 = Dense(input_dim, hidden_dim)
        self.relu1 = ReLU()
        self.dense2 = Dense(hidden_dim, output_dim)
        self.sigmoid1 = Sigmoid()

        self.loss_fn = BinaryCrossEntropy()

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z1 = self.dense1.forward(X)
        A1 = self.relu1.forward(Z1)
        Z2 = self.dense2.forward(A1)
        A2 = self.sigmoid1.forward(Z2)
        return A2

    def backward(self, learning_rate: float) -> None:
        dA2 = self.loss_fn.backward()
        dZ2 = self.sigmoid1.backward(dA2)
        dA1 = self.dense2.backward(dZ2, learning_rate)
        dZ1 = self.relu1.backward(dA1)
        self.dense1.backward(dZ1, learning_rate)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss_fn.forward(y_pred, y_true)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        y_pred = self.forward(X)
        return (y_pred >= threshold).astype(int)