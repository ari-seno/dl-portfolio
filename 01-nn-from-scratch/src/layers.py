import numpy as np


class Dense:
    """
    Fully connected (Dense) layer.
    Forward:  Z = X @ W + b
    Backward: computes dW, db, and dX using chain rule
    """

    def __init__(self, input_dim: int, output_dim: int):
        # He initialization — cocok untuk layer yang diikuti ReLU
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))

        # cache untuk backward pass
        self.X = None

        # simpan gradient terakhir (opsional, berguna untuk debugging/optimizer lanjutan)
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X  # simpan input untuk dipakai saat backward
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray, learning_rate: float) -> np.ndarray:
        m = self.X.shape[0]  # jumlah sample dalam batch

        self.dW = (self.X.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T

        # update parameter (gradient descent)
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

        return dX