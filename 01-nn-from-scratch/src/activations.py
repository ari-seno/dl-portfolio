import numpy as np


class ReLU:
    """Rectified Linear Unit: f(x) = max(0, x)"""

    def __init__(self):
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        dZ = dA.copy()
        dZ[self.X <= 0] = 0
        return dZ


class Sigmoid:
    """Sigmoid: f(x) = 1 / (1 + e^-x)"""

    def __init__(self):
        self.out = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.out = 1 / (1 + np.exp(-X))
        return self.out

    def backward(self, dA: np.ndarray) -> np.ndarray:
        # turunan sigmoid: s * (1 - s)
        return dA * self.out * (1 - self.out)


class Softmax:
    """
    Softmax untuk multi-class output.
    Biasanya dipasangkan langsung dengan Cross-Entropy loss,
    sehingga backward-nya sering disederhanakan di losses.py
    (gradient gabungan softmax+cross-entropy = y_pred - y_true).
    """

    def __init__(self):
        self.out = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        # kurangi max per baris untuk stabilitas numerik
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.out

    def backward(self, dA: np.ndarray) -> np.ndarray:
        # Placeholder — dalam praktiknya gradient softmax
        # digabung dengan cross-entropy di losses.py untuk efisiensi.
        # Method ini disediakan untuk kelengkapan API saja.
        return dA