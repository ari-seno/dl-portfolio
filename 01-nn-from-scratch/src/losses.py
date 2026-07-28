import numpy as np


class BinaryCrossEntropy:
    """
    Binary Cross-Entropy Loss.
    L = -(1/m) * sum( y*log(y_hat) + (1-y)*log(1-y_hat) )
    """

    def __init__(self, epsilon: float = 1e-8):
        # epsilon mencegah log(0) yang menghasilkan -inf/NaN
        self.epsilon = epsilon
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        self.y_true = y_true

        m = y_true.shape[0]
        loss = -np.sum(
            y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred)
        ) / m
        return loss

    def backward(self) -> np.ndarray:
        # turunan BCE terhadap y_pred: -(y/y_hat - (1-y)/(1-y_hat)) / m
        m = self.y_true.shape[0]
        dA = (
            -(self.y_true / self.y_pred) + (1 - self.y_true) / (1 - self.y_pred)
        ) / m
        return dA