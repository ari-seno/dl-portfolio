import numpy as np
from sklearn.datasets import make_moons
from src.model import NeuralNetwork


def load_data(n_samples: int = 500, noise: float = 0.2, seed: int = 42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = y.reshape(-1, 1)  # bentuk (m, 1) supaya cocok dengan output Sigmoid
    return X, y


def train(model: NeuralNetwork, X: np.ndarray, y: np.ndarray,
          epochs: int = 1000, learning_rate: float = 0.1,
          print_every: int = 100) -> list:
    losses = []

    for epoch in range(1, epochs + 1):
        y_pred = model.forward(X)
        loss = model.compute_loss(y_pred, y)
        model.backward(learning_rate)

        losses.append(loss)

        if epoch % print_every == 0 or epoch == 1:
            preds = model.predict(X)
            accuracy = np.mean(preds == y)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

    return losses


if __name__ == "__main__":
    X, y = load_data(n_samples=500, noise=0.2, seed=42)

    model = NeuralNetwork(input_dim=2, hidden_dim=16, output_dim=1)

    losses = train(model, X, y, epochs=1000, learning_rate=0.1, print_every=100)

    final_preds = model.predict(X)
    final_accuracy = np.mean(final_preds == y)
    print(f"\nFinal training accuracy: {final_accuracy:.4f}")