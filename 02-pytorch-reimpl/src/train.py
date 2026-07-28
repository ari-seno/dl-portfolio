import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from src.model import NeuralNetwork


def load_data(n_samples: int = 500, noise: float = 0.15, seed: int = 42, test_size: float = 0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # convert ke PyTorch tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test


def train(model: NeuralNetwork, X: torch.Tensor, y: torch.Tensor,
          X_val: torch.Tensor = None, y_val: torch.Tensor = None,
          epochs: int = 4000, learning_rate: float = 0.3,
          print_every: int = 500) -> list:
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()          # reset gradient dari epoch sebelumnya
        y_pred = model(X)              # forward pass
        loss = criterion(y_pred, y)    # hitung loss
        loss.backward()                # autograd: hitung semua gradient otomatis
        optimizer.step()               # update semua weight & bias sekaligus

        losses.append(loss.item())

        if epoch % print_every == 0 or epoch == 1:
            train_preds = model.predict(X)
            train_acc = (train_preds == y).float().mean().item()

            if X_val is not None:
                val_preds = model.predict(X_val)
                val_acc = (val_preds == y_val).float().mean().item()
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")

    return losses


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(
        n_samples=500, noise=0.15, seed=42, test_size=0.2
    )

    model = NeuralNetwork(input_dim=2, hidden_dim=32, output_dim=1)

    losses = train(model, X_train, y_train,
                    X_val=X_test, y_val=y_test,
                    epochs=4000, learning_rate=0.3, print_every=500)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_accuracy = (train_preds == y_train).float().mean().item()
    test_accuracy = (test_preds == y_test).float().mean().item()

    print(f"\nFinal Train Accuracy: {train_accuracy:.4f}")
    print(f"Final Test Accuracy:  {test_accuracy:.4f}")