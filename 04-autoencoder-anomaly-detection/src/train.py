import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, precision_recall_curve, recall_score, f1_score, roc_auc_score

from src.dataset import load_data
from src.model import Autoencoder


def reconstruction_error(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        X_reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - X_reconstructed) ** 2, dim=1)
    return errors.cpu().numpy()


def train(model, X_train, X_val, epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-5, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch_X,) in train_loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_reconstructed = model(X_val_tensor)
            val_loss = criterion(val_reconstructed, X_val_tensor).item()

        scheduler.step(epoch_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | Train MSE: {epoch_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {current_lr:.6f}")

    print(f"\nBest Val MSE: {best_val_loss:.6f} — loading best model weights")
    model.load_state_dict(best_state_dict)

    return model


def find_best_threshold(errors_val, y_val):
    precisions, recalls, thresholds = precision_recall_curve(y_val, errors_val)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate(errors, y_true, threshold):
    y_pred = (errors > threshold).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, errors)
    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X_train, X_val, y_val, X_test, y_test = load_data()

    # ==== Eksperimen: ganti nilai di sini ====
    LATENT_DIM = 6
    DROPOUT = 0.0    # matikan dropout  
    EPOCHS = 150
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    # ==========================================

    model = Autoencoder(input_dim=30, latent_dim=LATENT_DIM, dropout=DROPOUT)
    model = train(model, X_train, X_val, epochs=EPOCHS, batch_size=256, lr=LR, weight_decay=WEIGHT_DECAY, device=device)

    errors_val = reconstruction_error(model, X_val, device)
    threshold, best_val_f1 = find_best_threshold(errors_val, y_val)
    print(f"\nBest threshold: {threshold:.6f} | Val F1: {best_val_f1:.4f}")

    val_metrics = evaluate(errors_val, y_val, threshold)
    print(f"Val metrics: {val_metrics}")

    errors_test = reconstruction_error(model, X_test, device)
    test_metrics = evaluate(errors_test, y_test, threshold)
    print(f"\nTest metrics: {test_metrics}")

    torch.save(model.state_dict(), "outputs/model.pt")