import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.dataset import load_data
from src.model import FraudMLP


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    loss_fn = nn.BCELoss()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            probs = model(X_batch)
            loss = loss_fn(probs, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            all_probs.append(probs.cpu())
            all_preds.append((probs >= 0.5).float().cpu())
            all_labels.append(y_batch.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def train(model, train_loader, val_loader, device, epochs=50, lr=1e-3, print_every=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            probs = model(X_batch)
            loss = loss_fn(probs, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:3d} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | Prec: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f} | "
                f"ROC-AUC: {val_metrics['roc_auc']:.4f}"
            )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    batch_size = 512
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size, shuffle=False)

    model = FraudMLP(input_dim=X_train.shape[1]).to(device)

    train(model, train_loader, val_loader, device, epochs=50, lr=1e-3, print_every=5)

    print("\nFinal evaluation on test set:")
    test_metrics = evaluate(model, test_loader, device)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    torch.save(model.state_dict(), "outputs/model.pt")
    print("\nModel checkpoint saved to outputs/model.pt")