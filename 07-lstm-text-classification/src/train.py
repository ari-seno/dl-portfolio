"""
Training script for the SmSA BiLSTM sentiment classifier.

Because the classes are imbalanced (neutral is ~10-13% of the data vs
~30-58% for negative/positive), plain accuracy can be misleading. We
track macro F1 (unweighted average across classes) as the primary
metric for model selection, alongside accuracy for reference.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader

from src.dataset import SmSADataset, collate_fn, IDX2LABEL
from src.model import LSTMClassifier
from src.vocab import Vocab

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, lengths, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            logits = model(texts, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * texts.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, all_preds, all_labels


def get_device():
    """Select the best available device: XPU (Intel Arc) > CUDA > CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    vocab = Vocab.load(str(ROOT / "outputs" / "vocab.json"))
    print(f"Vocab size: {len(vocab)}")

    train_ds = SmSADataset(str(ROOT / "data" / "train_preprocess.tsv"), vocab)
    valid_ds = SmSADataset(str(ROOT / "data" / "valid_preprocess.tsv"), vocab)
    test_ds = SmSADataset(str(ROOT / "data" / "test_preprocess.tsv"), vocab)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = 0.0
    best_model_path = ROOT / "outputs" / "best_model.pt"
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for texts, lengths, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(texts, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            # gradient clipping - standard practice for RNNs to avoid
            # exploding gradients on longer sequences
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * texts.size(0)

        train_loss /= len(train_ds)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, valid_loader, criterion, device)

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_macro_f1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> new best model saved (val_macro_f1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest results -> loss: {test_loss:.4f} | acc: {test_acc:.4f} | macro_f1: {test_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=[IDX2LABEL[i] for i in range(3)],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM SmSA sentiment classifier")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=4, help="early stopping patience")
    args = parser.parse_args()

    train(args)