"""
Training script for CIFAR-10 CNN classification.

Trains CIFAR10CNN on the augmented train split, evaluates on the clean
val split each epoch, and checkpoints the model with the best val
accuracy to outputs/best_model.pt. Final test-set evaluation is left to
a separate evaluation step (notebooks/tests), not run here.
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import CIFAR10CNN

# ---- Config ----
SEED = 42
BATCH_SIZE = 128
VAL_SPLIT = 0.1
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE, val_split=VAL_SPLIT, augment=True, seed=SEED
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = CIFAR10CNN(num_classes=10, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> New best val_acc={val_acc:.4f}, checkpoint saved to {CHECKPOINT_PATH}")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()