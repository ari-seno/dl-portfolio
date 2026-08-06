import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders
from model import CIFAR10ResNet18

SEED = 42
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-5
BATCH_SIZE = 64
CHECKPOINT_PATH = "outputs/best_model.pt"


def get_device():
    if torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, criterion, device, epochs, lr, phase_name, best_val_acc):
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        print(f"[{phase_name}] Epoch {epoch}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> New best val_acc={val_acc:.4f}, checkpoint saved.")

    return best_val_acc


def main():
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, seed=SEED)

    model = CIFAR10ResNet18(num_classes=10, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    # Phase 1: train classifier head only (backbone frozen)
    best_val_acc = train_phase(
        model, train_loader, val_loader, criterion, device,
        epochs=PHASE1_EPOCHS, lr=PHASE1_LR, phase_name="Phase1-head",
        best_val_acc=best_val_acc,
    )

    # Phase 2: unfreeze backbone, fine-tune everything with a smaller LR
    model.unfreeze_backbone()
    best_val_acc = train_phase(
        model, train_loader, val_loader, criterion, device,
        epochs=PHASE2_EPOCHS, lr=PHASE2_LR, phase_name="Phase2-finetune",
        best_val_acc=best_val_acc,
    )

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()