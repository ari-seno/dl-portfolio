# Project 5 — CNN Image Classification (CIFAR-10)

A convolutional neural network trained from scratch to classify CIFAR-10 images into 10 categories, built as a baseline CNN before moving to transfer learning in Project 6.

## Overview

This project implements a mini-VGG style CNN in PyTorch, trained on the CIFAR-10 dataset. The goal is to establish a solid, simple baseline (no pretrained weights, no residual connections) to later contrast against transfer learning results in Project 6.

## Dataset

- **CIFAR-10**: 60,000 32x32 RGB images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Official split: 50,000 train images, 10,000 test images
- Train set further split 90/10 into train/val (45,000 / 5,000), seed=42
- Test set held out entirely during training — used only for final evaluation
- Loaded via `torchvision.datasets.CIFAR10`

**Preprocessing:**
- Normalization with standard CIFAR-10 channel-wise mean/std
- Train-only augmentation: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()`
- Val/test use eval-only transform (no augmentation)

## Architecture

`CIFAR10CNN` — mini-VGG style, no residual connections:

- 3 ConvBlocks (32 → 64 → 128 channels), each with 2x (Conv3x3 + BatchNorm2d + ReLU)
- `MaxPool2d(2)` after each block: 32x32 → 16x16 → 8x8 → 4x4
- Classifier head: Flatten → Linear(2048, 256) → ReLU → Dropout(0.5) → Linear(256, 10)
- Output: raw logits (paired with `nn.CrossEntropyLoss`)

## Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Loss           | CrossEntropyLoss |
| Optimizer      | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler      | ReduceLROnPlateau (factor=0.5, patience=5, monitors val_loss) |
| Batch size     | 128 |
| Epochs         | 50 |
| Seed           | 42 |
| Checkpointing  | Best val_accuracy (not val_loss — classification task) |
| Hardware       | CPU only (Intel Core Ultra 7, no discrete GPU) |

Training took ~65 minutes total (~47-96s/epoch) on CPU.

## Results

**Best checkpoint: epoch 48, val_acc = 0.8974**

By epoch 50, train_acc reached 0.9555 vs val_acc 0.8962 — a widening train/val gap after ~epoch 30 indicates mild overfitting, mitigated by checkpointing on best val accuracy rather than using the final epoch.

**Final test set evaluation (10,000 held-out images, never seen during training):**

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **0.8888** |
| Macro avg F1 | 0.89 |
| Weighted avg F1 | 0.89 |

**Per-class performance:**

| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| airplane    | 0.89 | 0.91 | 0.90 |
| automobile  | 0.94 | 0.94 | 0.94 |
| bird        | 0.87 | 0.82 | 0.85 |
| cat         | 0.76 | 0.78 | 0.77 |
| deer        | 0.88 | 0.90 | 0.89 |
| dog         | 0.84 | 0.82 | 0.83 |
| frog        | 0.92 | 0.94 | 0.93 |
| horse       | 0.95 | 0.90 | 0.93 |
| ship        | 0.92 | 0.94 | 0.93 |
| truck       | 0.92 | 0.93 | 0.93 |

**Weakest classes: cat (F1 0.77) and dog (F1 0.83)** — consistent with a well-known CIFAR-10 difficulty pattern, since cats and dogs share overlapping visual features (fur texture, body shape, indoor/outdoor backgrounds) that are hard to separate without much deeper or more specialized architectures. Vehicle classes (automobile, ship, truck) and horse scored highest (F1 0.93-0.94), likely due to more distinctive silhouettes and backgrounds.

## Project Structure

```
05-cnn-image-classification/
├── data/                      # CIFAR-10 (auto-downloaded / local cifar-10-batches-py)
├── src/
│   ├── dataset.py             # CIFAR10 loader, transforms, train/val/test split
│   ├── model.py                # CIFAR10CNN architecture
│   └── train.py                # training loop + checkpointing
├── notebooks/
│   └── exploration.ipynb      # test evaluation, confusion matrix, misclassified examples
├── tests/
│   └── test_model.py          # 7 unit tests (shape, NaN/Inf, gradients, determinism, etc.)
├── outputs/
│   └── best_model.pt          # best checkpoint (epoch 48, val_acc=0.8974)
├── requirements.txt
├── README.md
├── .gitignore
└── .python-version
```

## Setup & Usage

```bash
# Environment
pyenv local dl-portfolio
pip install -r requirements.txt

# Train
python src/train.py

# Test
pytest tests/ -v

# Explore results
jupyter notebook notebooks/exploration.ipynb
```

## Key Takeaways

- A simple mini-VGG CNN (no pretrained weights, no residual connections) reaches **88.9% test accuracy** on CIFAR-10 — a solid baseline result for this architecture class.
- BatchNorm + light augmentation (RandomCrop + HorizontalFlip) + Dropout(0.5) in the classifier head were enough to control overfitting reasonably well, though a train/val gap still emerged after ~epoch 30.
- Checkpointing on best validation accuracy (rather than saving the final epoch) meaningfully improved the final model, since the last few epochs showed the model beginning to overfit.
- Cat/dog confusion was the dominant error pattern — a natural target for improvement via deeper architectures or transfer learning, which is exactly the direction of Project 6.