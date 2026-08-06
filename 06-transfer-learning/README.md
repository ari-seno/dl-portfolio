# Project 6: Transfer Learning — CIFAR-10 Image Classification with ResNet18

## Overview

This project applies **transfer learning** to CIFAR-10 image classification, using a
ResNet18 backbone pretrained on ImageNet and fine-tuned in two phases. It is a direct
follow-up to [Project 5](../05-cnn-image-classification), which trained a CNN from
scratch on the same dataset — allowing an apples-to-apples comparison between
training from scratch and transfer learning.

## Dataset

- **CIFAR-10**: 60,000 32×32 RGB images across 10 classes (same as Project 5)
- Loaded via `torchvision.datasets.CIFAR10`
- Official 50k train set split 90/10 into train/val (seed=42); official 10k test set held out entirely during training
- Images resized from native 32×32 to **128×128** to better suit the ResNet18 backbone
- Augmentation (train only): `RandomCrop(128, padding=8)`, `RandomHorizontalFlip`
- Normalization: ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`), required for a pretrained backbone

## Architecture

`CIFAR10ResNet18`: `torchvision.models.resnet18` (ImageNet1K pretrained weights) with
the final FC layer replaced:

ResNet18 backbone (pretrained)
→ Linear(512, 256) → ReLU → Dropout(0.5) → Linear(256, 10)


The classifier head mirrors Project 5's head structure, isolating the backbone
(from-scratch vs. pretrained) as the main variable in the comparison.

## Training Strategy: Two-Phase Fine-Tuning

| Phase | Backbone | Epochs | LR | Purpose |
|---|---|---|---|---|
| 1 | Frozen | 10 | 1e-3 | Train classifier head only, adapting pretrained features to CIFAR-10 |
| 2 | Unfrozen | 15 | 1e-5 | Fine-tune the full network with a small LR to avoid destroying pretrained features |

- Optimizer: Adam, weight_decay=1e-4, rebuilt each phase (only trainable params passed)
- Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=3), fresh instance per phase
- Loss: CrossEntropyLoss
- Batch size: 64
- Checkpointing: best val accuracy across both phases → `outputs/best_model.pt`
- Hardware: **Intel Arc GPU (XPU)**, native PyTorch XPU support (no separate IPEX package — deprecated/EOL as of March 2026)

## Results

### Training

- Phase 1 (frozen): val accuracy plateaued around **77%**
- Phase 2 (fine-tuned): jumped to 88% in the first epoch, climbing steadily to a best **val accuracy of 94.98%** (epoch 14)
- Mild overfitting visible by epoch 15 (train acc 97.6% vs val acc 94.9%); best checkpoint (epoch 14) used for evaluation

### Test Set

**Test Accuracy: 94.53%**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| automobile | 0.9672 | 0.9730 | 0.9701 |
| frog | 0.9663 | 0.9740 | 0.9701 |
| truck | 0.9699 | 0.9660 | 0.9679 |
| ship | 0.9632 | 0.9690 | 0.9661 |
| horse | 0.9657 | 0.9580 | 0.9618 |
| airplane | 0.9546 | 0.9470 | 0.9508 |
| deer | 0.9332 | 0.9500 | 0.9415 |
| bird | 0.9443 | 0.9320 | 0.9381 |
| dog | 0.9021 | 0.9120 | 0.9070 |
| cat | 0.8862 | 0.8720 | 0.8790 |

Macro/weighted avg F1: **0.9453**

`cat` remains the hardest class (as in Project 5), reflecting CIFAR-10's classic
cat/dog confusion pattern — but its F1 improved from 0.77 to 0.879.

### Project 5 vs Project 6

| | Project 5 (from-scratch CNN) | Project 6 (Transfer Learning) | Δ |
|---|---|---|---|
| Test Accuracy | 0.8888 | 0.9453 | **+5.65 pts** |
| Training epochs | 50 | 25 (10+15) | -25 |
| Weakest class (F1) | cat (0.77) | cat (0.879) | +0.109 |

Transfer learning reached a substantially higher accuracy in half the epochs,
demonstrating the value of pretrained feature reuse over training from scratch —
especially for a relatively small, low-resolution dataset like CIFAR-10.

## Project Structure

06-transfer-learning/
├── data/ # CIFAR-10 (gitignored, auto-downloaded)
├── src/
│ ├── init.py
│ ├── dataset.py # DataLoaders, augmentation, ImageNet normalization
│ ├── model.py # CIFAR10ResNet18 with freeze/unfreeze support
│ └── train.py # Two-phase fine-tuning loop, XPU device selection
├── notebooks/
│ └── exploration.ipynb # Test evaluation, confusion matrix, P5 vs P6 comparison
├── tests/
│ └── test_model.py # 8 tests incl. freeze/unfreeze behavior
├── outputs/ # best_model.pt (gitignored)
├── requirements.txt
├── .python-version
└── README.md


## Setup & Usage

```bash
# Create/activate the pyenv virtualenv (dl-portfolio), then:
pip install -r requirements.txt

# Train
python src/train.py

# Evaluate (via notebook)
jupyter notebook notebooks/exploration.ipynb

# Run tests
python -m pytest tests/ -v
```

## Key Takeaways

- **Transfer learning outperformed training from scratch by 5.65 points** test accuracy on CIFAR-10, using half the training epochs.
- **Two-phase fine-tuning** (frozen head first, then full unfreeze at a low LR) was essential — Phase 1 alone only reached ~77% val accuracy; full fine-tuning was where most of the gain came from.
- Resizing CIFAR-10 up to 128×128 and using ImageNet normalization were necessary adaptations to make a pretrained ImageNet backbone effective on a very different native resolution.
- The same weak spot (cat classification) persisted across both projects, suggesting it's a genuinely hard case in CIFAR-10 rather than an architecture-specific weakness — but transfer learning still meaningfully narrowed the gap.