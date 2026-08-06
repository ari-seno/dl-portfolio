"""
CNN architecture for CIFAR-10 image classification.

A mini-VGG style network: stacked Conv-BatchNorm-ReLU blocks with
MaxPooling for downsampling, followed by a small fully-connected head.
Kept intentionally simple (no residual connections) since this project's
goal is to establish a solid baseline CNN before moving to transfer
learning in Project 6.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU, optionally repeated twice before pooling."""

    def __init__(self, in_channels: int, out_channels: int, double_conv: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if double_conv:
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CIFAR10CNN(nn.Module):
    """
    Mini-VGG style CNN for CIFAR-10 (10-class, 32x32x3 input).

    Feature extractor: 3 ConvBlocks (32 -> 64 -> 128 channels), each
    followed by MaxPool2d(2), taking spatial size 32 -> 16 -> 8 -> 4.
    Classifier head: flatten -> Linear -> ReLU -> Dropout -> Linear (logits).
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2),      # 32x32 -> 16x16

            ConvBlock(32, 64),
            nn.MaxPool2d(2),      # 16x16 -> 8x8

            ConvBlock(64, 128),
            nn.MaxPool2d(2),      # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logits, use nn.CrossEntropyLoss (applies softmax internally)


if __name__ == "__main__":
    model = CIFAR10CNN()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    dummy_input = torch.randn(8, 3, 32, 32)  # batch of 8 CIFAR-10 images
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # expected: (8, 10)