"""
Dataset loading and preprocessing for CIFAR-10 image classification.

Loads CIFAR-10 via torchvision (uses local data/cifar-10-batches-py if
present, otherwise auto-downloads), applies normalization + augmentation,
and splits the official 50k training set into train/val while keeping the
official 10k test set fully held out for final evaluation.
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms

# Standard precomputed CIFAR-10 channel-wise mean/std
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_transforms(augment: bool = True):
    """Return (train_transform, eval_transform) pipelines."""
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = eval_transform

    return train_transform, eval_transform


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    val_split: float = 0.1,
    augment: bool = True,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Build train/val/test DataLoaders for CIFAR-10.

    - Official train set (50,000 images) is split into train/val by `val_split`.
    - Official test set (10,000 images) stays fully held out.
    - Train split uses augmentation (if enabled); val/test always use the
      eval-only transform (no random crop/flip), even though val is carved
      out of the same underlying 50k images as train.
    """
    train_transform, eval_transform = get_transforms(augment=augment)

    # Two dataset objects over the same underlying files, so we can give
    # val the eval_transform while train keeps the augmented one.
    full_train_aug = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    full_train_eval = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=eval_transform
    )

    n_total = len(full_train_aug)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    train_dataset = Subset(full_train_aug, train_subset.indices)
    val_dataset = Subset(full_train_eval, val_subset.indices)

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(
        f"Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)} | "
        f"Test batches: {len(test_loader)}"
    )

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
    print(f"Label range: {labels.min().item()}-{labels.max().item()}")