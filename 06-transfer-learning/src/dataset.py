import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ImageNet stats (required since we're using a pretrained ResNet18 backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 128


def get_dataloaders(data_dir="data", batch_size=64, val_split=0.1, seed=42, num_workers=2):
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomCrop(IMAGE_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    # val_set inherits train_transform (with augmentation) via random_split — override it
    val_set.dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader