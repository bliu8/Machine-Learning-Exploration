"""Data loading pipeline for CIFAR-10.

Includes training data augmentation (RandomCrop, RandomHorizontalFlip) and
per-channel normalization using CIFAR-10 statistics.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(batch_size: int, data_dir: str = './data', num_workers: int = 2):
    """Load and return CIFAR-10 train and test DataLoaders.

    Augmentations used for training:
    - Random crop of size 32x32 with padding=4.
    - Random horizontal flip.
    - Standard normalization using CIFAR-10 dataset stats:
      mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010).

    Args:
        batch_size (int): Batch size for loaders.
        data_dir (str): Directory where dataset will be stored/downloaded.
        num_workers (int): Number of sub-processes for data loading.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Normalize with CIFAR-10 per-channel statistics
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
