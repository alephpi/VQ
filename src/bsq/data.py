from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int,
    num_workers: int,
    data_dir: str | Path,
) -> Tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
