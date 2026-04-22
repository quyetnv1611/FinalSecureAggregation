"""
experiments/datasets/mnist_loader.py
=====================================
MNIST dataset loader with IID and non-IID client partitioning.

Usage
-----
    from experiments.datasets.mnist_loader import load_mnist
    train_loaders, test_loader = load_mnist(n_clients=50, iid=True)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def load_mnist(
    n_clients: int = 50,        # số client (fix code)
    batch_size: int = 64,       # batch size (fix code)
    iid: bool = True,           # chia IID hay non-IID (fix code)
    data_root: str = "./data", # thư mục MNIST (fix code)
) -> Tuple[List[DataLoader], DataLoader]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if iid:
        total = len(train_ds)
        sizes = [total // n_clients] * n_clients
        sizes[-1] += total - sum(sizes)
        subsets = random_split(train_ds, sizes, generator=torch.Generator().manual_seed(42))
    else:
        indices = torch.argsort(torch.tensor(train_ds.targets))
        n_shards = 2 * n_clients
        shard_size = len(train_ds) // n_shards
        shards = [indices[i * shard_size: (i + 1) * shard_size] for i in range(n_shards)]
        subsets = [
            Subset(train_ds, torch.cat([shards[2 * c], shards[2 * c + 1]]).tolist())
            for c in range(n_clients)
        ]

    train_loaders = [
        DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        for sub in subsets
    ]
    return train_loaders, test_loader
