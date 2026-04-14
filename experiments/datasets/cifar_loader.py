 

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def load_cifar10(
    n_clients: int = 50,        # số client (fix code)
    batch_size: int = 64,       # batch size (fix code)
    iid: bool = True,           # chia IID hay non-IID (fix code)
    data_root: str = "./data", # thư mục CIFAR-10 (fix code)
) -> Tuple[List[DataLoader], DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_root, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if iid:
        total = len(train_ds)
        sizes = [total // n_clients] * n_clients
        sizes[-1] += total - sum(sizes)
        subsets = random_split(train_ds, sizes, generator=torch.Generator().manual_seed(42))
    else:
        targets = torch.tensor(train_ds.targets)
        indices = torch.argsort(targets)
        n_shards = 2 * n_clients
        shard_size = len(train_ds) // n_shards
        shards = [indices[i * shard_size: (i + 1) * shard_size] for i in range(n_shards)]
        subsets = [
            Subset(train_ds, torch.cat([shards[2 * c], shards[2 * c + 1]]).tolist())
            for c in range(n_clients)
        ]

    train_loaders = [
        DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0)
        for sub in subsets
    ]
    return train_loaders, test_loader
