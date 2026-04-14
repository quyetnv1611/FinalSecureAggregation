"""
experiments/datasets/webattack_loader.py
=========================================
NSL-KDD network intrusion detection dataset loader.
Binary classification: normal (0) vs. attack (1).

NSL-KDD is downloaded from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/nsl.html

"""




# Raw GitHub mirrors (KDDTrain+.txt / KDDTest+.txt)
_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
_TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

import os
import numpy as np
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

_CATEGORICAL_COLS = [1, 2, 3]   # protocol, service, flag
_N_RAW_FEATURES   = 41


def _fetch(url: str) -> str:

    import urllib.request
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")

_CATEGORICAL_COLS = [1, 2, 3]
_N_RAW_FEATURES   = 41
def _parse(raw: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse NSL-KDD CSV text into (X float64, y int64) arrays."""
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # type: ignore


    rows, labels = [], []
    for line in raw.strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < _N_RAW_FEATURES + 2:
            continue
        label = parts[_N_RAW_FEATURES]           # "normal" or attack name
        row = parts[:_N_RAW_FEATURES]
        rows.append(row)
        labels.append(0 if label.strip() == "normal" else 1)

    X_raw = np.array(rows)

    # Encode categorical features
    le = LabelEncoder()
    for col in _CATEGORICAL_COLS:
        X_raw[:, col] = le.fit_transform(X_raw[:, col])

    X = X_raw.astype(np.float64)

    # Min-max scale
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float64)
    y = np.array(labels, dtype=np.int64)
    return X, y


def load_webattack(

    n_clients: int = 50,
    batch_size: int = 64,
    iid: bool = True,
    seed: int = 42
) -> Tuple[List[DataLoader], DataLoader, int]:
    """Return per-client train loaders, a global test loader, and feature dim."""


    # Đường dẫn cache dữ liệu
    data_dir = Path(__file__).parent.parent.parent / "data" / "nslkdd"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_cache = data_dir / "KDDTrain+.txt"
    test_cache = data_dir / "KDDTest+.txt"

    if not train_cache.exists():
        print("  Downloading NSL-KDD train set ...", flush=True)
        train_cache.write_text(_fetch(_TRAIN_URL), encoding="utf-8")
    if not test_cache.exists():
        print("  Downloading NSL-KDD test set ...", flush=True)
        test_cache.write_text(_fetch(_TEST_URL), encoding="utf-8")

    X_train, y_train = _parse(train_cache.read_text(encoding="utf-8"))
    X_test,  y_test  = _parse(test_cache.read_text(encoding="utf-8"))

    input_dim = X_train.shape[1]  # 41

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size, shuffle=False,
    )

    full_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    total = len(full_ds)
    sizes = [total // n_clients] * n_clients
    sizes[-1] += total - sum(sizes)
    subsets = random_split(
        full_ds, sizes,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loaders = [
        DataLoader(sub, batch_size=batch_size, shuffle=True)
        for sub in subsets
    ]
    return train_loaders, test_loader, input_dim
