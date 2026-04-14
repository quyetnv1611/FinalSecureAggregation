"""
experiments/datasets/spam_loader.py
=====================================
SMS Spam Collection dataset loader.
Uses the HuggingFace ``sms_spam`` dataset with TF-IDF feature extraction.

Usage
-----
    from experiments.datasets.spam_loader import load_spam
    train_loaders, test_loader, input_dim = load_spam(n_clients=50)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_spam(
    n_clients: int = 50,
    batch_size: int = 64,
    n_features: int = 5_000,
    test_ratio: float = 0.2,
    iid: bool = True,
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader, int]:
    """Return per-client train loaders, a test loader, and the feature dimension.

    Downloads the ``sms_spam`` dataset from HuggingFace Hub, vectorises with
    TF-IDF (``n_features`` unigrams), and binary-encodes labels (ham=0, spam=1).

    Parameters
    ----------
    n_clients:  Number of FL clients.
    batch_size: Mini-batch size.
    n_features: TF-IDF vocabulary size.
    test_ratio: Fraction of samples reserved for the global test set.
    iid:        Uniform vs label-stratified client split.
    seed:       Random seed for reproducibility.

    Returns
    -------
    (train_loaders, test_loader, input_dim)
    """
    try:
        from datasets import load_dataset       # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install HuggingFace datasets and scikit-learn:\n"
            "  pip install datasets scikit-learn"
        ) from exc

    ds = load_dataset("sms_spam", split="train")  # 5 574 samples
    texts  = ds["sms"]
    labels = [1 if lbl == 1 else 0 for lbl in ds["label"]]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))
    n_test = int(len(texts) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    vec = TfidfVectorizer(max_features=n_features)
    X_train = vec.fit_transform([texts[i] for i in train_idx]).toarray().astype(np.float64)
    X_test  = vec.transform([texts[i] for i in test_idx]).toarray().astype(np.float64)
    y_train = np.array([labels[i] for i in train_idx], dtype=np.int64)
    y_test  = np.array([labels[i] for i in test_idx],  dtype=np.int64)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test)
    y_test_t  = torch.from_numpy(y_test)


    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))
    n_test = int(len(texts) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    vec = TfidfVectorizer(max_features=n_features)
    X_train = vec.fit_transform([texts[i] for i in train_idx]).toarray().astype(np.float64)
    X_test  = vec.transform([texts[i] for i in test_idx]).toarray().astype(np.float64)
    y_train = np.array([labels[i] for i in train_idx], dtype=np.int64)
    y_test  = np.array([labels[i] for i in test_idx],  dtype=np.int64)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test)
    y_test_t  = torch.from_numpy(y_test)

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size, shuffle=False,
    )

    full_ds = TensorDataset(X_train_t, y_train_t)
    total = len(full_ds)
    sizes = [total // n_clients] * n_clients
    sizes[-1] += total - sum(sizes)
    subsets = random_split(full_ds, sizes, generator=torch.Generator().manual_seed(seed))

    train_loaders = [
        DataLoader(sub, batch_size=batch_size, shuffle=True)
        for sub in subsets
    ]
    return train_loaders, test_loader, n_features
