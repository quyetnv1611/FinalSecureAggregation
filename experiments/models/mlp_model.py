"""
experiments/models/mlp_model.py
=================================
Generic two-layer MLP for tabular / text-feature classification.
Used for SMS Spam (TF-IDF features) and Web Attack (NSL-KDD features).

Architecture
------------
    Dense(input_dim → hidden_dim, ReLU) → Dropout(0.3) → Dense(hidden_dim → n_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Feedforward MLP for binary or multi-class classification.

    Parameters
    ----------
    input_dim:   Feature vector size.
    hidden_dim:  Width of the hidden layer (default 256).
    n_classes:   Number of output classes (default 2 for binary).
    dropout:     Dropout probability (default 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
