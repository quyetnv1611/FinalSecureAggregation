"""
experiments/models/spam_model.py
=================================
Improved neural network for SMS Spam classification.

Offers three architectures to find the best accuracy:
1. ImprovedMLP       — Enhanced MLP with better capacity & regularization
2. DeepMLP           — Deeper network with residual-style connections
3. TextCNN           — CNN-style for text feature extraction (1D conv on TF-IDF)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedMLP(nn.Module):
    """Enhanced MLP with better capacity and dropout regularization.
    
    Architecture:
        Dense(5000 → 512, ReLU) → Dropout(0.3)
        Dense(512 → 256, ReLU) → Dropout(0.3)
        Dense(256 → 128, ReLU) → Dropout(0.2)
        Dense(128 → 2)
    """

    def __init__(self, input_dim: int = 5000, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLP(nn.Module):
    """Deeper MLP with batch normalization and skip-like structure.
    
    Architecture:
        Dense(5000 → 1024, ReLU) → BN → Dropout(0.4)
        Dense(1024 → 512, ReLU) → BN → Dropout(0.4)
        Dense(512 → 256, ReLU) → BN → Dropout(0.3)
        Dense(256 → 128, ReLU) → Dropout(0.2)
        Dense(128 → 2)
    """

    def __init__(self, input_dim: int = 5000, dropout: float = 0.4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(256, 128)
        self.drop4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.bn1(F.relu(self.fc1(x))))
        x = self.drop2(self.bn2(F.relu(self.fc2(x))))
        x = self.drop3(self.bn3(F.relu(self.fc3(x))))
        x = self.drop4(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x


class TextCNN(nn.Module):
    """1D CNN adapted for TF-IDF text features.
    
    Architecture:
        Conv1d with multiple filter sizes (3, 5, 7)
        Max pooling over feature dimension
        Dense layers for classification
    """

    def __init__(self, input_dim: int = 5000, num_filters: int = 64) -> None:
        super().__init__()
        # Reshape (batch, 5000) → (batch, 1, 5000) for Conv1d
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(1, num_filters, kernel_size=7, padding=3)
        
        self.fc1 = nn.Linear(num_filters * 3, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 5000)
        x = x.unsqueeze(1)  # → (batch, 1, 5000)
        
        # Apply different-sized convolutions
        c1 = F.relu(self.conv1(x))
        c1 = F.adaptive_max_pool1d(c1, 1).squeeze(-1)  # → (batch, num_filters)
        
        c2 = F.relu(self.conv2(x))
        c2 = F.adaptive_max_pool1d(c2, 1).squeeze(-1)  # → (batch, num_filters)
        
        c3 = F.relu(self.conv3(x))
        c3 = F.adaptive_max_pool1d(c3, 1).squeeze(-1)  # → (batch, num_filters)
        
        # Concatenate pooled features
        x = torch.cat([c1, c2, c3], dim=1)  # → (batch, num_filters*3)
        
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
