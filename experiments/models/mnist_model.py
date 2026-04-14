"""
experiments/models/mnist_model.py
==================================
Small CNN for MNIST (28×28 greyscale, 10 classes).

Architecture
------------
    Conv2d(1→32, 3×3, ReLU)
    Conv2d(32→64, 3×3, ReLU)
    MaxPool2d(2) + Dropout(0.25)
    Flatten → Dense(9216→128, ReLU) + Dropout(0.5)
    Dense(128→10)

Matches the classic Keras MNIST example so accuracy figures are comparable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """Convolutional network for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.drop1   = nn.Dropout(0.25)
        self.fc1     = nn.Linear(9216, 128)
        self.drop2   = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))          # → (B, 32, 26, 26)
        x = F.relu(self.conv2(x))          # → (B, 64, 24, 24)
        x = F.max_pool2d(x, 2)             # → (B, 64, 12, 12)
        x = self.drop1(x)
        x = x.flatten(1)                   # → (B, 9216)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)                 # → (B, 10)
