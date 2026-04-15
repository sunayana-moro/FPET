from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class Snake(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sin(self.alpha * x).pow(2) / (self.alpha + 1e-6)


class StandardCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.classifier(feats)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class DEQBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, steps: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.recurrent = nn.Linear(hidden_dim, hidden_dim)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        with torch.no_grad():
            self.recurrent.weight.mul_(0.08)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_proj(x)
        z = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        history = []
        for _ in range(self.steps):
            z = torch.tanh(self.recurrent(z) + projected + self.bias)
            history.append(z)
        residual = torch.stack(history[-3:], dim=0).std(dim=0).mean()
        return z, residual

    def spectral_proxy(self) -> float:
        with torch.no_grad():
            return float(torch.linalg.matrix_norm(self.recurrent.weight, ord=2).item())


class LLDEQClassifier(nn.Module):
    def __init__(self, ll_size: int, num_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.deq = DEQBlock(input_dim=ll_size * ll_size, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, ll: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.extract_features(ll)
        logits = self.head(feats)
        return logits, feats

    def extract_features(self, ll: torch.Tensor) -> torch.Tensor:
        feats, _ = self.deq(ll.flatten(1))
        return feats


class FullDEQClassifier(nn.Module):
    def __init__(self, image_size: int, num_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.deq = DEQBlock(input_dim=image_size * image_size, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.extract_features(image)
        logits = self.head(feats)
        return logits, feats

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        feats, _ = self.deq(image.flatten(1))
        return feats


class FrequencyRefiner(nn.Module):
    def __init__(self, feature_dim: int, high_size: int, num_classes: int):
        super().__init__()
        input_dim = feature_dim + high_size * high_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            Snake(alpha=0.7),
            nn.Linear(48, 32),
            Snake(alpha=1.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, features: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([features, high.flatten(1)], dim=1)
        return self.net(merged)


@dataclass
class FPETBundle:
    ll_model: LLDEQClassifier
    refiner: FrequencyRefiner
