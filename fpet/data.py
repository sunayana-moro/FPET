from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pywt
import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class WaveletSample:
    image: torch.Tensor
    ll: torch.Tensor
    high: torch.Tensor
    label: int


class FrequencyPatternDataset(Dataset[WaveletSample]):
    """Synthetic images with both coarse and fine-grained frequency structure."""

    def __init__(self, size: int, image_size: int = 32, num_classes: int = 6, seed: int = 0):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.seed = seed
        self.samples = self._build_samples()

    def _build_samples(self) -> List[WaveletSample]:
        rng = np.random.default_rng(self.seed)
        samples: List[WaveletSample] = []
        for index in range(self.size):
            label = index % self.num_classes
            image = self._make_image(label=label, rng=rng)
            ll, high = self._wavelet_features(image)
            samples.append(
                WaveletSample(
                    image=torch.tensor(image, dtype=torch.float32).unsqueeze(0),
                    ll=torch.tensor(ll, dtype=torch.float32).unsqueeze(0),
                    high=torch.tensor(high, dtype=torch.float32).unsqueeze(0),
                    label=label,
                )
            )
        return samples

    def _make_image(self, label: int, rng: np.random.Generator) -> np.ndarray:
        size = self.image_size
        xs = np.linspace(-1.0, 1.0, size)
        ys = np.linspace(-1.0, 1.0, size)
        grid_x, grid_y = np.meshgrid(xs, ys)

        low_freq = np.exp(-((grid_x * 1.4) ** 2 + (grid_y * 1.4) ** 2))
        low_freq *= 0.25 + 0.1 * label

        phase = rng.uniform(0.0, np.pi)
        patterns = {
            0: np.sin(np.pi * (grid_x * 2.5 + phase)),
            1: np.sin(np.pi * (grid_y * 2.5 + phase)),
            2: np.sin(np.pi * (grid_x + grid_y) * 2.0 + phase),
            3: np.sign(np.sin(np.pi * grid_x * 5.0 + phase) * np.sin(np.pi * grid_y * 5.0 + phase)),
            4: np.cos(np.pi * np.sqrt(grid_x**2 + grid_y**2) * 7.0 + phase),
            5: np.sin(np.pi * grid_x * 4.0 + phase) + np.cos(np.pi * grid_y * 4.5 + phase),
        }
        image = low_freq + 0.35 * patterns[label % len(patterns)]
        image += 0.08 * rng.normal(size=(size, size))
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        return image.astype(np.float32)

    def _wavelet_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ll, (lh, hl, hh) = pywt.dwt2(image, "haar")
        high = np.stack([lh, hl, hh], axis=0).mean(axis=0)
        return ll.astype(np.float32), high.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "image": sample.image,
            "ll": sample.ll,
            "high": sample.high,
            "label": torch.tensor(sample.label, dtype=torch.long),
        }


def split_dataset(dataset: FrequencyPatternDataset, train_fraction: float = 0.7, val_fraction: float = 0.15):
    indices = np.arange(len(dataset))
    train_end = int(len(indices) * train_fraction)
    val_end = int(len(indices) * (train_fraction + val_fraction))
    return (
        Subset(dataset, indices[:train_end]),
        Subset(dataset, indices[train_end:val_end]),
        Subset(dataset, indices[val_end:]),
    )


class CIFAR10WaveletDataset(Dataset[WaveletSample]):
    """Local CIFAR-10 loader that does not depend on torchvision."""

    train_batches = (
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    )

    def __init__(self, root: str | Path, train: bool, limit: int | None = None, seed: int = 0):
        self.root = Path(root)
        self.train = train
        self.limit = limit
        self.seed = seed
        self.samples = self._build_samples()

    def _load_batch(self, batch_name: str) -> Tuple[np.ndarray, np.ndarray]:
        batch_path = self.root / batch_name
        with batch_path.open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        images = payload[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        labels = np.array(payload[b"labels"], dtype=np.int64)
        return images, labels

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        r, g, b = image
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.float32)

    def _wavelet_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ll, (lh, hl, hh) = pywt.dwt2(image, "haar")
        high = np.stack([lh, hl, hh], axis=0).mean(axis=0)
        return ll.astype(np.float32), high.astype(np.float32)

    def _build_samples(self) -> List[WaveletSample]:
        batch_names = self.train_batches if self.train else ("test_batch",)
        image_parts: List[np.ndarray] = []
        label_parts: List[np.ndarray] = []
        for batch_name in batch_names:
            images, labels = self._load_batch(batch_name)
            image_parts.append(images)
            label_parts.append(labels)
        images = np.concatenate(image_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)

        if self.limit is not None and self.limit < len(images):
            rng = np.random.default_rng(self.seed + (0 if self.train else 1))
            chosen = np.sort(rng.choice(len(images), size=self.limit, replace=False))
            images = images[chosen]
            labels = labels[chosen]

        samples: List[WaveletSample] = []
        for image, label in zip(images, labels, strict=False):
            gray = self._to_gray(image)
            ll, high = self._wavelet_features(gray)
            samples.append(
                WaveletSample(
                    image=torch.tensor(gray, dtype=torch.float32).unsqueeze(0),
                    ll=torch.tensor(ll, dtype=torch.float32).unsqueeze(0),
                    high=torch.tensor(high, dtype=torch.float32).unsqueeze(0),
                    label=int(label),
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "image": sample.image,
            "ll": sample.ll,
            "high": sample.high,
            "label": torch.tensor(sample.label, dtype=torch.long),
        }


def split_cifar_train(dataset: CIFAR10WaveletDataset, val_fraction: float = 0.1):
    indices = np.arange(len(dataset))
    val_size = int(len(indices) * val_fraction)
    return (
        Subset(dataset, indices[val_size:]),
        Subset(dataset, indices[:val_size]),
    )
