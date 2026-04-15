from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    group_label: int = -1


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
            "group_label": torch.tensor(sample.group_label, dtype=torch.long),
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
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    group_names: List[str] = []
    class_to_group: List[int] = []

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
                    group_label=-1,
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
            "group_label": torch.tensor(sample.group_label, dtype=torch.long),
        }


def split_cifar_train(dataset: CIFAR10WaveletDataset, val_fraction: float = 0.1):
    indices = np.arange(len(dataset))
    val_size = int(len(indices) * val_fraction)
    return (
        Subset(dataset, indices[val_size:]),
        Subset(dataset, indices[:val_size]),
    )


class CIFAR100WaveletDataset(Dataset[WaveletSample]):
    """Local CIFAR-100 loader with fine labels and coarse superclass labels."""

    coarse_names = [
        "aquatic_mammals",
        "fish",
        "flowers",
        "food_containers",
        "fruit_and_vegetables",
        "household_electrical_devices",
        "household_furniture",
        "insects",
        "large_carnivores",
        "large_man-made_outdoor_things",
        "large_natural_outdoor_scenes",
        "large_omnivores_and_herbivores",
        "medium_mammals",
        "non-insect_invertebrates",
        "people",
        "reptiles",
        "small_mammals",
        "trees",
        "vehicles_1",
        "vehicles_2",
    ]

    def __init__(self, root: str | Path, train: bool, limit: int | None = None, seed: int = 0):
        self.root = Path(root)
        self.train = train
        self.limit = limit
        self.seed = seed
        self.class_names = self._load_fine_names()
        self.group_names = self.coarse_names
        self.class_to_group = self._build_class_to_group()
        self.samples = self._build_samples()

    def _load_fine_names(self) -> List[str]:
        meta_path = self.root / "meta"
        with meta_path.open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        return [name.decode("utf-8") for name in payload[b"fine_label_names"]]

    def _build_class_to_group(self) -> List[int]:
        train_path = self.root / "train"
        with train_path.open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        fine = np.array(payload[b"fine_labels"], dtype=np.int64)
        coarse = np.array(payload[b"coarse_labels"], dtype=np.int64)
        mapping = [-1] * len(self.class_names)
        for fine_label, coarse_label in zip(fine, coarse, strict=False):
            mapping[int(fine_label)] = int(coarse_label)
        return mapping

    def _load_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        split_name = "train" if self.train else "test"
        split_path = self.root / split_name
        with split_path.open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        images = payload[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        fine = np.array(payload[b"fine_labels"], dtype=np.int64)
        coarse = np.array(payload[b"coarse_labels"], dtype=np.int64)
        return images, fine, coarse

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        r, g, b = image
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.float32)

    def _wavelet_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ll, (lh, hl, hh) = pywt.dwt2(image, "haar")
        high = np.stack([lh, hl, hh], axis=0).mean(axis=0)
        return ll.astype(np.float32), high.astype(np.float32)

    def _build_samples(self) -> List[WaveletSample]:
        images, fine_labels, coarse_labels = self._load_split()

        if self.limit is not None and self.limit < len(images):
            rng = np.random.default_rng(self.seed + (0 if self.train else 1))
            chosen = np.sort(rng.choice(len(images), size=self.limit, replace=False))
            images = images[chosen]
            fine_labels = fine_labels[chosen]
            coarse_labels = coarse_labels[chosen]

        samples: List[WaveletSample] = []
        for image, fine_label, coarse_label in zip(images, fine_labels, coarse_labels, strict=False):
            gray = self._to_gray(image)
            ll, high = self._wavelet_features(gray)
            samples.append(
                WaveletSample(
                    image=torch.tensor(gray, dtype=torch.float32).unsqueeze(0),
                    ll=torch.tensor(ll, dtype=torch.float32).unsqueeze(0),
                    high=torch.tensor(high, dtype=torch.float32).unsqueeze(0),
                    label=int(fine_label),
                    group_label=int(coarse_label),
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
            "group_label": torch.tensor(sample.group_label, dtype=torch.long),
        }


class FilteredWaveletDataset(Dataset[WaveletSample]):
    """View over a wavelet dataset with remapped labels/groups."""

    def __init__(
        self,
        base_dataset: Dataset[WaveletSample],
        selected_indices: Sequence[int],
        class_names: List[str],
        group_names: List[str],
        class_label_map: Dict[int, int],
        group_label_map: Dict[int, int],
        class_to_group: List[int],
    ):
        self.base_dataset = base_dataset
        self.selected_indices = list(selected_indices)
        self.class_names = class_names
        self.group_names = group_names
        self.class_label_map = class_label_map
        self.group_label_map = group_label_map
        self.class_to_group = class_to_group

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[self.selected_indices[index]]
        label = int(sample["label"].item())
        group_label = int(sample["group_label"].item())
        remapped_group = self.group_label_map.get(group_label, -1)
        return {
            "image": sample["image"],
            "ll": sample["ll"],
            "high": sample["high"],
            "label": torch.tensor(self.class_label_map[label], dtype=torch.long),
            "group_label": torch.tensor(remapped_group, dtype=torch.long),
        }


def filter_dataset_by_names(
    dataset: Dataset[WaveletSample],
    allowed_class_names: Sequence[str] | None = None,
    allowed_group_names: Sequence[str] | None = None,
    max_classes_per_group: int | None = None,
):
    class_names = list(getattr(dataset, "class_names", []))
    group_names = list(getattr(dataset, "group_names", []))
    base_class_to_group = list(getattr(dataset, "class_to_group", []))

    selected_group_ids = set(range(len(group_names))) if group_names else set()
    if allowed_group_names:
        selected_group_ids = {group_names.index(name) for name in allowed_group_names}

    selected_class_ids = set(range(len(class_names))) if class_names else set()
    if allowed_class_names:
        selected_class_ids = {class_names.index(name) for name in allowed_class_names}

    if group_names and selected_group_ids:
        selected_class_ids = {
            class_id
            for class_id in selected_class_ids
            if base_class_to_group[class_id] in selected_group_ids
        }

    if max_classes_per_group is not None and group_names:
        grouped: Dict[int, List[int]] = {}
        for class_id in sorted(selected_class_ids):
            group_id = base_class_to_group[class_id]
            grouped.setdefault(group_id, []).append(class_id)
        limited_ids: set[int] = set()
        for _, class_ids in grouped.items():
            limited_ids.update(class_ids[:max_classes_per_group])
        selected_class_ids = limited_ids

    selected_indices: List[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        label = int(sample["label"].item())
        if label in selected_class_ids:
            selected_indices.append(index)

    new_class_ids = sorted(selected_class_ids)
    class_label_map = {old_id: new_id for new_id, old_id in enumerate(new_class_ids)}
    new_class_names = [class_names[class_id] for class_id in new_class_ids]

    if group_names:
        used_group_ids = sorted({base_class_to_group[class_id] for class_id in new_class_ids})
        group_label_map = {old_id: new_id for new_id, old_id in enumerate(used_group_ids)}
        new_group_names = [group_names[group_id] for group_id in used_group_ids]
        new_class_to_group = [group_label_map[base_class_to_group[class_id]] for class_id in new_class_ids]
    else:
        group_label_map = {}
        new_group_names = []
        new_class_to_group = []

    return FilteredWaveletDataset(
        base_dataset=dataset,
        selected_indices=selected_indices,
        class_names=new_class_names,
        group_names=new_group_names,
        class_label_map=class_label_map,
        group_label_map=group_label_map,
        class_to_group=new_class_to_group,
    )
