from __future__ import annotations

import argparse
import copy
import json
import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from fpet.data import (
    CIFAR10WaveletDataset,
    CIFAR100WaveletDataset,
    FrequencyPatternDataset,
    filter_dataset_by_names,
    split_cifar_train,
    split_dataset,
)
from fpet.models import FPETBundle, FrequencyRefiner, FullDEQClassifier, LLDEQClassifier, StandardCNN


@dataclass
class ExperimentConfig:
    dataset_name: str = "cifar10"
    cifar_root: str = "data/cifar-10-batches-py"
    seed: int = 7
    dataset_size: int = 900
    train_limit: int = 5000
    test_limit: int = 1000
    image_size: int = 32
    num_classes: int = 10
    batch_size: int = 32
    baseline_epochs: int = 3
    ll_epochs: int = 4
    refiner_epochs: int = 3
    learning_rate: float = 1e-3
    coreset_fraction: float = 0.55
    perturbation_epsilons: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
    class_names: tuple[str, ...] = ()
    group_names: tuple[str, ...] = ()
    max_classes_per_group: int | None = None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = logits.topk(min(k, logits.size(1)), dim=1).indices
    hits = topk.eq(labels.unsqueeze(1)).any(dim=1).float().mean()
    return float(hits.item())


def dataset_meta(dataset) -> tuple[List[str], List[str], List[int]]:
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    class_names = getattr(base, "class_names", [])
    group_names = getattr(base, "group_names", [])
    class_to_group = getattr(base, "class_to_group", [])
    return list(class_names), list(group_names), list(class_to_group)


def summarize_group_accuracy(predictions: torch.Tensor, labels: torch.Tensor, names: List[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not names:
        return metrics
    for index, name in enumerate(names):
        mask = labels == index
        if int(mask.sum().item()) == 0:
            continue
        accuracy = (predictions[mask] == labels[mask]).float().mean().item()
        metrics[name] = float(accuracy)
    return metrics


def compute_eval_summary(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    group_labels: torch.Tensor | None = None,
    group_names: List[str] | None = None,
    class_to_group: List[int] | None = None,
) -> Dict[str, object]:
    predictions = logits.argmax(dim=1)
    summary: Dict[str, object] = {
        "top1": topk_accuracy(logits, labels, k=1),
        "top5": topk_accuracy(logits, labels, k=5),
        "per_class_top1": summarize_group_accuracy(predictions, labels, class_names),
    }
    if group_labels is not None and group_names and class_to_group:
        valid = group_labels >= 0
        predicted_groups = torch.tensor([class_to_group[int(pred)] for pred in predictions.tolist()], dtype=torch.long)
        summary["per_group_top1"] = summarize_group_accuracy(
            predicted_groups[valid],
            group_labels[valid],
            group_names,
        )
    return summary


def train_classifier(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, kind: str):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        if kind == "baseline":
            logits = model(batch["image"].to(device))
        elif kind == "ll_cnn":
            logits = model(batch["ll"].to(device))
        elif kind == "full_deq":
            logits, _ = model(batch["image"].to(device))
        elif kind == "ll":
            logits, _ = model(batch["ll"].to(device))
        else:
            raise ValueError(f"unsupported kind {kind}")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train_refiner(bundle: FPETBundle, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device):
    bundle.ll_model.eval()
    bundle.refiner.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        ll = batch["ll"].to(device)
        high = batch["high"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            features = bundle.ll_model.extract_features(ll)
        logits = bundle.refiner(features, high)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


def evaluate_baseline(model: StandardCNN, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    logits_all = []
    labels_all = []
    groups_all = []
    with torch.no_grad():
        for batch in loader:
            logits_all.append(model(batch["image"].to(device)).cpu())
            labels_all.append(batch["label"])
            groups_all.append(batch["group_label"])
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    group_labels = torch.cat(groups_all)
    class_names, group_names, class_to_group = dataset_meta(loader.dataset)
    return compute_eval_summary(logits, labels, class_names, group_labels, group_names, class_to_group)


def evaluate_cnn_ll(model: StandardCNN, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    logits_all = []
    labels_all = []
    groups_all = []
    with torch.no_grad():
        for batch in loader:
            logits_all.append(model(batch["ll"].to(device)).cpu())
            labels_all.append(batch["label"])
            groups_all.append(batch["group_label"])
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    group_labels = torch.cat(groups_all)
    class_names, group_names, class_to_group = dataset_meta(loader.dataset)
    return compute_eval_summary(logits, labels, class_names, group_labels, group_names, class_to_group)


def evaluate_deq(model: FullDEQClassifier, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    logits_all = []
    labels_all = []
    groups_all = []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["image"].to(device))
            logits_all.append(logits.cpu())
            labels_all.append(batch["label"])
            groups_all.append(batch["group_label"])
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    group_labels = torch.cat(groups_all)
    class_names, group_names, class_to_group = dataset_meta(loader.dataset)
    return compute_eval_summary(logits, labels, class_names, group_labels, group_names, class_to_group)


def evaluate_bundle(bundle: FPETBundle, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    bundle.ll_model.eval()
    bundle.refiner.eval()
    logits_all = []
    labels_all = []
    groups_all = []
    with torch.no_grad():
        for batch in loader:
            ll = batch["ll"].to(device)
            high = batch["high"].to(device)
            features = bundle.ll_model.extract_features(ll)
            logits_all.append(bundle.refiner(features, high).cpu())
            labels_all.append(batch["label"])
            groups_all.append(batch["group_label"])
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    group_labels = torch.cat(groups_all)
    class_names, group_names, class_to_group = dataset_meta(loader.dataset)
    return compute_eval_summary(logits, labels, class_names, group_labels, group_names, class_to_group)


def select_coreset(model: LLDEQClassifier, subset: Subset, fraction: float, device: torch.device) -> List[int]:
    base_dataset = subset.dataset
    per_class: Dict[int, List[int]] = {}
    feature_bank: Dict[int, List[torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for index in subset.indices:
            sample = base_dataset[index]
            label = int(sample["label"].item())
            feat = model.extract_features(sample["ll"].unsqueeze(0).to(device)).cpu().squeeze(0)
            per_class.setdefault(label, []).append(index)
            feature_bank.setdefault(label, []).append(feat)

    selected: List[int] = []
    for label, indices in per_class.items():
        feats = torch.stack(feature_bank[label])
        centroid = feats.mean(dim=0, keepdim=True)
        distances = torch.norm(feats - centroid, dim=1)
        keep = max(1, math.ceil(len(indices) * fraction))
        order = torch.argsort(distances)[:keep].tolist()
        selected.extend(indices[idx] for idx in order)
    return selected


def feature_distortion(
    original_features: torch.Tensor,
    perturbed_features: torch.Tensor,
) -> float:
    numerator = torch.norm(perturbed_features - original_features, dim=1)
    denominator = torch.norm(original_features, dim=1) + 1e-6
    return float((numerator / denominator).mean().item())


def perturb_model(model: nn.Module, epsilon: float) -> nn.Module:
    candidate = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in candidate.parameters():
            noise = torch.randn_like(parameter) * epsilon
            parameter.add_(noise)
    return candidate


def collect_features(model: nn.Module, loader: DataLoader, device: torch.device, kind: str) -> torch.Tensor:
    model.eval()
    features = []
    with torch.no_grad():
        for batch in loader:
            if kind == "baseline":
                feats = model.extract_features(batch["image"].to(device))
            elif kind == "ll_cnn":
                feats = model.extract_features(batch["ll"].to(device))
            elif kind == "full_deq":
                feats = model.extract_features(batch["image"].to(device))
            elif kind == "ll":
                feats = model.extract_features(batch["ll"].to(device))
            else:
                raise ValueError(f"unsupported feature kind {kind}")
            features.append(feats.cpu())
    return torch.cat(features, dim=0)


def peak_memory_mb() -> float:
    _, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


def build_datasets(config: ExperimentConfig):
    if config.dataset_name == "synthetic":
        dataset = FrequencyPatternDataset(
            size=config.dataset_size,
            image_size=config.image_size,
            num_classes=config.num_classes,
            seed=config.seed,
        )
        return split_dataset(dataset)

    if config.dataset_name == "cifar10":
        train_dataset = CIFAR10WaveletDataset(
            root=config.cifar_root,
            train=True,
            limit=config.train_limit,
            seed=config.seed,
        )
        test_dataset = CIFAR10WaveletDataset(
            root=config.cifar_root,
            train=False,
            limit=config.test_limit,
            seed=config.seed,
        )
        if config.class_names:
            train_dataset = filter_dataset_by_names(train_dataset, allowed_class_names=config.class_names)
            test_dataset = filter_dataset_by_names(test_dataset, allowed_class_names=config.class_names)
        train_set, val_set = split_cifar_train(train_dataset)
        return train_set, val_set, test_dataset

    if config.dataset_name == "cifar100":
        train_dataset = CIFAR100WaveletDataset(
            root=config.cifar_root,
            train=True,
            limit=config.train_limit,
            seed=config.seed,
        )
        test_dataset = CIFAR100WaveletDataset(
            root=config.cifar_root,
            train=False,
            limit=config.test_limit,
            seed=config.seed,
        )
        if config.class_names or config.group_names or config.max_classes_per_group is not None:
            train_dataset = filter_dataset_by_names(
                train_dataset,
                allowed_class_names=config.class_names or None,
                allowed_group_names=config.group_names or None,
                max_classes_per_group=config.max_classes_per_group,
            )
            test_dataset = filter_dataset_by_names(
                test_dataset,
                allowed_class_names=config.class_names or None,
                allowed_group_names=config.group_names or None,
                max_classes_per_group=config.max_classes_per_group,
            )
        train_set, val_set = split_cifar_train(train_dataset)
        return train_set, val_set, test_dataset

    raise ValueError(f"unsupported dataset {config.dataset_name}")


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    set_seed(config.seed)
    device = torch.device("cpu")
    tracemalloc.start()

    train_set, val_set, test_set = build_datasets(config)
    class_names, group_names, _ = dataset_meta(train_set)
    config.num_classes = len(class_names) if class_names else config.num_classes
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)

    baseline = StandardCNN(num_classes=config.num_classes).to(device)
    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=config.learning_rate)
    start = time.perf_counter()
    for _ in range(config.baseline_epochs):
        train_classifier(baseline, train_loader, baseline_optimizer, device, kind="baseline")
    baseline_time = time.perf_counter() - start
    baseline_metrics = evaluate_baseline(baseline, test_loader, device)

    ll_cnn = StandardCNN(num_classes=config.num_classes).to(device)
    ll_cnn_optimizer = torch.optim.Adam(ll_cnn.parameters(), lr=config.learning_rate)
    start = time.perf_counter()
    for _ in range(config.baseline_epochs):
        train_classifier(ll_cnn, train_loader, ll_cnn_optimizer, device, kind="ll_cnn")
    ll_cnn_time = time.perf_counter() - start
    ll_cnn_metrics = evaluate_cnn_ll(ll_cnn, test_loader, device)

    full_deq = FullDEQClassifier(image_size=config.image_size, num_classes=config.num_classes).to(device)
    full_deq_optimizer = torch.optim.Adam(full_deq.parameters(), lr=config.learning_rate)
    start = time.perf_counter()
    for _ in range(config.ll_epochs):
        train_classifier(full_deq, train_loader, full_deq_optimizer, device, kind="full_deq")
    full_deq_time = time.perf_counter() - start
    full_deq_metrics = evaluate_deq(full_deq, test_loader, device)

    ll_size = config.image_size // 2
    ll_model = LLDEQClassifier(ll_size=ll_size, num_classes=config.num_classes).to(device)
    ll_optimizer = torch.optim.Adam(ll_model.parameters(), lr=config.learning_rate)
    start = time.perf_counter()
    for _ in range(config.ll_epochs):
        train_classifier(ll_model, train_loader, ll_optimizer, device, kind="ll")
    ll_stage_time = time.perf_counter() - start

    start = time.perf_counter()
    coreset_indices = select_coreset(ll_model, train_set, config.coreset_fraction, device)
    coreset_time = time.perf_counter() - start
    if isinstance(train_set, Subset):
        coreset_dataset = Subset(train_set.dataset, coreset_indices)
    else:
        coreset_dataset = Subset(train_set, coreset_indices)
    coreset_loader = DataLoader(coreset_dataset, batch_size=config.batch_size, shuffle=True)

    refiner = FrequencyRefiner(feature_dim=32, high_size=ll_size, num_classes=config.num_classes).to(device)
    bundle = FPETBundle(ll_model=ll_model, refiner=refiner)
    refiner_optimizer = torch.optim.Adam(refiner.parameters(), lr=config.learning_rate)
    start = time.perf_counter()
    for _ in range(config.refiner_epochs):
        train_refiner(bundle, coreset_loader, refiner_optimizer, device)
    refine_time = time.perf_counter() - start
    fpet_metrics = evaluate_bundle(bundle, test_loader, device)

    perturb_loader = DataLoader(val_set, batch_size=config.batch_size)
    baseline_reference = collect_features(baseline, perturb_loader, device, kind="baseline")
    ll_cnn_reference = collect_features(ll_cnn, perturb_loader, device, kind="ll_cnn")
    full_deq_reference = collect_features(full_deq, perturb_loader, device, kind="full_deq")
    ll_reference = collect_features(ll_model, perturb_loader, device, kind="ll")
    perturbations = []
    for epsilon in config.perturbation_epsilons:
        perturbed_baseline = perturb_model(baseline, epsilon)
        perturbed_ll_cnn = perturb_model(ll_cnn, epsilon)
        perturbed_full_deq = perturb_model(full_deq, epsilon)
        perturbed_ll = perturb_model(ll_model, epsilon)
        baseline_distortion = feature_distortion(
            baseline_reference, collect_features(perturbed_baseline, perturb_loader, device, kind="baseline")
        )
        ll_cnn_distortion = feature_distortion(
            ll_cnn_reference, collect_features(perturbed_ll_cnn, perturb_loader, device, kind="ll_cnn")
        )
        full_deq_distortion = feature_distortion(
            full_deq_reference, collect_features(perturbed_full_deq, perturb_loader, device, kind="full_deq")
        )
        ll_distortion = feature_distortion(
            ll_reference, collect_features(perturbed_ll, perturb_loader, device, kind="ll")
        )
        perturbations.append({"model": "baseline_cnn", "epsilon": epsilon, "distortion": baseline_distortion})
        perturbations.append({"model": "ll_cnn", "epsilon": epsilon, "distortion": ll_cnn_distortion})
        perturbations.append({"model": "full_deq", "epsilon": epsilon, "distortion": full_deq_distortion})
        perturbations.append({"model": "ll_deq", "epsilon": epsilon, "distortion": ll_distortion})

    result = {
        "dataset": {
            "dataset_name": config.dataset_name,
            "num_classes": config.num_classes,
            "selected_classes": class_names,
            "selected_groups": group_names,
        },
        "baseline": {**baseline_metrics, "train_time_sec": baseline_time, "param_count": count_parameters(baseline)},
        "cnn_ll": {**ll_cnn_metrics, "train_time_sec": ll_cnn_time, "param_count": count_parameters(ll_cnn)},
        "deq_full": {
            **full_deq_metrics,
            "train_time_sec": full_deq_time,
            "param_count": count_parameters(full_deq),
            "spectral_proxy": full_deq.deq.spectral_proxy(),
        },
        "fpet": {
            **fpet_metrics,
            "ll_stage_time_sec": ll_stage_time,
            "coreset_time_sec": coreset_time,
            "refine_time_sec": refine_time,
            "end_to_end_time_sec": ll_stage_time + coreset_time + refine_time,
            "param_count": count_parameters(ll_model) + count_parameters(refiner),
            "coreset_fraction_realized": len(coreset_indices) / len(train_set),
            "spectral_proxy": ll_model.deq.spectral_proxy(),
        },
        "memory_peak_mb": peak_memory_mb(),
        "perturbations": perturbations,
    }
    tracemalloc.stop()
    return result


def build_report(result: Dict[str, object]) -> str:
    lines = ["Dataset:"]
    for metric_name, metric_value in result["dataset"].items():
        lines.append(f"- {metric_name}: {metric_value}")
    lines.append("")
    lines.append("Stage Metrics:")
    for stage_name in ("baseline", "cnn_ll", "deq_full", "fpet"):
        lines.append(f"- {stage_name}")
        for metric_name, metric_value in result[stage_name].items():
            if isinstance(metric_value, dict):
                continue
            lines.append(f"  - {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"  - {metric_name}: {metric_value}")
    lines.append(f"- runtime\n  - memory_peak_mb: {result['memory_peak_mb']:.4f}")
    lines.append("")
    lines.append("Per-Class Top-1:")
    for stage_name in ("baseline", "cnn_ll", "deq_full", "fpet"):
        lines.append(f"- {stage_name}")
        for class_name, score in result[stage_name].get("per_class_top1", {}).items():
            lines.append(f"  - {class_name}: {score:.4f}")
    any_groups = any(result[stage_name].get("per_group_top1") for stage_name in ("baseline", "cnn_ll", "deq_full", "fpet"))
    if any_groups:
        lines.append("")
        lines.append("Per-Group Top-1:")
        for stage_name in ("baseline", "cnn_ll", "deq_full", "fpet"):
            group_metrics = result[stage_name].get("per_group_top1", {})
            if not group_metrics:
                continue
            lines.append(f"- {stage_name}")
            for group_name, score in group_metrics.items():
                lines.append(f"  - {group_name}: {score:.4f}")
        lines.append("")
    lines.append("Perturbation Distortion:")
    for item in result["perturbations"]:
        lines.append(f"- {item['model']} epsilon={item['epsilon']:.3f}: {item['distortion']:.4f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local FPET prototype experiment.")
    parser.add_argument("--dataset", default="cifar10", choices=("cifar10", "cifar100", "synthetic"))
    parser.add_argument("--cifar-root", default="data/cifar-10-batches-py")
    parser.add_argument("--class-names", default="")
    parser.add_argument("--group-names", default="")
    parser.add_argument("--max-classes-per-group", type=int)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--baseline-epochs", type=int, default=3)
    parser.add_argument("--ll-epochs", type=int, default=4)
    parser.add_argument("--refiner-epochs", type=int, default=3)
    parser.add_argument("--report-path", default="artifacts/latest_report.txt")
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset_name=args.dataset,
        cifar_root=args.cifar_root,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        baseline_epochs=args.baseline_epochs,
        ll_epochs=args.ll_epochs,
        refiner_epochs=args.refiner_epochs,
        class_names=tuple(name.strip() for name in args.class_names.split(",") if name.strip()),
        group_names=tuple(name.strip() for name in args.group_names.split(",") if name.strip()),
        max_classes_per_group=args.max_classes_per_group,
    )
    if config.dataset_name == "cifar100":
        config.num_classes = 100
    result = run_experiment(config)

    report = build_report(result)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report)
    print("")
    print(json.dumps({"report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
