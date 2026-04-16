from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment import ExperimentConfig, build_report, run_experiment
from fpet.data import CIFAR100WaveletDataset


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR-100 one-superclass-at-a-time experiments.")
    parser.add_argument("--cifar-root", default="data/cifar-100-python")
    parser.add_argument("--output-dir", default="artifacts/cifar100_superclasses")
    parser.add_argument("--baseline-epochs", type=int, default=1)
    parser.add_argument("--ll-epochs", type=int, default=1)
    parser.add_argument("--refiner-epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = CIFAR100WaveletDataset.coarse_names
    summary_rows = []

    for group_name in groups:
        config = ExperimentConfig(
            dataset_name="cifar100",
            cifar_root=args.cifar_root,
            num_classes=100,
            train_limit=None if args.train_limit == 0 else args.train_limit,
            test_limit=None if args.test_limit == 0 else args.test_limit,
            baseline_epochs=args.baseline_epochs,
            ll_epochs=args.ll_epochs,
            refiner_epochs=args.refiner_epochs,
            group_names=(group_name,),
        )
        result = run_experiment(config)
        report_text = build_report(result)
        report_path = output_dir / f"{slugify(group_name)}.txt"
        report_path.write_text(report_text)

        summary_rows.append(
            {
                "group": group_name,
                "baseline_top1": result["baseline"]["top1"],
                "cnn_ll_top1": result["cnn_ll"]["top1"],
                "deq_full_top1": result["deq_full"]["top1"],
                "fpet_top1": result["fpet"]["top1"],
                "baseline_top5": result["baseline"]["top5"],
                "cnn_ll_top5": result["cnn_ll"]["top5"],
                "deq_full_top5": result["deq_full"]["top5"],
                "fpet_top5": result["fpet"]["top5"],
                "report_path": str(report_path),
            }
        )

    summary_lines = ["CIFAR-100 Superclass Sweep", ""]
    for row in summary_rows:
        summary_lines.append(
            (
                f"- {row['group']}: "
                f"baseline={row['baseline_top1']:.4f}, "
                f"cnn_ll={row['cnn_ll_top1']:.4f}, "
                f"deq_full={row['deq_full_top1']:.4f}, "
                f"fpet={row['fpet_top1']:.4f}"
            )
        )
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))

    print(json.dumps({"output_dir": str(output_dir), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
