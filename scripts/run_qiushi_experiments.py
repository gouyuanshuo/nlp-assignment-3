"""Train the modality and training-size experiments on the shared CSD cache.

The prepared arrays, official folds and full-training normalization stay as
released. Validation and test are never subsampled. Finished run directories
are reused, and an interrupted run is resumed.
"""

import argparse
import csv
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mosei.config import load_config  # noqa: E402
from mosei.data import load_data  # noqa: E402
from mosei.evaluate import evaluate_run, read_predictions  # noqa: E402
from mosei.io import read_json, write_json  # noqa: E402
from mosei.report import paired_bootstrap  # noqa: E402
from mosei.train import train_run  # noqa: E402

EXPECTED_FINGERPRINT = "50eba4dec0f14e7c2afaa9c60f5519366165b013ac08788e05353d6d71519682"
SEEDS = (1111, 1112, 1113)
FRACTIONS = tuple(range(10, 101, 10))
MODALITY_CONFIGS = {
    "transformer-text-vision": ROOT / "configs" / "transformer_text_vision.toml",
    "transformer-text-audio": ROOT / "configs" / "transformer_text_audio.toml",
}


def jobs():
    for percent in FRACTIONS:
        for model, path in (
            ("text", ROOT / "configs" / "text.toml"),
            ("transformer", ROOT / "configs" / "transformer.toml"),
        ):
            for seed in SEEDS:
                yield {
                    "name": f"{model}-frac{percent}",
                    "config": path,
                    "seed": seed,
                    "fraction": percent / 100,
                    "robustness": percent == 100,
                }
    for name, path in MODALITY_CONFIGS.items():
        for seed in SEEDS:
            yield {
                "name": name,
                "config": path,
                "seed": seed,
                "fraction": 1.0,
                "robustness": True,
            }


def run_dir(output: Path, job: dict) -> Path:
    return output / f"{job['name']}-seed{job['seed']}"


def ensure_trained(data, output: Path, job: dict, device: str):
    directory = run_dir(output, job)
    config = load_config(job["config"])
    config = replace(
        config,
        name=job["name"],
        training=replace(
            config.training,
            seed=job["seed"],
            train_fraction=job["fraction"],
            device=device,
        ),
    )
    if (directory / "run.json").is_file():
        return read_json(directory / "run.json")
    return train_run(data, config, directory, resume=directory.exists())


def ensure_evaluated(data, output: Path, job: dict, device: str):
    directory = run_dir(output, job)
    metrics_path = directory / "evaluation" / "test" / "metrics.json"
    if metrics_path.is_file():
        return read_json(metrics_path)
    partial = directory / "evaluation" / "test"
    if partial.exists():
        shutil.rmtree(directory / "evaluation")
    return evaluate_run(
        data,
        directory,
        split="test",
        robustness=job["robustness"],
        device=device,
    )


def row_from(job, trained, evaluated, full_train_clips):
    clean = evaluated["metrics"]["clean"]
    validation = trained["best_validation"]
    subset = trained.get("train_subset") or {}
    return {
        "experiment": job["name"],
        "seed": job["seed"],
        "train_fraction": job["fraction"],
        "modalities": " ".join(trained["config"]["model"]["modalities"]),
        "train_clips": subset.get("clip_count", full_train_clips),
        "train_videos": subset.get("video_count", ""),
        "best_epoch": trained["best_epoch"],
        "val_macro_f1": validation["macro_f1"],
        "val_accuracy": validation["accuracy"],
        "val_loss": validation["loss"],
        "test_macro_f1": clean["macro_f1"],
        "test_accuracy": clean["accuracy"],
        "test_loss": clean["loss"],
        "training_seconds": trained["training_seconds"],
        "device": trained["device_name"],
    }


def write_summary(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predictions(output: Path, name: str, seed: int):
    return read_predictions(output / f"{name}-seed{seed}" / "evaluation" / "test" / "clean.csv")


def write_comparisons(output: Path):
    pairs = [
        ("transformer-text-vision", "text-frac100"),
        ("transformer-text-audio", "text-frac100"),
        ("transformer-text-vision", "transformer-frac100"),
        ("transformer-text-audio", "transformer-frac100"),
        ("transformer-frac100", "text-frac100"),
    ]
    comparisons = {}
    for left, right in pairs:
        try:
            comparisons[f"{left} minus {right}"] = paired_bootstrap(
                [predictions(output, left, seed) for seed in SEEDS],
                [predictions(output, right, seed) for seed in SEEDS],
            )
        except (OSError, ValueError) as exc:
            comparisons[f"{left} minus {right}"] = {"unavailable": str(exc)}
    write_json(output / "comparisons.json", comparisons)


def write_curve(output: Path, rows: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for model, color in (("text", "#4c78a8"), ("transformer", "#f58518")):
        xs, means = [], []
        for percent in FRACTIONS:
            scores = [
                float(row["test_macro_f1"])
                for row in rows
                if row["experiment"] == f"{model}-frac{percent}"
            ]
            if not scores:
                continue
            xs.append(percent)
            means.append(sum(scores) / len(scores))
            axis.scatter([percent] * len(scores), scores, color=color, alpha=0.45, s=24)
        if xs:
            axis.plot(xs, means, color=color, marker="o", label=model)
    axis.set_xlabel("Target training clips (%) — whole videos")
    axis.set_ylabel("Test macro-F1")
    axis.set_ylim(0.78, 0.85)
    axis.set_xticks(list(FRACTIONS))
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "learning_curve.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "prepared-csd")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "qiushi")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    data = load_data(args.data)
    counts = {split: len(part) for split, part in data.splits.items()}
    if data.fingerprint != EXPECTED_FINGERPRINT or counts != {
        "train": 12789,
        "valid": 1438,
        "test": 3637,
    }:
        raise SystemExit(f"Unexpected prepared cache: {data.fingerprint} {counts}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    planned = list(jobs())
    for index, job in enumerate(planned, start=1):
        print(
            f"[{index}/{len(planned)}] {job['name']} seed={job['seed']} fraction={job['fraction']}",
            flush=True,
        )
        trained = ensure_trained(data, args.output, job, args.device)
        evaluated = ensure_evaluated(data, args.output, job, args.device)
        rows.append(row_from(job, trained, evaluated, counts["train"]))
        write_summary(args.output / "summary.csv", rows)
        print(
            f"test_macro_f1={rows[-1]['test_macro_f1']:.4f} "
            f"test_accuracy={rows[-1]['test_accuracy']:.4f}",
            flush=True,
        )
    write_comparisons(args.output)
    write_curve(args.output, rows)
    print(f"Wrote {args.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
