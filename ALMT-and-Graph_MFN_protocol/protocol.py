"""Shared training and reporting protocol for the project experiments."""

from __future__ import annotations

import argparse
import csv
import statistics
from typing import Any

import numpy as np


DEFAULT_SEEDS = (1111, 1112, 1113)
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 6


class ValidationCheckpoint:
    """Select by validation Macro-F1, breaking ties with lower validation loss."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_macro_f1 = float("-inf")
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0

    def update(self, macro_f1: float, validation_loss: float) -> tuple[bool, bool]:
        improved = macro_f1 > self.best_macro_f1 or (
            macro_f1 == self.best_macro_f1
            and validation_loss < self.best_validation_loss
        )
        if improved:
            self.best_macro_f1 = macro_f1
            self.best_validation_loss = validation_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        should_stop = self.epochs_without_improvement >= self.patience
        return improved, should_stop


def add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_seeds: bool = True,
    include_patience: bool = True,
) -> None:
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    if include_seeds:
        parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    if include_patience:
        parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)


def validate_training_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    include_seeds: bool = True,
    include_patience: bool = True,
) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive and weight-decay non-negative")
    if include_patience and args.patience < 1:
        parser.error("patience must be positive")
    if include_seeds and (
        not args.seeds or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("seeds must be non-empty and contain no duplicates")


def mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric list.")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_seed_results(
    rows: list[dict[str, Any]],
    *,
    run_key: str,
    metric_names: tuple[str, ...],
    paired_metric: str = "test_macro_f1",
    baseline_name: str = "full",
) -> list[dict[str, Any]]:
    """Return mean/std by run and paired per-seed deltas versus the baseline."""
    baseline_by_seed = {
        int(row["seed"]): float(row[paired_metric])
        for row in rows
        if row[run_key] == baseline_name
    }
    summaries: list[dict[str, Any]] = []
    run_names = list(dict.fromkeys(str(row[run_key]) for row in rows))
    for run_name in run_names:
        group = [row for row in rows if str(row[run_key]) == run_name]
        summary: dict[str, Any] = {
            run_key: run_name,
            "seeds_completed": len(group),
        }
        for key in group[0]:
            if key not in {run_key, "seed", "checkpoint"} and not isinstance(group[0][key], float):
                summary[key] = group[0][key]
        for metric in metric_names:
            mean, std = mean_and_std([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        paired = [
            float(row[paired_metric]) - baseline_by_seed[int(row["seed"])]
            for row in group
            if int(row["seed"]) in baseline_by_seed
        ]
        if paired:
            mean, std = mean_and_std(paired)
            summary[f"paired_{paired_metric}_delta_mean"] = mean
            summary[f"paired_{paired_metric}_delta_std"] = std
        summaries.append(summary)
    return summaries


def add_mean_std_display(
    rows: list[dict[str, Any]], metric_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    for row in rows:
        for metric in metric_names:
            mean_key, std_key = f"{metric}_mean", f"{metric}_std"
            if mean_key in row and std_key in row:
                row[f"{metric}_mean_std"] = f"{row[mean_key]:.4f} ± {row[std_key]:.4f}"
    return rows


def _binary_metric_inputs(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same number of values.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only binary values 0 and 1.")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite values.")
    return labels.astype(np.int64, copy=False), probabilities


def binary_confusion_matrix(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, int | list[int]]:
    labels, probabilities = _binary_metric_inputs(labels, probabilities)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [tn, fp, fn, tp],
    }


def binary_roc_curve(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    labels, probabilities = _binary_metric_inputs(labels, probabilities)
    if labels.size == 0 or np.unique(labels).size < 2:
        return []
    thresholds = np.concatenate(
        ([np.nextafter(probabilities.max(), np.inf)], np.unique(probabilities)[::-1])
    )
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int64)
        tp = int(np.sum((labels == 1) & (predictions == 1)))
        fp = int(np.sum((labels == 0) & (predictions == 1)))
        tn = int(np.sum((labels == 0) & (predictions == 0)))
        fn = int(np.sum((labels == 1) & (predictions == 0)))
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        rows.append({"threshold": float(threshold), "fpr": float(fpr), "tpr": float(tpr)})
    rows = sorted(rows, key=lambda item: item["threshold"], reverse=True)
    return rows


def binary_roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    curve = binary_roc_curve(labels, probabilities)
    if not curve:
        return float("nan")
    fpr = np.asarray([point["fpr"] for point in curve], dtype=np.float64)
    tpr = np.asarray([point["tpr"] for point in curve], dtype=np.float64)
    auc = np.trapezoid(tpr, fpr)
    return float(auc)


def write_roc_csv(path: str | Any, rows: list[dict[str, Any]], *, split: str) -> None:
    path = str(path)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "threshold", "fpr", "tpr"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"split": split, **row})


def print_summary_table(
    title: str,
    columns: tuple[tuple[str, str], ...],
    rows: list[dict[str, Any]],
) -> None:
    headers = [header for header, _ in columns]
    values = [
        [
            ("on" if row[key] else "off")
            if isinstance(row[key], bool)
            else str(row[key])
            for _, key in columns
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in values))
        for index, header in enumerate(headers)
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(f"\n{title}")
    print(border)
    print("| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |")
    print(border)
    for row in values:
        print("| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |")
    print(border)
