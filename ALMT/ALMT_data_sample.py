from __future__ import annotations
import argparse
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from mosei_dataset import FEATURE_DIMS, MoseiFeatureDataset

from .ALMT_data import (
    SequenceSubset,
    PooledEmbeddingDataset,
    PreparedCSDClipDataset,
    prepared_clip_collate,
    aggregate_window_logits,
    load_embedding_splits,
    sequence_collate,
)
from .ALMT_model import (
    MSALMT,
)
from .ALMT_evaluation import (
    evaluate,
    evaluate_prepared,
    resolve_device,
    sentiment_classes,
    set_seed,
    write_csv,
)

from ..common.protocol import (
    ValidationCheckpoint,
    add_training_arguments,
    add_mean_std_display,
    validate_training_arguments,
)

def print_summary_table(
    title: str,
    columns: tuple[tuple[str, str], ...],
    rows: list[dict[str, Any]],
) -> None:
    headers = [header for header, _ in columns]
    values = [
        [
            f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key])
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

def nested_stratified_order(labels: list[int], seed: int) -> list[int]:
    """Create a deterministic order whose prefixes are nested and class-balanced."""
    rng = np.random.default_rng(seed)
    class_ids = np.unique(np.asarray(labels, dtype=np.int64))
    rng.shuffle(class_ids)
    class_indices: dict[int, list[int]] = {}
    for class_id in class_ids:
        indices = np.flatnonzero(np.asarray(labels) == class_id)
        rng.shuffle(indices)
        class_indices[int(class_id)] = indices.tolist()

    order: list[int] = []
    max_class_size = max(len(indices) for indices in class_indices.values())
    for offset in range(max_class_size):
        for class_id in class_ids:
            indices = class_indices[int(class_id)]
            if offset < len(indices):
                order.append(indices[offset])
    return order

def summarize_sample_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    metric_names = (
        "train_accuracy", "train_macro_f1", "valid_accuracy", "valid_macro_f1",
        "test_accuracy", "test_macro_f1",
    )
    for percentage in sorted({int(row["training_data_percent"]) for row in rows}):
        group = [row for row in rows if int(row["training_data_percent"]) == percentage]
        summary: dict[str, Any] = {
            "training_data_percent": percentage,
            "seeds_completed": len(group),
            "train_samples": group[0]["train_samples"],
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        summaries.append(summary)
    return summaries

def train_sample_run(
    percentage: int,
    train_data: Dataset[dict[str, torch.Tensor]],
    train_indices: list[int],
    valid_data: Dataset[dict[str, torch.Tensor]],
    test_data: Dataset[dict[str, torch.Tensor]],
    feature_dims: dict[str, int],
    label_mapping: dict[str, int],
    input_mode: str,
    args: argparse.Namespace,
    device: torch.device,
    prepared_mode: bool = False,
) -> dict[str, Any]:
    set_seed(args.seed)
    sample_count = len(train_indices)
    train_subset = Subset(train_data, train_indices)
    loader_args = {
        "batch_size": args.batch_size,
        "collate_fn": prepared_clip_collate if prepared_mode else sequence_collate,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_args,
    )
    train_eval_loader = DataLoader(train_subset, shuffle=False, **loader_args)
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)

    model = MSALMT(
        feature_dims=feature_dims,
        hidden_dim=args.hidden_dim,
        token_count=args.token_count,
        max_input_length=args.max_input_length,
        num_heads=args.num_heads,
        ahl_depth=args.ahl_depth,
        fusion_depth=args.fusion_depth,
        dropout=args.dropout,
        output_dim=2 if prepared_mode else 1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    criterion = nn.MSELoss()
    checkpoint_selector = ValidationCheckpoint(args.patience)
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    print(f"\nData sample {percentage:3d}% | train={sample_count} | device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if prepared_mode:
                labels = batch["labels"].to(device)
                window_to_clip = batch["window_to_clip"].to(device)
                durations = batch["durations"].to(device)
                model_batch = {
                    name: value.to(device)
                    for name, value in batch.items()
                    if name in {
                        "text_features", "text_mask", "audio_features", "audio_mask",
                        "image_features", "image_mask",
                    }
                }
                window_logits = model(model_batch)
                clip_logits = aggregate_window_logits(
                    window_logits, window_to_clip, durations, labels.shape[0]
                )
                loss = nn.functional.cross_entropy(clip_logits, labels)
                batch_count = labels.shape[0]
            else:
                scores = batch.pop("scores").to(device)
                device_batch = {name: value.to(device) for name, value in batch.items()}
                loss = criterion(model(device_batch), scores)
                batch_count = scores.shape[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * batch_count
            total_items += batch_count

        if prepared_mode:
            valid_metrics, _, _ = evaluate_prepared(model, valid_loader, device)
        else:
            valid_metrics, _, _ = evaluate(model, valid_loader, device, args.neutral_threshold)
        mean_loss = total_loss / max(1, total_items)
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "valid_loss": valid_metrics["loss"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | Loss {mean_loss:.4f} | "
            f"valid Acc {valid_metrics['accuracy']:.4f} | "
            f"valid Macro-F1 {valid_metrics['macro_f1']:.4f}"
        )
        improved, should_stop = checkpoint_selector.update(
            valid_metrics["macro_f1"], valid_metrics["loss"]
        )
        if improved:
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if should_stop:
            print(f"Early stopping after {epoch} epochs.")
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint produced for {percentage}% data sample.")
    run_name = f"sample_{percentage:03d}pct"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "training_history.csv", history)
    model.load_state_dict(best_state)
    checkpoint_path = run_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "input_mode": input_mode,
            "model_config": {
                "feature_dims": feature_dims,
                "hidden_dim": args.hidden_dim,
                "token_count": args.token_count,
                "max_input_length": args.max_input_length,
                "num_heads": args.num_heads,
                "ahl_depth": args.ahl_depth,
                "fusion_depth": args.fusion_depth,
                "dropout": args.dropout,
                "output_dim": 2 if prepared_mode else 1,
            },
            "label_mapping": label_mapping,
            "neutral_threshold": args.neutral_threshold,
            "sample_percentage": percentage,
            "best_valid_macro_f1": checkpoint_selector.best_macro_f1,
            "best_valid_loss": checkpoint_selector.best_validation_loss,
            "training_config": {
                "max_epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            "seed": args.seed,
        },
        checkpoint_path,
    )

    if prepared_mode:
        train_metrics, _, _ = evaluate_prepared(model, train_eval_loader, device)
        valid_metrics, _, _ = evaluate_prepared(model, valid_loader, device)
        test_metrics, _, _ = evaluate_prepared(model, test_loader, device)
    else:
        train_metrics, _, _ = evaluate(model, train_eval_loader, device, args.neutral_threshold)
        valid_metrics, _, _ = evaluate(model, valid_loader, device, args.neutral_threshold)
        test_metrics, _, _ = evaluate(model, test_loader, device, args.neutral_threshold)
    result: dict[str, Any] = {
        "training_data_percent": percentage,
        "train_samples": sample_count,
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "valid_accuracy": valid_metrics["accuracy"],
        "valid_macro_f1": valid_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "best_valid_macro_f1": checkpoint_selector.best_macro_f1,
        "best_valid_loss": checkpoint_selector.best_validation_loss,
        "epochs_trained": len(history),
        "checkpoint": str(checkpoint_path),
    }
    write_csv(run_dir / "metrics.csv", [result])
    print(
        f"{percentage:3d}% | Test Accuracy {result['test_accuracy']:.4f} | "
        f"Test Macro-F1 {result['test_macro_f1']:.4f}"
    )
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--dataset-dir", type=Path, default=Path("CMU-MOSEI"))
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        help="Read the HANDOFF prepared-csd cache with binary clip labels and duration-weighted window pooling.",
    )
    parser.add_argument(
        "--embedding-file",
        type=Path,
        help="Use precomputed NPZ embeddings instead of reading raw timestamped CSD sequences.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("msa_almt_samples_standard"))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ahl-depth", type=int, default=3)
    parser.add_argument("--fusion-depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neutral-threshold", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    validate_training_arguments(parser, args)
    if args.max_input_length < 1:
        parser.error("max-input-length must be positive")
    if args.num_heads < 1 or args.hidden_dim % args.num_heads:
        parser.error("hidden-dim must be divisible by num-heads")
    if args.ahl_depth != 3:
        parser.error("ALMT uses exactly three AHL layers for its three language guidance scales")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("warmup-ratio must be in [0, 1)")
    if args.neutral_threshold < 0:
        parser.error("neutral-threshold must be non-negative")

    set_seed(args.split_seed)
    if args.prepared_dir is not None and args.embedding_file is not None:
        parser.error("--prepared-dir and --embedding-file are mutually exclusive")
    if args.prepared_dir is not None:
        prepared_dir = args.prepared_dir.resolve()
        train_data = PreparedCSDClipDataset(prepared_dir, "train")
        valid_data = PreparedCSDClipDataset(prepared_dir, "valid")
        test_data = PreparedCSDClipDataset(prepared_dir, "test")
        feature_dims = {
            modality: int(values.shape[-1])
            for modality, values in train_data.features.items()
        }
        label_mapping = {"negative": 0, "positive": 1}
        input_description = "HANDOFF prepared-csd windows; binary labels, neutral excluded"
        prepared_mode = True
    elif args.embedding_file is not None:
        embedding_path = args.embedding_file.resolve()
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Embedding file does not exist: {embedding_path}")
        split_data, feature_dims, label_mapping = load_embedding_splits(embedding_path)
        train_data = split_data["train"]
        valid_data = split_data["valid"]
        test_data = split_data["test"]
        input_description = f"pooled NPZ embeddings ({embedding_path.name}; one token per modality)"
        prepared_mode = False
    else:
        dataset = MoseiFeatureDataset(args.dataset_dir, args.neutral_threshold)
        split_indices = dataset.video_disjoint_splits(
            seed=args.split_seed,
            train_fraction=args.train_fraction,
            valid_fraction=args.valid_fraction,
        )
        train_data = SequenceSubset(dataset, split_indices["train"], args.max_input_length)
        valid_data = SequenceSubset(dataset, split_indices["valid"], args.max_input_length)
        test_data = SequenceSubset(dataset, split_indices["test"], args.max_input_length)
        feature_dims = FEATURE_DIMS
        label_mapping = dataset.label_to_id
        input_description = "timestamped CSD feature sequences"
        prepared_mode = False
    device = resolve_device(args.device)
    if isinstance(train_data, PreparedCSDClipDataset):
        training_labels = train_data.labels.tolist()
    elif isinstance(train_data, PooledEmbeddingDataset):
        training_labels = sentiment_classes(train_data.scores, args.neutral_threshold).tolist()
    elif isinstance(train_data, SequenceSubset):
        training_labels = [
            train_data.dataset.labels[index]
            for index in train_data.indices
        ]
    else:
        raise TypeError(f"Unsupported training dataset type: {type(train_data).__name__}")

    nested_order = nested_stratified_order(training_labels, args.split_seed)
    root_output_dir = args.output_dir
    root_output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    print(
        f"ALMT data-sample experiment | device={device} | "
        f"train pool={len(train_data)} | valid={len(valid_data)} | "
        f"test={len(test_data)} | seeds={args.seeds} | split_seed={args.split_seed} | "
        f"input={input_description}"
    )

    for seed in args.seeds:
        args.seed = seed
        set_seed(seed)
        args.output_dir = root_output_dir / f"seed_{seed}"
        seed_results: list[dict[str, Any]] = []
        for percentage in range(10, 101, 10):
            sample_size = max(1, int(math.ceil(len(nested_order) * percentage / 100)))
            sample_indices = nested_order[:sample_size]
            result = train_sample_run(
                percentage,
                train_data,
                sample_indices,
                valid_data,
                test_data,
                feature_dims,
                label_mapping,
                "prepared_csd_binary_clips" if prepared_mode else "pooled_embeddings" if args.embedding_file is not None else "csd_sequences",
                args,
                device,
                prepared_mode,
            )
            result["seed"] = seed
            seed_results.append(result)
            all_results.append(result)
            write_csv(root_output_dir / f"data_sample_results_seed_{seed}.csv", seed_results)
            write_csv(root_output_dir / "data_sample_results_by_seed.csv", all_results)

    summary = summarize_sample_results(all_results)
    write_csv(root_output_dir / "data_sample_results_summary.csv", summary)
    print_summary_table(
        "Final ALMT data-sample results",
        (
            ("Train %", "training_data_percent"),
            ("Seeds", "seeds_completed"),
            ("Train N", "train_samples"),
            ("Valid Acc Mean±SD", "valid_accuracy_mean_std"),
            ("Valid Macro-F1 Mean±SD", "valid_macro_f1_mean_std"),
            ("Test Acc Mean±SD", "test_accuracy_mean_std"),
            ("Test Macro-F1 Mean±SD", "test_macro_f1_mean_std"),
        ),
        summary,
    )
    print(f"Per-seed results: {root_output_dir / 'data_sample_results_by_seed.csv'}")
    print(f"Mean/std summary: {root_output_dir / 'data_sample_results_summary.csv'}")

if __name__ == "__main__":
    main()
