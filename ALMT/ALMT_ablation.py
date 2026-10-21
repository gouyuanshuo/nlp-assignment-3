from __future__ import annotations
import argparse
import math
import statistics
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from mosei_dataset import FEATURE_DIMS, MoseiFeatureDataset

from .ALMT_data import (
    SequenceSubset,
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
    summarize_seed_results,
    validate_training_arguments,
)

ABLATION_CONFIGS = {
    "full": (True, True, True, True),
    "without_modality_embedding": (False, True, True, True),
    "without_ahl": (True, False, True, True),
    "without_cross_modal_fusion": (True, True, False, True),
    "without_graph_fusion": (True, True, True, False),
}

def weighted_f1_from_classes(
    labels: torch.Tensor, predictions: torch.Tensor, class_count: int
) -> float:
    labels = labels.detach().long().cpu()
    predictions = predictions.detach().long().cpu()
    confusion = torch.zeros((class_count, class_count), dtype=torch.float64)
    for actual, predicted in zip(labels, predictions):
        confusion[actual, predicted] += 1
    true_positive = confusion.diag()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    support = confusion.sum(dim=1)
    return float((f1 * support).sum() / support.sum().clamp_min(1))

def evaluate_ablation(
    model: MSALMT,
    loader: DataLoader,
    device: torch.device,
    neutral_threshold: float,
    prepared_mode: bool,
) -> dict[str, float]:
    if prepared_mode:
        metrics, labels, predictions = evaluate_prepared(model, loader, device)
        class_count = 2
    else:
        metrics, scores, predicted_scores = evaluate(
            model, loader, device, neutral_threshold
        )
        labels = sentiment_classes(scores, neutral_threshold)
        predictions = sentiment_classes(predicted_scores, neutral_threshold)
        class_count = 3
    return {
        **metrics,
        "weighted_f1": weighted_f1_from_classes(labels, predictions, class_count),
    }

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
            else f"{row[key]:.4f}"
            if isinstance(row[key], float)
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

def run_ablation(
    name: str,
    switches: tuple[bool, bool, bool, bool],
    train_data: Dataset[dict[str, torch.Tensor]],
    valid_data: Dataset[dict[str, torch.Tensor]],
    test_data: Dataset[dict[str, torch.Tensor]],
    feature_dims: dict[str, int],
    label_mapping: dict[str, int],
    input_mode: str,
    input_description: str,
    args: argparse.Namespace,
    device: torch.device,
    train_sample_count: int,
    valid_sample_count: int,
    test_sample_count: int,
    prepared_mode: bool = False,
) -> dict[str, Any]:
    set_seed(args.seed)
    use_modality_embedding, use_ahl, use_cross_modal_fusion, use_graph_fusion = switches
    loader_args = {
        "batch_size": args.batch_size,
        "collate_fn": prepared_clip_collate if prepared_mode else sequence_collate,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_args,
    )
    train_eval_loader = DataLoader(train_data, shuffle=False, **loader_args)
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
        use_modality_embedding=use_modality_embedding,
        use_ahl=use_ahl,
        use_cross_modal_fusion=use_cross_modal_fusion,
        use_graph_fusion=use_graph_fusion,
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
    print(
        f"\nAblation {name} | device={device} | train={train_sample_count} | "
        f"valid={valid_sample_count} | test={test_sample_count} | input={input_description}"
    )

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
                    key: value.to(device)
                    for key, value in batch.items()
                    if key in {
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
                device_batch = {key: value.to(device) for key, value in batch.items()}
                loss = criterion(model(device_batch), scores)
                batch_count = scores.shape[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * batch_count
            total_items += batch_count

        valid_metrics = evaluate_ablation(
            model, valid_loader, device, args.neutral_threshold, prepared_mode
        )
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
            f"valid Loss {valid_metrics['loss']:.4f} | "
            f"valid Acc {valid_metrics['accuracy']:.4f} | "
            f"valid Macro-F1 {valid_metrics['macro_f1']:.4f}"
        )
        improved, should_stop = checkpoint_selector.update(
            valid_metrics["macro_f1"], valid_metrics["loss"]
        )
        if improved:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if should_stop:
            print(f"Early stopping after {epoch} epochs; validation Macro-F1 did not improve.")
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint produced for ablation {name}.")
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "training_history.csv", history)
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "best_model.pt"
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
                "use_modality_embedding": use_modality_embedding,
                "use_ahl": use_ahl,
                "use_cross_modal_fusion": use_cross_modal_fusion,
                "use_graph_fusion": use_graph_fusion,
                "output_dim": 2 if prepared_mode else 1,
            },
            "label_mapping": label_mapping,
            "neutral_threshold": args.neutral_threshold,
            "best_valid_macro_f1": checkpoint_selector.best_macro_f1,
            "best_valid_loss": checkpoint_selector.best_validation_loss,
            "seed": args.seed,
            "training_config": {
                "max_epochs": args.epochs,
                "early_stopping_patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
            },
        },
        checkpoint_path,
    )

    train_metrics = evaluate_ablation(
        model, train_eval_loader, device, args.neutral_threshold, prepared_mode
    )
    valid_metrics = evaluate_ablation(
        model, valid_loader, device, args.neutral_threshold, prepared_mode
    )
    test_metrics = evaluate_ablation(
        model, test_loader, device, args.neutral_threshold, prepared_mode
    )
    result: dict[str, Any] = {
        "ablation": name,
        "train_samples": train_sample_count,
        "valid_accuracy": valid_metrics["accuracy"],
        "valid_macro_f1": valid_metrics["macro_f1"],
        "valid_weighted_f1": valid_metrics["weighted_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "train_weighted_f1": train_metrics["weighted_f1"],
        "best_valid_macro_f1": checkpoint_selector.best_macro_f1,
        "epochs_trained": len(history),
        "use_modality_embedding": use_modality_embedding,
        "use_ahl": use_ahl,
        "use_cross_modal_fusion": use_cross_modal_fusion,
        "use_graph_fusion": use_graph_fusion,
        "checkpoint": str(checkpoint_path),
    }
    write_csv(output_dir / "metrics.csv", [result])
    print(
        f"{name} | Test Accuracy {test_metrics['accuracy']:.4f} | "
        f"Test Macro-F1 {test_metrics['macro_f1']:.4f}"
    )
    return result

def mean_and_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_seed = {
        int(row["seed"]): float(row["test_macro_f1"])
        for row in rows
        if row["ablation"] == "full"
    }
    metric_names = (
        "valid_accuracy", "valid_macro_f1", "valid_weighted_f1",
        "test_accuracy", "test_macro_f1", "test_weighted_f1", "epochs_trained",
    )
    summaries: list[dict[str, Any]] = []
    for name, switches in ABLATION_CONFIGS.items():
        group = [row for row in rows if row["ablation"] == name]
        if not group:
            continue
        summary: dict[str, Any] = {
            "ablation": name,
            "seeds_completed": len(group),
            "train_samples": group[0]["train_samples"],
            "use_modality_embedding": switches[0],
            "use_ahl": switches[1],
            "use_cross_modal_fusion": switches[2],
            "use_graph_fusion": switches[3],
        }
        for metric in metric_names:
            mean, std = mean_and_std([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        deltas = [
            float(row["test_macro_f1"]) - full_by_seed[int(row["seed"])]
            for row in group
        ]
        summary["paired_test_macro_f1_delta_mean"], summary["paired_test_macro_f1_delta_std"] = mean_and_std(deltas)
        summaries.append(summary)
    return summaries

def add_summary_display_values(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        for metric in (
            "valid_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1",
            "paired_test_macro_f1_delta",
        ):
            row[f"{metric}_mean_std"] = (
                f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"
            )
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, include_seeds=False, include_patience=False)
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
    parser.add_argument("--output-dir", type=Path, default=Path("msa_almt_ablation_standard"))
    parser.add_argument(
        "--ablation",
        choices=("all", *ABLATION_CONFIGS.keys()),
        default="all",
        help="Run all four component ablations plus full ALMT, or select one variant.",
    )
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
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111, 1112, 1113])
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    validate_training_arguments(parser, args, include_seeds=False, include_patience=False)
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
    if args.patience < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("patience must be positive and seeds must be non-empty without duplicates")
    if args.prepared_dir is not None and args.embedding_file is not None:
        parser.error("--prepared-dir and --embedding-file are mutually exclusive")

    set_seed(args.split_seed)
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

    if args.ablation == "all":
        selected_ablations = list(ABLATION_CONFIGS.items())
    elif args.ablation == "full":
        selected_ablations = [("full", ABLATION_CONFIGS["full"])]
    else:
        selected_ablations = [
            ("full", ABLATION_CONFIGS["full"]),
            (args.ablation, ABLATION_CONFIGS[args.ablation]),
        ]
    root_output_dir = args.output_dir
    root_output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    print(
        f"ALMT ablation | device={device} | train={len(train_data)} | "
        f"valid={len(valid_data)} | test={len(test_data)} | seeds={args.seeds} | "
        f"patience={args.patience} | split_seed={args.split_seed} | input={input_description}"
    )
    for seed in args.seeds:
        args.seed = seed
        set_seed(seed)
        args.output_dir = root_output_dir / f"seed_{seed}"
        seed_results: list[dict[str, Any]] = []
        for name, switches in selected_ablations:
            result = run_ablation(
                name,
                switches,
                train_data,
                valid_data,
                test_data,
                feature_dims,
                label_mapping,
                "prepared_csd_binary_clips" if prepared_mode else "pooled_embeddings" if args.embedding_file is not None else "csd_sequences",
                input_description,
                args,
                device,
                len(train_data),
                len(valid_data),
                len(test_data),
                prepared_mode,
            )
            result["seed"] = seed
            seed_results.append(result)
            all_results.append(result)
            write_csv(root_output_dir / f"ablation_results_seed_{seed}.csv", seed_results)
            write_csv(root_output_dir / "ablation_results_by_seed.csv", all_results)

    summary = add_summary_display_values(summarize_results(all_results))
    write_csv(root_output_dir / "ablation_results_summary.csv", summary)
    print_summary_table(
        "Final ALMT ablation results",
        (
            ("Ablation", "ablation"),
            ("Seeds", "seeds_completed"),
            ("Train N", "train_samples"),
            ("Valid Macro-F1 Mean±SD", "valid_macro_f1_mean_std"),
            ("Test Acc Mean±SD", "test_accuracy_mean_std"),
            ("Test Macro-F1 Mean±SD", "test_macro_f1_mean_std"),
            ("Test Weighted-F1 Mean±SD", "test_weighted_f1_mean_std"),
            ("Paired ΔTest F1 vs Full", "paired_test_macro_f1_delta_mean_std"),
            ("Modality Embedding", "use_modality_embedding"),
            ("AHL", "use_ahl"),
            ("Cross-modal Fusion", "use_cross_modal_fusion"),
            ("Graph Fusion", "use_graph_fusion"),
        ),
        summary,
    )
    print(f"Per-seed results: {root_output_dir / 'ablation_results_by_seed.csv'}")
    print(f"Mean/std summary: {root_output_dir / 'ablation_results_summary.csv'}")

if __name__ == "__main__":
    main()
