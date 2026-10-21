from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
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
    set_seed,
    write_csv,
)

from ..common.protocol import (
    ValidationCheckpoint,
    add_training_arguments,
    add_mean_std_display,
    binary_confusion_matrix,
    binary_roc_auc,
    binary_roc_curve,
    print_summary_table,
    summarize_seed_results,
    validate_training_arguments,
)

def train_one_seed(
    train_data: Any,
    valid_data: Any,
    test_data: Any,
    feature_dims: dict[str, int],
    label_mapping: dict[str, int],
    input_mode: str,
    input_description: str,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    prepared_mode: bool,
) -> dict[str, Any]:
    set_seed(args.seed)
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
        output_dim=2 if prepared_mode else 1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    checkpoint_selector = ValidationCheckpoint(args.patience)
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"ALMT | seed={args.seed} | device={device} | train={len(train_data)} | "
        f"valid={len(valid_data)} | test={len(test_data)} | input={input_description}"
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
                item_count = labels.shape[0]
            else:
                scores = batch.pop("scores").to(device)
                device_batch = {name: value.to(device) for name, value in batch.items()}
                loss = criterion(model(device_batch), scores)
                item_count = scores.shape[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * item_count
            total_items += item_count

        if prepared_mode:
            valid_metrics, _, _ = evaluate_prepared(model, valid_loader, device)
        else:
            valid_metrics, _, _ = evaluate(model, valid_loader, device, args.neutral_threshold)
        train_loss = total_loss / max(1, total_items)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_metrics["loss"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | Loss {train_loss:.4f} | "
            f"valid Acc {valid_metrics['accuracy']:.4f} | "
            f"valid Macro-F1 {valid_metrics['macro_f1']:.4f} | "
            f"valid Loss {valid_metrics['loss']:.4f}"
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
            print(f"Early stopping after {epoch} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training completed without producing a best-validation checkpoint.")
    write_csv(output_dir / "training_history.csv", history)
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "best_model.pt"
    input_mode = (
        "prepared_csd_binary_clips"
        if prepared_mode
        else "pooled_embeddings" if input_mode == "pooled_embeddings" else "csd_sequences"
    )
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
            "best_valid_macro_f1": checkpoint_selector.best_macro_f1,
            "best_valid_loss": checkpoint_selector.best_validation_loss,
            "seed": args.seed,
            "training_config": {
                "max_epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
        },
        checkpoint_path,
    )
    split_results = []
    confusion_rows: list[dict[str, Any]] = []
    roc_rows: list[dict[str, Any]] = []
    for split, loader in (
        ("train", train_eval_loader),
        ("valid", valid_loader),
        ("test", test_loader),
    ):
        if prepared_mode:
            metrics, labels, predictions = evaluate_prepared(model, loader, device)
        else:
            metrics, labels, predictions = evaluate(model, loader, device, args.neutral_threshold)
        split_results.append({"split": split, **metrics})
        if prepared_mode:
            positive_probabilities = []
            with torch.no_grad():
                for batch in loader:
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
                        window_logits,
                        batch["window_to_clip"].to(device),
                        batch["durations"].to(device),
                        batch["labels"].shape[0],
                    )
                    positive_probabilities.append(clip_logits.softmax(dim=-1)[:, 1].cpu())
            positive_probabilities = torch.cat(positive_probabilities)
        else:
            with torch.no_grad():
                positive_probabilities = []
                for batch in loader:
                    scores = batch.pop("scores").to(device)
                    device_batch = {name: value.to(device) for name, value in batch.items()}
                    output = model(device_batch)
                    positive_probabilities.append(output.cpu())
                positive_probabilities = torch.cat(positive_probabilities)
            positive_probabilities = torch.sigmoid(positive_probabilities)
        labels_array = labels.detach().cpu().numpy()
        probabilities_array = positive_probabilities.detach().cpu().numpy().reshape(-1)
        confusion = binary_confusion_matrix(labels_array, probabilities_array)
        confusion_rows.append(
            {
                "split": split,
                "tn": confusion["tn"],
                "fp": confusion["fp"],
                "fn": confusion["fn"],
                "tp": confusion["tp"],
                "roc_auc": binary_roc_auc(labels_array, probabilities_array),
            }
        )
        for point in binary_roc_curve(labels_array, probabilities_array):
            roc_rows.append({"split": split, **point})
    write_csv(output_dir / "metrics.csv", split_results)
    write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    write_csv(output_dir / "roc_curve.csv", roc_rows)
    for row in split_results:
        print(
            f"{row['split'].title()} | Accuracy {row['accuracy']:.4f} | "
            f"Macro-F1 {row['macro_f1']:.4f}"
        )
    for row in confusion_rows:
        print(
            f"{row['split'].title()} | tn={row['tn']} fp={row['fp']} fn={row['fn']} tp={row['tp']} | ROC-AUC {row['roc_auc']:.4f}"
        )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Confusion matrix: {output_dir / 'confusion_matrix.csv'}")
    print(f"ROC curve: {output_dir / 'roc_curve.csv'}")
    return {
        "run": "full",
        "seed": args.seed,
        "train_samples": len(train_data),
        "valid_accuracy": split_results[1]["accuracy"],
        "valid_macro_f1": split_results[1]["macro_f1"],
        "test_accuracy": split_results[2]["accuracy"],
        "test_macro_f1": split_results[2]["macro_f1"],
        "epochs_trained": len(history),
        "checkpoint": str(checkpoint_path),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--dataset-dir", type=Path, default=Path("CMU-MOSEI"))
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("msa_almt_results"))
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
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    validate_training_arguments(parser, args)
    if args.max_input_length < 1 or args.num_heads < 1 or args.hidden_dim % args.num_heads:
        parser.error("max-input-length and num-heads must be positive; hidden-dim divisible by num-heads")
    if args.ahl_depth != 3 or args.fusion_depth < 1:
        parser.error("ALMT requires ahl-depth=3 and positive fusion-depth")
    if args.neutral_threshold < 0 or not 0 <= args.warmup_ratio < 1 or args.num_workers < 0:
        parser.error("neutral-threshold, warmup-ratio, or num-workers is invalid")
    if args.prepared_dir is not None and args.embedding_file is not None:
        parser.error("--prepared-dir and --embedding-file are mutually exclusive")

    set_seed(args.split_seed)
    if args.prepared_dir is not None:
        prepared_dir = args.prepared_dir.resolve()
        train_data = PreparedCSDClipDataset(prepared_dir, "train")
        valid_data = PreparedCSDClipDataset(prepared_dir, "valid")
        test_data = PreparedCSDClipDataset(prepared_dir, "test")
        feature_dims = {modality: int(values.shape[-1]) for modality, values in train_data.features.items()}
        label_mapping = {"negative": 0, "positive": 1}
        input_description = "HANDOFF prepared-csd windows; binary labels, neutral excluded"
        input_mode = "prepared_csd_binary_clips"
        prepared_mode = True
    elif args.embedding_file is not None:
        embedding_path = args.embedding_file.resolve()
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Embedding file does not exist: {embedding_path}")
        split_data, feature_dims, label_mapping = load_embedding_splits(embedding_path)
        train_data, valid_data, test_data = (split_data[name] for name in ("train", "valid", "test"))
        input_description = f"pooled NPZ embeddings ({embedding_path.name}; one token per modality)"
        input_mode = "pooled_embeddings"
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
        input_mode = "csd_sequences"
        prepared_mode = False

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    print(
        f"ALMT | device={device} | train={len(train_data)} | valid={len(valid_data)} | "
        f"test={len(test_data)} | seeds={args.seeds} | input={input_description}"
    )
    for seed in args.seeds:
        args.seed = seed
        seed_result = train_one_seed(
            train_data, valid_data, test_data, feature_dims, label_mapping, input_mode,
            input_description, args, device, args.output_dir / f"seed_{seed}", prepared_mode,
        )
        results.append(seed_result)
        write_csv(args.output_dir / f"metrics_seed_{seed}.csv", [seed_result])
        write_csv(args.output_dir / "metrics_by_seed.csv", results)
    summary = summarize_seed_results(
        results,
        run_key="run",
        metric_names=("valid_accuracy", "valid_macro_f1", "test_accuracy", "test_macro_f1"),
    )
    add_mean_std_display(summary, ("valid_accuracy", "valid_macro_f1", "test_accuracy", "test_macro_f1"))
    write_csv(args.output_dir / "metrics_summary.csv", summary)
    print_summary_table(
        "ALMT three-seed results",
        (("Run", "run"), ("Seeds", "seeds_completed"), ("Valid Acc", "valid_accuracy_mean_std"), ("Valid Macro-F1", "valid_macro_f1_mean_std"), ("Test Acc", "test_accuracy_mean_std"), ("Test Macro-F1", "test_macro_f1_mean_std")),
        summary,
    )
    print(f"Seed metrics: {args.output_dir / 'metrics_by_seed.csv'}")
    print(f"Summary metrics: {args.output_dir / 'metrics_summary.csv'}")

if __name__ == "__main__":
    main()
