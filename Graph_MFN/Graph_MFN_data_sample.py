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
from ..common.protocol import (
	ValidationCheckpoint,
	add_training_arguments,
	add_mean_std_display,
	validate_training_arguments,
)
from .Graph_MFN_data import (
    MODALITIES,
    FEATURE_DIMS,
    PreparedCSDClipDataset,
    prepared_clip_collate,
    aggregate_window_logits,
)
from .Graph_MFN_model import GraphMFN
from .Graph_MFN_evaluation import (
    set_seed,
    resolve_device,
    model_inputs,
    evaluate,
    write_csv,
)


def nested_stratified_order(labels: list[int], seed: int) -> list[int]:
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


def print_summary(rows: list[dict[str, Any]]) -> None:
	columns = (
		("Train %", "training_data_percent"),
		("Seeds", "seeds_completed"),
		("Train N", "train_samples"),
		("Valid Acc Mean±SD", "valid_accuracy_mean_std"),
		("Valid Macro-F1 Mean±SD", "valid_macro_f1_mean_std"),
		("Test Acc Mean±SD", "test_accuracy_mean_std"),
		("Test Macro-F1 Mean±SD", "test_macro_f1_mean_std"),
	)
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
	print("\nFinal Graph-MFN data-sample results")
	print(border)
	print("| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |")
	print(border)
	for row in values:
		print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
	print(border)


def summarize_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	summaries: list[dict[str, Any]] = []
	for percentage in sorted({int(row["training_data_percent"]) for row in rows}):
		group = [row for row in rows if int(row["training_data_percent"]) == percentage]
		summary: dict[str, Any] = {
			"training_data_percent": percentage,
			"seeds_completed": len(group),
			"train_samples": group[0]["train_samples"],
		}
		for metric in (
			"train_accuracy", "train_macro_f1", "valid_accuracy", "valid_macro_f1",
			"test_accuracy", "test_macro_f1",
		):
			values = [float(row[metric]) for row in group]
			mean = statistics.mean(values)
			std = statistics.stdev(values) if len(values) > 1 else 0.0
			summary[f"{metric}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
		summaries.append(summary)
	return summaries


def train_sample(
	percentage: int,
	train_data: Dataset,
	indices: list[int],
	valid_data: Dataset,
	test_data: Dataset,
	args: argparse.Namespace,
	device: torch.device,
) -> dict[str, Any]:
	set_seed(args.seed)
	train_subset = Subset(train_data, indices)
	loader_args = {
		"batch_size": args.batch_size,
		"collate_fn": prepared_clip_collate,
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
	model = GraphMFN(
		hidden_dim=args.hidden_dim,
		dropout=args.dropout,
		gat_heads=args.gat_heads,
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
	checkpoint_selector = ValidationCheckpoint(args.patience)
	best_state: dict[str, torch.Tensor] | None = None
	history: list[dict[str, Any]] = []
	print(f"\nData sample {percentage:3d}% | train={len(train_subset)} | device={device}")

	for epoch in range(1, args.epochs + 1):
		model.train()
		total_loss = 0.0
		total_clips = 0
		for batch in train_loader:
			labels = batch["labels"].to(device)
			window_logits = model(model_inputs(batch, device))
			clip_logits = aggregate_window_logits(
				window_logits,
				batch["window_to_clip"].to(device),
				batch["durations"].to(device),
				labels.shape[0],
			)
			loss = nn.functional.cross_entropy(clip_logits, labels)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()
			total_loss += loss.item() * labels.shape[0]
			total_clips += labels.shape[0]

		valid_metrics, _, _ = evaluate(model, valid_loader, device)
		mean_loss = total_loss / max(1, total_clips)
		history.append(
			{
				"epoch": epoch,
				"train_loss": mean_loss,
				"valid_accuracy": valid_metrics["accuracy"],
				"valid_macro_f1": valid_metrics["macro_f1"],
				"valid_loss": valid_metrics["loss"],
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
				key: value.detach().cpu().clone()
				for key, value in model.state_dict().items()
			}
		if should_stop:
			print(f"Early stopping after {epoch} epochs.")
			break

	if best_state is None:
		raise RuntimeError(f"No best checkpoint produced for {percentage}% data.")
	run_dir = args.output_dir / f"sample_{percentage:03d}pct"
	run_dir.mkdir(parents=True, exist_ok=True)
	write_csv(run_dir / "training_history.csv", history)
	model.load_state_dict(best_state)
	checkpoint_path = run_dir / "best_model.pt"
	torch.save(
		{
			"model_state_dict": best_state,
			"input_mode": "prepared_csd_binary_clips",
			"model_config": {
				"feature_dims": FEATURE_DIMS,
				"hidden_dim": args.hidden_dim,
				"dropout": args.dropout,
				"gat_heads": args.gat_heads,
				"output_dim": 2,
			},
			"label_mapping": {"negative": 0, "positive": 1},
			"sample_percentage": percentage,
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
	train_metrics = evaluate(model, train_eval_loader, device)[0]
	valid_metrics = evaluate(model, valid_loader, device)[0]
	test_metrics = evaluate(model, test_loader, device)[0]
	result = {
		"training_data_percent": percentage,
		"train_samples": len(train_subset),
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
	parser.add_argument("--prepared-dir", type=Path, default=Path("Data/prepared-csd"))
	parser.add_argument("--output-dir", type=Path, default=Path("graph_mfn_samples_standard"))
	parser.add_argument("--percentages", nargs="+", type=int, default=list(range(10, 101, 10)))
	parser.add_argument("--split-seed", type=int, default=42)
	parser.add_argument("--hidden-dim", type=int, default=128)
	parser.add_argument("--gat-heads", type=int, default=4)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--warmup-ratio", type=float, default=0.1)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
	args = parser.parse_args()
	validate_training_arguments(parser, args)
	if args.hidden_dim < 1:
		parser.error("hidden-dim must be positive")
	if args.gat_heads < 1 or args.hidden_dim % args.gat_heads:
		parser.error("gat-heads must be positive and divide hidden-dim evenly")
	if not args.percentages or any(not 1 <= value <= 100 for value in args.percentages):
		parser.error("percentages must be between 1 and 100")
	if len(set(args.percentages)) != len(args.percentages):
		parser.error("percentages must not contain duplicates")
	if not 0 <= args.warmup_ratio < 1 or args.num_workers < 0:
		parser.error("warmup-ratio must be in [0, 1) and num-workers non-negative")

	set_seed(args.split_seed)
	prepared_dir = args.prepared_dir.resolve()
	train_data = PreparedCSDClipDataset(prepared_dir, "train")
	valid_data = PreparedCSDClipDataset(prepared_dir, "valid")
	test_data = PreparedCSDClipDataset(prepared_dir, "test")
	labels = train_data.labels.tolist()
	order = nested_stratified_order(labels, args.split_seed)
	device = resolve_device(args.device)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	root_output_dir = args.output_dir
	results: list[dict[str, Any]] = []
	print(
		f"Graph-MFN data-sample | device={device} | train pool={len(train_data)} | "
		f"valid={len(valid_data)} | test={len(test_data)} | seeds={args.seeds} | "
		f"split_seed={args.split_seed} | data={prepared_dir}"
	)

	for seed in args.seeds:
		args.seed = seed
		set_seed(seed)
		args.output_dir = root_output_dir / f"seed_{seed}"
		seed_results: list[dict[str, Any]] = []
		for percentage in args.percentages:
			sample_size = max(1, math.ceil(len(order) * percentage / 100))
			result = train_sample(
				percentage, train_data, order[:sample_size], valid_data, test_data, args, device
			)
			result["seed"] = seed
			seed_results.append(result)
			results.append(result)
			write_csv(root_output_dir / f"data_sample_results_seed_{seed}.csv", seed_results)
			write_csv(root_output_dir / "data_sample_results_by_seed.csv", results)
	summary = summarize_samples(results)
	write_csv(root_output_dir / "data_sample_results_summary.csv", summary)
	print_summary(summary)
	print(f"Per-seed results: {root_output_dir / 'data_sample_results_by_seed.csv'}")
	print(f"Mean/std summary: {root_output_dir / 'data_sample_results_summary.csv'}")


if __name__ == "__main__":
	main()
