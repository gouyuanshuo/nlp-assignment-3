from __future__ import annotations
import argparse
import math
import statistics
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..common.protocol import (
	ValidationCheckpoint,
	add_training_arguments,
	add_mean_std_display,
	summarize_seed_results,
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

ABLATION_CONFIGS: dict[str, tuple[bool, bool, bool, bool]] = {
	"full": (True, True, True, True),
	"without_dynamic_graph": (False, True, True, True),
	"without_view_memory": (True, False, True, True),
	"without_fusion_memory": (True, True, False, True),
	"without_temporal_attention": (True, True, True, False),
}
SWITCH_NAMES = (
	"use_dynamic_graph",
	"use_view_memory",
	"use_fusion_memory",
	"use_temporal_attention",
)


def calculate_weighted_f1(labels: torch.Tensor, predictions: torch.Tensor) -> float:
	labels = labels.detach().long().cpu()
	predictions = predictions.detach().long().cpu()
	confusion = torch.zeros((2, 2), dtype=torch.float64)
	for actual, predicted in zip(labels, predictions):
		confusion[actual, predicted] += 1
	true_positive = confusion.diag()
	precision = true_positive / confusion.sum(dim=0).clamp_min(1)
	recall = true_positive / confusion.sum(dim=1).clamp_min(1)
	f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
	support = confusion.sum(dim=1)
	return float((f1 * support).sum() / support.sum().clamp_min(1))


def evaluate_metrics(model: GraphMFN, loader: DataLoader, device: torch.device) -> dict[str, float]:
	metrics, labels, predictions = evaluate(model, loader, device)
	return {**metrics, "weighted_f1": calculate_weighted_f1(labels, predictions)}


class AblationGraphMFN(GraphMFN):
	"""Graph-MFN with one independently switchable fusion component per run."""

	def __init__(
		self,
		hidden_dim: int,
		dropout: float,
		gat_heads: int,
		switches: tuple[bool, bool, bool, bool],
	):
		super().__init__(hidden_dim=hidden_dim, dropout=dropout, gat_heads=gat_heads)
		for name, enabled in zip(SWITCH_NAMES, switches):
			setattr(self, name, enabled)

	def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
		first_features = batch[f"{MODALITIES[0]}_features"]
		batch_size, time_steps = first_features.shape[:2]
		device = first_features.device
		dtype = first_features.dtype
		projected = {
			modality: self.input_projection[modality](batch[f"{modality}_features"])
			for modality in MODALITIES
		}
		masks = {
			modality: batch[f"{modality}_mask"].bool()
			for modality in MODALITIES
		}
		modality_states = [
			torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
			for _ in MODALITIES
		]
		modality_cells = [torch.zeros_like(modality_states[0]) for _ in MODALITIES]
		view_memories = [torch.zeros_like(modality_states[0]) for _ in MODALITIES]
		shared_memory = torch.zeros_like(modality_states[0])
		sequence_views: list[torch.Tensor] = []
		sequence_valid: list[torch.Tensor] = []

		for time_index in range(time_steps):
			current_valid = torch.stack(
				[masks[modality][:, time_index] for modality in MODALITIES], dim=1
			)
			for index, modality in enumerate(MODALITIES):
				candidate_state, candidate_cell = self.unimodal_cells[modality](
					projected[modality][:, time_index],
					(modality_states[index], modality_cells[index]),
				)
				active = current_valid[:, index].unsqueeze(-1)
				modality_states[index] = torch.where(
					active, candidate_state, modality_states[index]
				)
				modality_cells[index] = torch.where(
					active, candidate_cell, modality_cells[index]
				)

			nodes = torch.stack(modality_states, dim=1)
			if self.use_dynamic_graph:
				graph_nodes, messages = self.dynamic_graph(nodes, current_valid)
			else:
				graph_nodes = nodes * current_valid.unsqueeze(-1)
				messages = graph_nodes

			for index, cell in enumerate(self.view_memory_cells):
				active = current_valid[:, index].unsqueeze(-1)
				if self.use_view_memory:
					candidate_view = cell(
						torch.cat((graph_nodes[:, index], messages[:, index]), dim=-1),
						view_memories[index],
					)
				else:
					candidate_view = graph_nodes[:, index]
				view_memories[index] = torch.where(
					active, candidate_view, view_memories[index]
				)

			if self.use_fusion_memory:
				candidate_shared = self.fusion_memory(
					torch.cat(view_memories, dim=-1), shared_memory
				)
			else:
				candidate_shared = torch.stack(view_memories, dim=1).mean(dim=1)
			frame_active = current_valid.any(dim=1, keepdim=True)
			shared_memory = torch.where(frame_active, candidate_shared, shared_memory)
			sequence_views.append(torch.cat((*view_memories, shared_memory), dim=-1))
			sequence_valid.append(frame_active.squeeze(-1))

		sequence = torch.stack(sequence_views, dim=1)
		valid_steps = torch.stack(sequence_valid, dim=1)
		if self.use_temporal_attention:
			attention_logits = self.temporal_attention(sequence).squeeze(-1)
			attention_logits = attention_logits.masked_fill(~valid_steps, -1e4)
			weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
			representation = (sequence * weights).sum(dim=1)
		else:
			weights = valid_steps.unsqueeze(-1).to(sequence.dtype)
			representation = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
		return self.classifier(representation)


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
			**dict(zip(SWITCH_NAMES, switches)),
		}
		for metric in metric_names:
			mean, std = mean_and_std([float(row[metric]) for row in group])
			summary[f"{metric}_mean"] = mean
			summary[f"{metric}_std"] = std
		deltas = [
			float(row["test_macro_f1"]) - full_by_seed[int(row["seed"])]
			for row in group
		]
		delta_mean, delta_std = mean_and_std(deltas)
		summary["paired_test_macro_f1_delta_mean"] = delta_mean
		summary["paired_test_macro_f1_delta_std"] = delta_std
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


def print_summary(rows: list[dict[str, Any]]) -> None:
	columns = (
		("Ablation", "ablation"),
		("Seeds", "seeds_completed"),
		("Train N", "train_samples"),
		("Valid Macro-F1 Mean±SD", "valid_macro_f1_mean_std"),
		("Test Acc Mean±SD", "test_accuracy_mean_std"),
		("Test Macro-F1 Mean±SD", "test_macro_f1_mean_std"),
		("Test Weighted-F1 Mean±SD", "test_weighted_f1_mean_std"),
		("Paired ΔTest F1 vs Full", "paired_test_macro_f1_delta_mean_std"),
		("Dynamic Graph", "use_dynamic_graph"),
		("View Memory", "use_view_memory"),
		("Fusion Memory", "use_fusion_memory"),
		("Temporal Attention", "use_temporal_attention"),
	)
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
	print("\nFinal Graph-MFN ablation results")
	print(border)
	print("| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |")
	print(border)
	for row in values:
		print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
	print(border)


def run_ablation(
	name: str,
	switches: tuple[bool, bool, bool, bool],
	train_data: Any,
	valid_data: Any,
	test_data: Any,
	args: argparse.Namespace,
	device: torch.device,
) -> dict[str, Any]:
	set_seed(args.seed)
	loader_args = {
		"batch_size": args.batch_size,
		"collate_fn": prepared_clip_collate,
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
	model = AblationGraphMFN(
		hidden_dim=args.hidden_dim,
		dropout=args.dropout,
		gat_heads=args.gat_heads,
		switches=switches,
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
	print(
		f"\nAblation {name} | device={device} | train={len(train_data)} | "
		f"valid={len(valid_data)} | test={len(test_data)}"
	)

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

		valid_metrics = evaluate_metrics(model, valid_loader, device)
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
			print(f"Early stopping after {epoch} epochs; validation Macro-F1 did not improve.")
			break

	if best_state is None:
		raise RuntimeError(f"No best checkpoint produced for ablation {name}.")
	run_dir = args.output_dir / name
	run_dir.mkdir(parents=True, exist_ok=True)
	write_csv(run_dir / "training_history.csv", history)
	model.load_state_dict(best_state)
	checkpoint_path = run_dir / "best_model.pt"
	model_config = {
		"feature_dims": FEATURE_DIMS,
		"hidden_dim": args.hidden_dim,
		"dropout": args.dropout,
		"gat_heads": args.gat_heads,
		"output_dim": 2,
		**dict(zip(SWITCH_NAMES, switches)),
	}
	torch.save(
		{
			"model_state_dict": best_state,
			"input_mode": "prepared_csd_binary_clips",
			"model_config": model_config,
			"label_mapping": {"negative": 0, "positive": 1},
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
	train_metrics = evaluate_metrics(model, train_eval_loader, device)
	valid_metrics = evaluate_metrics(model, valid_loader, device)
	test_metrics = evaluate_metrics(model, test_loader, device)
	result: dict[str, Any] = {
		"ablation": name,
		"train_samples": len(train_data),
		"train_accuracy": train_metrics["accuracy"],
		"train_macro_f1": train_metrics["macro_f1"],
		"train_weighted_f1": train_metrics["weighted_f1"],
		"valid_accuracy": valid_metrics["accuracy"],
		"valid_macro_f1": valid_metrics["macro_f1"],
		"valid_weighted_f1": valid_metrics["weighted_f1"],
		"test_accuracy": test_metrics["accuracy"],
		"test_macro_f1": test_metrics["macro_f1"],
		"test_weighted_f1": test_metrics["weighted_f1"],
		"best_valid_macro_f1": checkpoint_selector.best_macro_f1,
		"best_valid_loss": checkpoint_selector.best_validation_loss,
		"epochs_trained": len(history),
		**dict(zip(SWITCH_NAMES, switches)),
		"checkpoint": str(checkpoint_path),
	}
	write_csv(run_dir / "metrics.csv", [result])
	print(
		f"{name} | Test Accuracy {result['test_accuracy']:.4f} | "
		f"Test Macro-F1 {result['test_macro_f1']:.4f}"
	)
	return result


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	add_training_arguments(parser)
	parser.add_argument("--prepared-dir", type=Path, default=Path("Data/prepared-csd"))
	parser.add_argument("--output-dir", type=Path, default=Path("graph_mfn_ablation_standard"))
	parser.add_argument(
		"--ablation",
		choices=("all", *ABLATION_CONFIGS),
		default="all",
		help="Run the full model and four component ablations, or select one variant.",
	)
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
	if not 0 <= args.warmup_ratio < 1 or args.num_workers < 0:
		parser.error("warmup-ratio must be in [0, 1) and num-workers non-negative")

	prepared_dir = args.prepared_dir.resolve()
	train_data = PreparedCSDClipDataset(prepared_dir, "train")
	valid_data = PreparedCSDClipDataset(prepared_dir, "valid")
	test_data = PreparedCSDClipDataset(prepared_dir, "test")
	device = resolve_device(args.device)
	selected = (
		list(ABLATION_CONFIGS.items())
		if args.ablation == "all"
		else (
			[("full", ABLATION_CONFIGS["full"])]
			if args.ablation == "full"
			else [
				("full", ABLATION_CONFIGS["full"]),
				(args.ablation, ABLATION_CONFIGS[args.ablation]),
			]
		)
	)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	root_output_dir = args.output_dir
	results: list[dict[str, Any]] = []
	print(
		f"Graph-MFN ablation | device={device} | train={len(train_data)} | "
		f"valid={len(valid_data)} | test={len(test_data)} | "
		f"seeds={args.seeds} | patience={args.patience} | data={prepared_dir}"
	)
	for seed in args.seeds:
		args.seed = seed
		set_seed(seed)
		args.output_dir = root_output_dir / f"seed_{seed}"
		seed_results: list[dict[str, Any]] = []
		for name, switches in selected:
			result = run_ablation(
				name, switches, train_data, valid_data, test_data, args, device
			)
			result["seed"] = seed
			seed_results.append(result)
			results.append(result)
			write_csv(root_output_dir / f"ablation_results_seed_{seed}.csv", seed_results)
			write_csv(root_output_dir / "ablation_results_by_seed.csv", results)

	summary = add_summary_display_values(summarize_results(results))
	write_csv(root_output_dir / "ablation_results_summary.csv", summary)
	print_summary(summary)
	print(f"Per-seed results: {root_output_dir / 'ablation_results_by_seed.csv'}")
	print(f"Mean/std summary: {root_output_dir / 'ablation_results_summary.csv'}")


if __name__ == "__main__":
	main()
