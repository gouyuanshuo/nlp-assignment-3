from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
from .Graph_MFN_data import (
	FEATURE_DIMS,
	PreparedCSDClipDataset,
	aggregate_window_logits,
	prepared_clip_collate,
)
from .Graph_MFN_model import GraphMFN
from .Graph_MFN_evaluation import evaluate, model_inputs, resolve_device, set_seed, write_csv
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
	args: argparse.Namespace,
	device: torch.device,
	output_dir: Path,
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
	output_dir.mkdir(parents=True, exist_ok=True)
	print(
		f"Graph-MFN | seed={args.seed} | device={device} | train={len(train_data)} | "
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

		valid_metrics = evaluate(model, valid_loader, device)[0]
		mean_loss = total_loss / max(1, total_clips)
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
		raise RuntimeError("Training completed without a best-validation checkpoint.")
	write_csv(output_dir / "training_history.csv", history)
	model.load_state_dict(best_state)
	checkpoint_path = output_dir / "best_model.pt"
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
				"architecture": "dynamic_fusion_graph_multiview_gated_memory",
			},
			"label_mapping": {"negative": 0, "positive": 1},
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
	metrics_rows = []
	confusion_rows: list[dict[str, Any]] = []
	roc_rows: list[dict[str, Any]] = []
	for split, loader in (
		("train", train_eval_loader),
		("valid", valid_loader),
		("test", test_loader),
	):
		metrics, labels, predictions = evaluate(model, loader, device)
		metrics_rows.append({"split": split, **metrics})
		if len(labels) > 0:
			probability_list: list[torch.Tensor] = []
			with torch.no_grad():
				for batch in loader:
					logits = model(model_inputs(batch, device))
					clip_logits = aggregate_window_logits(
						logits,
						batch["window_to_clip"].to(device),
						batch["durations"].to(device),
						batch["labels"].shape[0],
					)
					probability_list.append(clip_logits.softmax(dim=-1)[:, 1].cpu())
			probability_tensor = torch.cat(probability_list)
			confusion = binary_confusion_matrix(labels.cpu().numpy(), probability_tensor.numpy())
			confusion_rows.append(
				{
					"split": split,
					"tn": confusion["tn"],
					"fp": confusion["fp"],
					"fn": confusion["fn"],
					"tp": confusion["tp"],
					"roc_auc": binary_roc_auc(labels.cpu().numpy(), probability_tensor.numpy()),
				}
			)
			for point in binary_roc_curve(labels.cpu().numpy(), probability_tensor.numpy()):
				roc_rows.append({"split": split, **point})
	write_csv(output_dir / "metrics.csv", metrics_rows)
	write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
	write_csv(output_dir / "roc_curve.csv", roc_rows)
	for row in metrics_rows:
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
		"valid_accuracy": metrics_rows[1]["accuracy"],
		"valid_macro_f1": metrics_rows[1]["macro_f1"],
		"test_accuracy": metrics_rows[2]["accuracy"],
		"test_macro_f1": metrics_rows[2]["macro_f1"],
		"epochs_trained": len(history),
		"checkpoint": str(checkpoint_path),
	}

def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	add_training_arguments(parser)
	parser.add_argument("--prepared-dir", type=Path, default=Path("Data/prepared-csd"))
	parser.add_argument("--output-dir", type=Path, default=Path("graph_mfn_results"))
	parser.add_argument("--hidden-dim", type=int, default=128)
	parser.add_argument("--gat-heads", type=int, default=4)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--warmup-ratio", type=float, default=0.1)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
	args = parser.parse_args()
	validate_training_arguments(parser, args)
	if args.num_workers < 0 or not 0 <= args.warmup_ratio < 1:
		parser.error("num-workers must be non-negative and warmup-ratio must be in [0, 1)")
	if args.gat_heads <= 0 or args.hidden_dim % args.gat_heads != 0:
		parser.error("gat-heads must be positive and divide hidden-dim evenly")
	prepared_dir = args.prepared_dir.resolve()
	train_data = PreparedCSDClipDataset(prepared_dir, "train")
	valid_data = PreparedCSDClipDataset(prepared_dir, "valid")
	test_data = PreparedCSDClipDataset(prepared_dir, "test")
	device = resolve_device(args.device)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	results: list[dict[str, Any]] = []
	for seed in args.seeds:
		args.seed = seed
		result = train_one_seed(
			train_data,
			valid_data,
			test_data,
			args,
			device,
			args.output_dir / f"seed_{seed}",
		)
		results.append(result)
		write_csv(args.output_dir / f"metrics_seed_{seed}.csv", [result])
		write_csv(args.output_dir / "metrics_by_seed.csv", results)
	summary = summarize_seed_results(
		results,
		run_key="run",
		metric_names=("valid_accuracy", "valid_macro_f1", "test_accuracy", "test_macro_f1"),
	)
	add_mean_std_display(summary, ("valid_accuracy", "valid_macro_f1", "test_accuracy", "test_macro_f1"))
	write_csv(args.output_dir / "metrics_summary.csv", summary)
	print_summary_table(
		"Graph-MFN three-seed results",
		(("Run", "run"), ("Seeds", "seeds_completed"), ("Valid Acc", "valid_accuracy_mean_std"), ("Valid Macro-F1", "valid_macro_f1_mean_std"), ("Test Acc", "test_accuracy_mean_std"), ("Test Macro-F1", "test_macro_f1_mean_std")),
		summary,
	)
	print(f"Seed metrics: {args.output_dir / 'metrics_by_seed.csv'}")
	print(f"Summary metrics: {args.output_dir / 'metrics_summary.csv'}")

if __name__ == "__main__":
	main()
