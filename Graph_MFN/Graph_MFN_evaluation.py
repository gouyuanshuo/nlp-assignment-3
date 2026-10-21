from __future__ import annotations
import csv
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .Graph_MFN_data import MODALITIES, aggregate_window_logits
from .Graph_MFN_model import GraphMFN


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

def resolve_device(requested: str) -> torch.device:
	if requested == "auto":
		if torch.cuda.is_available():
			return torch.device("cuda")
		if torch.backends.mps.is_available():
			return torch.device("mps")
		return torch.device("cpu")
	if requested == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
	if requested == "mps" and not torch.backends.mps.is_available():
		raise RuntimeError("MPS was requested, but this PyTorch build/device does not support MPS.")
	return torch.device(requested)

def calculate_binary_metrics(
	labels: torch.Tensor, predictions: torch.Tensor
) -> dict[str, float]:
	labels = labels.detach().long().cpu()
	predictions = predictions.detach().long().cpu()
	confusion = torch.zeros((2, 2), dtype=torch.float64)
	for actual, predicted in zip(labels, predictions):
		confusion[actual, predicted] += 1
	true_positive = confusion.diag()
	precision = true_positive / confusion.sum(dim=0).clamp_min(1)
	recall = true_positive / confusion.sum(dim=1).clamp_min(1)
	f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
	return {
		"accuracy": float((labels == predictions).float().mean()),
		"macro_f1": float(f1.mean()),
	}

def model_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
	return {
		key: value.to(device)
		for key, value in batch.items()
		if any(key == f"{modality}_{suffix}" for modality in MODALITIES for suffix in ("features", "mask"))
	}

def evaluate(
	model: GraphMFN, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
	model.eval()
	all_labels: list[torch.Tensor] = []
	all_predictions: list[torch.Tensor] = []
	total_loss = 0.0
	total_clips = 0
	with torch.no_grad():
		for batch in loader:
			labels = batch["labels"]
			logits = model(model_inputs(batch, device))
			clip_logits = aggregate_window_logits(
				logits,
				batch["window_to_clip"].to(device),
				batch["durations"].to(device),
				labels.shape[0],
			)
			all_labels.append(labels)
			all_predictions.append(clip_logits.argmax(dim=1).cpu())
			total_loss += nn.functional.cross_entropy(
				clip_logits, labels.to(device), reduction="sum"
			).item()
			total_clips += labels.shape[0]
	labels_tensor = torch.cat(all_labels)
	predictions_tensor = torch.cat(all_predictions)
	metrics = calculate_binary_metrics(labels_tensor, predictions_tensor)
	metrics["loss"] = total_loss / max(1, total_clips)
	return (
		metrics,
		labels_tensor,
		predictions_tensor,
	)

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as output_file:
		writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
