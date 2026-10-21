from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("text", "audio", "vision")
FEATURE_DIMS = {"text": 768, "audio": 73, "vision": 40}

class PreparedCSDClipDataset(Dataset[dict[str, Any]]):
	def __init__(self, prepared_dir: Path, split: str):
		self.split_dir = Path(prepared_dir) / split
		required = [
			"text.npy", "audio.npy", "vision.npy", "text_mask.npy", "audio_mask.npy",
			"vision_mask.npy", "offsets.npy", "durations.npy", "labels.npy", "scores.npy",
		]
		missing = [name for name in required if not (self.split_dir / name).is_file()]
		if missing:
			raise FileNotFoundError(
				f"Prepared {split} split is missing {', '.join(missing)} in {self.split_dir}"
			)

		self.features = {
			modality: np.load(
				self.split_dir / f"{modality}.npy", mmap_mode="r", allow_pickle=False
			)
			for modality in MODALITIES
		}
		self.masks = {
			modality: np.load(
				self.split_dir / f"{modality}_mask.npy", mmap_mode="r", allow_pickle=False
			)
			for modality in MODALITIES
		}
		self.offsets = np.asarray(
			np.load(self.split_dir / "offsets.npy", allow_pickle=False), dtype=np.int64
		)
		self.durations = np.asarray(
			np.load(self.split_dir / "durations.npy", allow_pickle=False), dtype=np.float32
		)
		self.labels = np.asarray(
			np.load(self.split_dir / "labels.npy", allow_pickle=False), dtype=np.int64
		)
		self.scores = np.asarray(
			np.load(self.split_dir / "scores.npy", allow_pickle=False), dtype=np.float32
		)

		if self.labels.ndim != 1 or not np.isin(self.labels, (0, 1)).all():
			raise ValueError(f"Prepared {split} labels must be binary integers 0 or 1.")
		if self.scores.shape != self.labels.shape or not np.isfinite(self.scores).all():
			raise ValueError(f"Prepared {split} scores must be finite and match labels.")
		if self.offsets.shape != (len(self.labels) + 1,) or self.offsets[0] != 0:
			raise ValueError(f"Prepared {split} offsets must contain one boundary per clip.")
		if self.offsets[-1] != len(self.durations) or np.any(np.diff(self.offsets) <= 0):
			raise ValueError(f"Prepared {split} offsets do not match the window durations.")
		if not np.isfinite(self.durations).all() or np.any(self.durations <= 0):
			raise ValueError(f"Prepared {split} durations must be finite and positive.")
		for modality in MODALITIES:
			values = self.features[modality]
			mask = self.masks[modality]
			if values.ndim != 3 or values.shape[0] != len(self.durations):
				raise ValueError(f"Prepared {split} {modality} must have shape [windows, steps, dim].")
			if values.shape[2] != FEATURE_DIMS[modality] or mask.shape != values.shape[:2]:
				raise ValueError(f"Prepared {split} {modality} feature or mask dimensions differ.")

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, index: int) -> dict[str, Any]:
		first, last = int(self.offsets[index]), int(self.offsets[index + 1])
		return {
			"features": {
				modality: np.array(self.features[modality][first:last], dtype=np.float32, copy=True)
				for modality in MODALITIES
			},
			"masks": {
				modality: np.array(self.masks[modality][first:last], dtype=np.bool_, copy=True)
				for modality in MODALITIES
			},
			"durations": torch.from_numpy(self.durations[first:last].copy()),
			"label": torch.tensor(self.labels[index], dtype=torch.long),
		}

def prepared_clip_collate(clips: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
	feature_parts: dict[str, list[torch.Tensor]] = {name: [] for name in MODALITIES}
	mask_parts: dict[str, list[torch.Tensor]] = {name: [] for name in MODALITIES}
	window_to_clip: list[torch.Tensor] = []
	duration_parts: list[torch.Tensor] = []
	for clip_index, clip in enumerate(clips):
		window_count = clip["durations"].numel()
		window_to_clip.append(torch.full((window_count,), clip_index, dtype=torch.long))
		duration_parts.append(clip["durations"])
		for modality in MODALITIES:
			feature_parts[modality].append(torch.from_numpy(clip["features"][modality]))
			mask_parts[modality].append(torch.from_numpy(clip["masks"][modality]))

	batch: dict[str, torch.Tensor] = {
		"labels": torch.stack([clip["label"] for clip in clips]),
		"window_to_clip": torch.cat(window_to_clip),
		"durations": torch.cat(duration_parts),
	}
	for modality in MODALITIES:
		batch[f"{modality}_features"] = torch.cat(feature_parts[modality], dim=0)
		batch[f"{modality}_mask"] = torch.cat(mask_parts[modality], dim=0)
	return batch

def aggregate_window_logits(
	window_logits: torch.Tensor,
	window_to_clip: torch.Tensor,
	durations: torch.Tensor,
	clip_count: int,
) -> torch.Tensor:
	weights = durations.to(window_logits.dtype).unsqueeze(-1)
	clip_logits = window_logits.new_zeros((clip_count, window_logits.shape[-1]))
	clip_weights = window_logits.new_zeros((clip_count, 1))
	clip_logits.index_add_(0, window_to_clip, window_logits * weights)
	clip_weights.index_add_(0, window_to_clip, weights)
	if torch.any(clip_weights <= 0):
		raise ValueError("Every clip must have positive total window duration.")
	return clip_logits / clip_weights
