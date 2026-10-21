from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from mosei_dataset import MoseiFeatureDataset

MODALITIES = ("text", "image", "audio")

class SequenceSubset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, dataset: MoseiFeatureDataset, indices: list[int], max_length: int):
        self.dataset = dataset
        self.indices = indices
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self.indices[index]
        sample = self.dataset[source_index]
        for modality in MODALITIES:
            sequence = sample[f"{modality}_features"]
            if sequence.shape[0] > self.max_length:
                positions = torch.linspace(
                    0, sequence.shape[0] - 1, self.max_length, device=sequence.device
                ).round().long()
                sequence = sequence.index_select(0, positions)
            sample[f"{modality}_features"] = sequence
        sample["score"] = torch.tensor(
            self.dataset.items[source_index]["sentiment_score"], dtype=torch.float32
        )
        return sample

class PooledEmbeddingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        text: np.ndarray,
        image: np.ndarray,
        audio: np.ndarray,
        scores: np.ndarray,
    ):
        self.features = {
            "text": torch.from_numpy(np.asarray(text, dtype=np.float32).copy()),
            "image": torch.from_numpy(np.asarray(image, dtype=np.float32).copy()),
            "audio": torch.from_numpy(np.asarray(audio, dtype=np.float32).copy()),
        }
        self.scores = torch.from_numpy(np.asarray(scores, dtype=np.float32).copy())
        sample_count = self.scores.shape[0]
        for modality, values in self.features.items():
            if values.ndim != 2 or values.shape[0] != sample_count:
                raise ValueError(
                    f"NPZ {modality} features must have shape [samples, dimensions] "
                    "and match the score count."
                )
            if not torch.isfinite(values).all():
                raise ValueError(f"NPZ {modality} features contain NaN or infinite values.")
        if self.scores.ndim != 1 or not torch.isfinite(self.scores).all():
            raise ValueError("NPZ scores must be a finite 1D array.")

    def __len__(self) -> int:
        return self.scores.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "text_features": self.features["text"][index].unsqueeze(0),
            "image_features": self.features["image"][index].unsqueeze(0),
            "audio_features": self.features["audio"][index].unsqueeze(0),
            "score": self.scores[index],
        }

class PreparedCSDClipDataset(Dataset[dict[str, Any]]):
    CACHE_MODALITIES = {"text": "text", "audio": "audio", "image": "vision"}

    def __init__(self, prepared_dir: Path, split: str):
        self.split_dir = Path(prepared_dir) / split
        required = [
            "text.npy", "audio.npy", "vision.npy", "text_mask.npy", "audio_mask.npy",
            "vision_mask.npy", "offsets.npy", "durations.npy", "labels.npy", "scores.npy",
        ]
        missing = [name for name in required if not (self.split_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Prepared {split} split is missing: {', '.join(missing)} in {self.split_dir}"
            )

        self.features = {
            modality: np.load(
                self.split_dir / f"{cache_name}.npy", mmap_mode="r", allow_pickle=False
            )
            for modality, cache_name in self.CACHE_MODALITIES.items()
        }
        self.masks = {
            modality: np.load(
                self.split_dir / f"{cache_name}_mask.npy", mmap_mode="r", allow_pickle=False
            )
            for modality, cache_name in self.CACHE_MODALITIES.items()
        }
        self.offsets = np.asarray(np.load(self.split_dir / "offsets.npy", allow_pickle=False), dtype=np.int64)
        self.durations = np.asarray(np.load(self.split_dir / "durations.npy", allow_pickle=False), dtype=np.float32)
        self.labels = np.asarray(np.load(self.split_dir / "labels.npy", allow_pickle=False), dtype=np.int64)
        self.scores = np.asarray(np.load(self.split_dir / "scores.npy", allow_pickle=False), dtype=np.float32)

        if self.labels.ndim != 1 or not np.isin(self.labels, (0, 1)).all():
            raise ValueError(f"Prepared {split} labels must be binary integers 0 or 1.")
        if self.scores.shape != self.labels.shape or not np.isfinite(self.scores).all():
            raise ValueError(f"Prepared {split} scores must be finite and match labels.")
        if self.offsets.shape != (len(self.labels) + 1,) or self.offsets[0] != 0:
            raise ValueError(f"Prepared {split} offsets must have one boundary per clip plus zero.")
        if self.offsets[-1] != len(self.durations) or np.any(np.diff(self.offsets) <= 0):
            raise ValueError(f"Prepared {split} offsets do not align with windows/durations.")
        if not np.isfinite(self.durations).all() or np.any(self.durations <= 0):
            raise ValueError(f"Prepared {split} window durations must be finite and positive.")
        for modality in MODALITIES:
            values, mask = self.features[modality], self.masks[modality]
            if values.ndim != 3 or mask.shape != values.shape[:2]:
                raise ValueError(f"Prepared {split} {modality} features and mask shapes differ.")
            if values.shape[0] != len(self.durations):
                raise ValueError(f"Prepared {split} {modality} window count differs from durations.")

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
            "score": torch.tensor(self.scores[index], dtype=torch.float32),
        }

def prepared_clip_collate(clips: list[dict[str, Any]]) -> dict[str, Any]:
    features: dict[str, list[torch.Tensor]] = {modality: [] for modality in MODALITIES}
    masks: dict[str, list[torch.Tensor]] = {modality: [] for modality in MODALITIES}
    window_to_clip: list[torch.Tensor] = []
    durations: list[torch.Tensor] = []
    for clip_index, clip in enumerate(clips):
        count = clip["durations"].numel()
        window_to_clip.append(torch.full((count,), clip_index, dtype=torch.long))
        durations.append(clip["durations"])
        for modality in MODALITIES:
            features[modality].append(torch.from_numpy(clip["features"][modality]))
            masks[modality].append(torch.from_numpy(clip["masks"][modality]))
    batch: dict[str, Any] = {
        "labels": torch.stack([clip["label"] for clip in clips]),
        "scores": torch.stack([clip["score"] for clip in clips]),
        "window_to_clip": torch.cat(window_to_clip),
        "durations": torch.cat(durations),
    }
    for modality in MODALITIES:
        batch[f"{modality}_features"] = torch.cat(features[modality], dim=0)
        batch[f"{modality}_mask"] = torch.cat(masks[modality], dim=0)
    return batch

def aggregate_window_logits(
    window_logits: torch.Tensor,
    window_to_clip: torch.Tensor,
    durations: torch.Tensor,
    clip_count: int,
) -> torch.Tensor:
    """Aggregate window predictions into one duration-weighted logit per clip."""
    weights = durations.to(window_logits.dtype).unsqueeze(-1)
    clip_logits = window_logits.new_zeros((clip_count, window_logits.shape[-1]))
    clip_weights = window_logits.new_zeros((clip_count, 1))
    clip_logits.index_add_(0, window_to_clip, window_logits * weights)
    clip_weights.index_add_(0, window_to_clip, weights)
    if torch.any(clip_weights <= 0):
        raise ValueError("Every clip must have positive total window duration.")
    return clip_logits / clip_weights

def load_embedding_splits(
    path: Path,
) -> tuple[dict[str, PooledEmbeddingDataset], dict[str, int], dict[str, int]]:
    required = {
        f"{split}_{name}"
        for split in ("train", "valid", "test")
        for name in ("text", "image", "audio", "scores")
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"Embedding NPZ is missing keys: {', '.join(sorted(missing))}. "
                "Expected train/valid/test_{text,image,audio,scores}."
            )
        splits = {
            split: PooledEmbeddingDataset(
                archive[f"{split}_text"],
                archive[f"{split}_image"],
                archive[f"{split}_audio"],
                archive[f"{split}_scores"],
            )
            for split in ("train", "valid", "test")
        }
        feature_dims = {
            modality: int(archive[f"train_{modality}"].shape[1])
            for modality in MODALITIES
        }
        labels = archive["label_names"].tolist() if "label_names" in archive.files else []
    label_mapping = {
        (value.decode("utf-8") if isinstance(value, bytes) else str(value)): index
        for index, value in enumerate(labels)
    }
    if not label_mapping:
        label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    return splits, feature_dims, label_mapping

def sequence_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for modality in MODALITIES:
        sequences = [sample[f"{modality}_features"] for sample in batch]
        lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True)
        result[f"{modality}_features"] = padded
        result[f"{modality}_mask"] = (
            torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        )
    result["scores"] = torch.stack([sample["score"] for sample in batch])
    return result
