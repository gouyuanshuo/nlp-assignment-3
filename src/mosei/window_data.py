"""Variable windows remain a single supervised example per source clip."""

from dataclasses import dataclass

import numpy as np
import torch

from .data import MODALITIES, FeatureDataset


@dataclass
class WindowDataset(FeatureDataset):
    offsets: np.ndarray
    durations: np.ndarray
    intervals: np.ndarray

    def __post_init__(self):
        n = len(self.durations)
        if (
            self.offsets.shape != (len(self) + 1,)
            or self.offsets.dtype.kind not in "iu"
            or self.offsets[0] != 0
            or self.offsets[-1] != n
            or np.any(np.diff(self.offsets) <= 0)
        ):
            raise ValueError("Clip offsets must partition every window into nonempty clips")
        if not np.isfinite(self.durations).all() or np.any(self.durations <= 0):
            raise ValueError("Window durations must be finite and positive")
        if self.intervals.shape != (n, 2) or not np.isfinite(self.intervals).all():
            raise ValueError("Invalid window intervals")
        for m in MODALITIES:
            if self.features[m].ndim != 3 or self.features[m].shape[0] != n:
                raise ValueError("Feature arrays must contain every window")
            if self.masks[m].shape != self.features[m].shape[:2]:
                raise ValueError("Window feature masks have the wrong shape")
        if any(len(x) != len(self) for x in (self.ids, self.video_ids, self.raw_text, self.scores)):
            raise ValueError("Clip metadata length differs from labels")

    def __getitem__(self, index):
        start, stop = self.offsets[index : index + 2]
        return {
            "features": {
                m: torch.from_numpy(self.features[m][start:stop].copy()) for m in MODALITIES
            },
            "masks": {m: torch.from_numpy(self.masks[m][start:stop].copy()) for m in MODALITIES},
            "durations": torch.from_numpy(self.durations[start:stop].copy()),
            "label": int(self.labels[index]),
            "id": self.ids[index],
            "video_id": self.video_ids[index],
            "raw_text": self.raw_text[index],
            "score": float(self.scores[index]),
        }


def collate_clips(items):
    """Flatten windows, preserving clip ownership, metadata and stable corruption IDs."""
    features, masks = {m: [] for m in MODALITIES}, {m: [] for m in MODALITIES}
    indices, durations, window_ids = [], [], []
    for i, item in enumerate(items):
        multi = item["features"]["text"].ndim == 3
        count = len(item["features"]["text"]) if multi else 1
        for m in MODALITIES:
            features[m].append(item["features"][m] if multi else item["features"][m][None])
            masks[m].append(item["masks"][m] if multi else item["masks"][m][None])
        indices.extend([i] * count)
        durations.append(item["durations"] if multi else torch.ones(1))
        window_ids.extend(
            [f"{item['id']}::window:{j}" for j in range(count)] if multi else [item["id"]]
        )
    return {
        "features": {m: torch.cat(v) for m, v in features.items()},
        "masks": {m: torch.cat(v) for m, v in masks.items()},
        "clip_indices": torch.tensor(indices, dtype=torch.long),
        "durations": torch.cat(durations).float(),
        "window_ids": window_ids,
        "label": torch.tensor([i["label"] for i in items], dtype=torch.long),
        "score": torch.tensor([i["score"] for i in items], dtype=torch.float64),
        **{k: [i[k] for i in items] for k in ("id", "video_id", "raw_text")},
    }


def clip_logits(window_logits, clip_indices, durations, clip_count):
    """Aggregate BEFORE loss/softmax; dense reduction is deterministic on CUDA."""
    indices = clip_indices.to(device=window_logits.device, dtype=torch.long)
    weights = durations.to(window_logits)
    if (
        weights.shape != (len(window_logits),)
        or not torch.isfinite(weights).all()
        or torch.any(weights <= 0)
    ):
        raise ValueError("Window durations must be finite, positive and match logits")
    if indices.shape != weights.shape or torch.any(indices < 0) or torch.any(indices >= clip_count):
        raise ValueError("Window ownership indices are invalid")
    membership = torch.nn.functional.one_hot(indices, clip_count).T.to(window_logits)
    weighted = membership * weights[None]
    totals = weighted.sum(dim=1, keepdim=True)
    if torch.any(totals == 0):
        raise ValueError("Each clip must own at least one window")
    return (weighted / totals) @ window_logits
