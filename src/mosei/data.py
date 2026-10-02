"""Audit MMSA's aligned MOSEI release and cache train-normalized, masked features.

The input is a trusted, checksum-pinned pickle. Prepared caches contain only
JSON and NumPy arrays (loaded with allow_pickle=False) and are memory-mapped.
"""

import pickle
import shutil
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .io import fingerprint, read_json, sha256_file, write_json

MODALITIES = ("text", "audio", "vision")
SPLITS = ("train", "valid", "test")
FEATURE_DIMS = {"text": 768, "audio": 74, "vision": 35}
MOSEI_SHA256 = "45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791"
LABEL_RULE = "negative:y<0;positive:y>0;exclude:y==0"
CACHE_VERSION = 1


@dataclass
class FeatureDataset:
    features: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]  # True means VALID, unlike PyTorch's padding mask.
    labels: np.ndarray
    ids: list[str]
    video_ids: list[str]
    raw_text: list[str]
    scores: np.ndarray

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Copies avoid exposing read-only memmaps as writable torch tensors.
        import torch

        return {
            "features": {m: torch.from_numpy(self.features[m][index].copy()) for m in MODALITIES},
            "masks": {m: torch.from_numpy(self.masks[m][index].copy()) for m in MODALITIES},
            "label": int(self.labels[index]),
            "id": self.ids[index],
            "video_id": self.video_ids[index],
            "raw_text": self.raw_text[index],
            "score": float(self.scores[index]),
        }


@dataclass
class PreparedData:
    directory: Path
    splits: dict[str, FeatureDataset]
    normalization: dict
    audit: dict

    @property
    def feature_dims(self):
        return self.audit["feature_dims"]

    @property
    def fingerprint(self):
        return self.audit["fingerprint"]


def _strings(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _mask(raw: dict, modality: str, shape: tuple, shared: np.ndarray | None):
    n, length, _ = shape
    candidates = []
    key = f"{modality}_mask"
    if key in raw:
        candidates.append((np.asarray(raw[key]), key))
    if f"{modality}_lengths" in raw:
        lengths = np.asarray(raw[f"{modality}_lengths"])
        if lengths.shape != (n,) or not np.isfinite(lengths).all():
            raise ValueError(f"Invalid {modality} lengths")
        if (
            np.any(lengths < 0)
            or np.any(lengths > length)
            or np.any(lengths != lengths.astype(int))
        ):
            raise ValueError(f"Invalid {modality} lengths for aligned sequence length {length}")
        candidates.append((np.arange(length)[None, :] < lengths[:, None], f"{modality}_lengths"))
    # BERT attention is another description of text validity. For audio/vision
    # it is only an aligned-release fallback, since explicit stream masks may differ.
    if shared is not None and (modality == "text" or not candidates):
        candidates.append((shared, "text_bert attention mask (word-aligned release)"))
    if not candidates:
        raise ValueError(
            f"Missing documented mask or lengths for {modality}; cannot infer BERT padding"
        )
    for value, _ in candidates:
        if value.shape != (n, length) or not np.isin(value, [0, 1]).all():
            raise ValueError(
                f"Invalid {modality} mask: expected {(n, length)} with only 0/1 values"
            )
        if not np.array_equal(value, candidates[0][0]):
            raise ValueError(f"Conflicting {modality} mask/length metadata")
    return candidates[0][0].astype(bool), "; ".join(source for _, source in candidates)


def _moments(array, mask, indices):
    width = array.shape[-1]
    total = np.zeros(width, dtype=np.float64)
    squared = np.zeros(width, dtype=np.float64)
    count = np.zeros(width, dtype=np.int64)
    for start in range(0, len(indices), 128):
        batch_ids = indices[start : start + 128]
        batch = array[batch_ids].astype(np.float64)
        finite = np.isfinite(batch) & mask[batch_ids, :, None]
        batch = np.where(finite, batch, 0)
        total += batch.sum(axis=(0, 1))
        squared += (batch * batch).sum(axis=(0, 1))
        count += finite.sum(axis=(0, 1))
    if np.any(count == 0):
        raise ValueError("A feature has no finite valid training values")
    mean = total / count
    scale = np.sqrt(np.maximum(squared / count - mean**2, 0))
    scale[scale < 1e-6] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist(), "finite_count": count.tolist()}


def prepare_data(
    source: Path,
    output: Path,
    expected_sha256: str = MOSEI_SHA256,
    expected_dims: dict | None = FEATURE_DIMS,
) -> dict:
    source, output = Path(source), Path(output)
    actual_sha = sha256_file(source)
    if actual_sha != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}")
    if output.exists():
        raise FileExistsError(f"Prepared directory already exists: {output}; use a new directory")
    with source.open("rb") as stream:
        raw = pickle.load(stream)  # Only after verifying the caller's trusted checksum.
    masks, selection, metadata, split_audits = {}, {}, {}, {}
    dims = None
    for split in SPLITS:
        part = raw[split]
        labels = np.asarray(part["regression_labels"]).reshape(-1)
        n = len(labels)
        if n == 0 or not np.isfinite(labels).all() or np.any(np.abs(labels) > 3):
            raise ValueError(f"Invalid {split} sentiment labels; expected finite values in [-3,3]")
        ids = _strings(part["id"])
        if len(ids) != n or len(set(ids)) != n:
            raise ValueError(f"Duplicate or missing clip IDs in {split}")
        if any("$_$" not in clip_id for clip_id in ids):
            raise ValueError("Expected MMSA clip IDs in video_id$_$clip_id format")
        videos = [clip_id.rsplit("$_$", 1)[0] for clip_id in ids]
        texts = _strings(part.get("raw_text", [""] * n))
        if len(texts) != n:
            raise ValueError(f"raw_text count does not match {split} labels")
        arrays = {m: np.asarray(part[m]) for m in MODALITIES}
        for m, array in arrays.items():
            if array.ndim != 3 or array.shape[0] != n or array.shape[1] == 0:
                raise ValueError(f"Invalid {split}/{m} feature shape {array.shape}")
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"Non-numeric {split}/{m} features")
        these_dims = {m: a.shape[2] for m, a in arrays.items()}
        if (expected_dims is not None and these_dims != expected_dims) or (
            dims and these_dims != dims
        ):
            raise ValueError(
                f"Feature dimensions {these_dims} do not match expected {expected_dims or dims}"
            )
        dims = these_dims
        lengths = {a.shape[1] for a in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("This loader expects aligned sequences of equal length")
        shared = None
        if "text_bert" in part:
            bert = np.asarray(part["text_bert"])
            if bert.shape != (n, 3, arrays["text"].shape[1]):
                raise ValueError(f"Invalid text_bert mask layout in {split}: {bert.shape}")
            shared = bert[:, 1, :]
        masks[split], provenance = {}, {}
        for m in MODALITIES:
            masks[split][m], provenance[m] = _mask(part, m, arrays[m].shape, shared)
        keep = np.flatnonzero(labels != 0)
        if set((labels[keep] > 0).astype(int).tolist()) != {0, 1}:
            raise ValueError(
                f"{split} must contain both negative and positive labels after filtering"
            )
        selection[split] = keep
        metadata[split] = {"ids": ids, "video_ids": videos, "raw_text": texts, "scores": labels}
        split_audits[split] = {
            "original_count": n,
            "eligible_count": len(keep),
            "excluded_neutral": int((labels == 0).sum()),
            "class_counts": {
                "negative": int((labels < 0).sum()),
                "positive": int((labels > 0).sum()),
            },
            "shapes": {m: list(a.shape) for m, a in arrays.items()},
            "mask_sources": provenance,
            "nonfinite_valid": {},
        }
    for left, right in combinations(SPLITS, 2):
        overlap = set(metadata[left]["video_ids"]) & set(metadata[right]["video_ids"])
        if overlap:
            raise ValueError(f"Source video overlap between {left}/{right}: {sorted(overlap)[:5]}")
    normalization = {
        m: _moments(raw["train"][m], masks["train"][m], selection["train"]) for m in MODALITIES
    }
    audit = {
        "cache_version": CACHE_VERSION,
        "dataset": "CMU-MOSEI",
        "language": "English",
        "source_sha256": actual_sha,
        "official_release_checksum": actual_sha == MOSEI_SHA256,
        "label_rule": LABEL_RULE,
        "feature_dims": dims,
        "splits": split_audits,
        "video_overlap": [],
        "speaker_metadata": "not supplied in MMSA feature archive",
        "normalization": "finite valid non-neutral training tokens only; nonfinite imputed with training mean",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        for split in SPLITS:
            keep = selection[split]
            folder = staging / split
            folder.mkdir()
            meta = metadata[split]
            write_json(
                folder / "clips.json",
                {k: [meta[k][i] for i in keep] for k in ("ids", "video_ids", "raw_text")},
            )
            np.save(
                folder / "scores.npy", meta["scores"][keep].astype(np.float32), allow_pickle=False
            )
            np.save(
                folder / "labels.npy",
                (meta["scores"][keep] > 0).astype(np.int64),
                allow_pickle=False,
            )
            for m in MODALITIES:
                array = raw[split][m]
                mask = masks[split][m][keep]
                np.save(folder / f"{m}_mask.npy", mask, allow_pickle=False)
                normalized = np.lib.format.open_memmap(
                    folder / f"{m}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(keep), *array.shape[1:]),
                )
                mean = np.array(normalization[m]["mean"], dtype=np.float32)
                scale = np.array(normalization[m]["scale"], dtype=np.float32)
                bad_count = 0
                for start in range(0, len(keep), 128):
                    stop = start + 128
                    block = array[keep[start:stop]].astype(np.float32)
                    valid = mask[start:stop, :, None]
                    bad_count += int((~np.isfinite(block) & valid).sum())
                    block = np.where(np.isfinite(block), block, mean)
                    normalized[start:stop] = np.where(valid, (block - mean) / scale, 0)
                normalized.flush()
                del normalized
                split_audits[split]["nonfinite_valid"][m] = bad_count
        audit["fingerprint"] = fingerprint({"audit": audit, "normalization": normalization})
        write_json(staging / "normalization.json", normalization)
        write_json(staging / "audit.json", audit)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return audit


def load_data(directory: Path) -> PreparedData:
    directory = Path(directory)
    audit = read_json(directory / "audit.json")
    normalization = read_json(directory / "normalization.json")
    if audit["cache_version"] not in {CACHE_VERSION, 2} or audit["label_rule"] != LABEL_RULE:
        raise ValueError("Incompatible prepared data protocol")
    check = dict(audit)
    saved_fingerprint = check.pop("fingerprint")
    if fingerprint({"audit": check, "normalization": normalization}) != saved_fingerprint:
        raise ValueError("Prepared metadata fingerprint mismatch")
    splits = {}
    for split in SPLITS:
        folder = directory / split
        clips = read_json(folder / "clips.json")

        def array(name):
            return np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)

        dataset_class, extra = FeatureDataset, {}
        if audit["cache_version"] == 2:
            from .window_data import WindowDataset

            dataset_class = WindowDataset
            extra = {name: array(name) for name in ("offsets", "durations", "intervals")}
        splits[split] = dataset_class(
            features={m: array(m) for m in MODALITIES},
            masks={m: array(m + "_mask") for m in MODALITIES},
            labels=array("labels"),
            scores=array("scores"),
            **clips,
            **extra,
        )
    return PreparedData(directory, splits, normalization, audit)
