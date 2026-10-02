"""Matched feature corruption and held-out evaluation."""

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data import MODALITIES, PreparedData
from .io import write_json
from .metrics import classification_metrics
from .train import choose_device, predict, restore_run


@dataclass(frozen=True)
class Condition:
    kind: str = "clean"
    modality: str | None = None
    std: float = 0.0
    seed: int = 2026

    def __post_init__(self):
        if self.kind not in {"clean", "missing", "noise"}:
            raise ValueError("Unknown corruption kind")
        if self.kind != "clean" and self.modality not in MODALITIES:
            raise ValueError("A modality is required for corruption")
        if self.kind == "noise" and (not np.isfinite(self.std) or self.std <= 0):
            raise ValueError("Noise standard deviation must be finite and positive")

    @property
    def key(self):
        if self.kind == "clean":
            return "clean"
        suffix = f"_{self.std:g}" if self.kind == "noise" else ""
        return f"{self.kind}_{self.modality}{suffix}"


def protocol_conditions(robustness: bool, noise_seed: int = 2026) -> list[Condition]:
    conditions = [Condition()]
    if robustness:
        for m in MODALITIES:
            conditions.append(Condition("missing", m, seed=noise_seed))
            conditions.extend(Condition("noise", m, sigma, noise_seed) for sigma in (0.1, 0.3, 0.5))
    return conditions


def corrupt_batch(features, masks, ids: list[str], condition: Condition):
    result, valid = dict(features), dict(masks)
    if condition.kind == "clean":
        return result, valid
    modality = condition.modality
    if condition.kind == "missing":
        result[modality] = torch.zeros_like(features[modality])
        valid[modality] = torch.zeros_like(masks[modality])
    else:
        noise = []
        # Hash IDs instead of consuming a global RNG: invariant to model, run seed,
        # ordering and batch size. Reuse the same standard-normal field across SDs.
        for clip_id in ids:
            key = f"{condition.seed}\0{modality}\0{clip_id}".encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            rng = np.random.Generator(np.random.PCG64(seed))
            noise.append(rng.standard_normal(features[modality].shape[1:], dtype=np.float32))
        noise = torch.from_numpy(np.stack(noise)).to(features[modality])
        result[modality] = features[modality] + noise * condition.std * masks[modality][..., None]
    return result, valid


def write_predictions(path: Path, records: list[dict]):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_predictions(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("label", "prediction"):
            row[field] = int(row[field])
        for field in ("score", "prob_negative", "prob_positive", "loss"):
            row[field] = float(row[field])
    return rows


def evaluate_run(
    data: PreparedData,
    run: Path,
    split: str = "test",
    robustness: bool = True,
    device: str = "auto",
    batch_size: int | None = None,
    noise_seed: int = 2026,
) -> dict:
    if split not in {"valid", "test"}:
        raise ValueError("Evaluation split must be valid or test")
    run = Path(run)
    device = choose_device(device)
    model, metadata = restore_run(run, device)
    if (
        metadata["data_fingerprint"] != data.fingerprint
        or metadata["normalization"] != data.normalization
    ):
        raise ValueError("Data/preprocessing differs from the training checkpoint")
    output = run / "evaluation" / split
    output.mkdir(parents=True, exist_ok=False)
    batch_size = batch_size or metadata["config"]["training"]["batch_size"]
    conditions = protocol_conditions(robustness, noise_seed)
    result = {
        "split": split,
        "data_fingerprint": data.fingerprint,
        "label_rule": metadata["label_rule"],
        "experiment": metadata["config"]["name"],
        "seed": metadata["config"]["training"]["seed"],
        "conditions": [asdict(c) for c in conditions],
        "noise_generator": "NumPy PCG64, SHA256(seed, modality, clip ID)",
        "metrics": {},
    }
    clean_f1 = None
    for condition in conditions:
        records = predict(model, data.splits[split], batch_size, device, condition)
        metrics = classification_metrics(records)
        if clean_f1 is None:
            clean_f1 = metrics["macro_f1"]
        metrics["macro_f1_drop"] = clean_f1 - metrics["macro_f1"]
        result["metrics"][condition.key] = metrics
        write_predictions(output / f"{condition.key}.csv", records)
        print(
            f"{result['experiment']} seed={result['seed']} {split}/{condition.key}: macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
    write_json(output / "metrics.json", result)
    return result
