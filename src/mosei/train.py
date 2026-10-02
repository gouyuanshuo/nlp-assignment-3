"""Reproducible training with validation-only selection and reloadable checkpoints."""

import os
import platform
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig, config_from_dict
from .data import LABEL_RULE, FeatureDataset, PreparedData
from .io import write_json
from .metrics import classification_metrics
from .models import build_model


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return requested


def seed_everything(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def is_better(current: dict, best: dict | None) -> bool:
    if best is None:
        return True
    return (current["macro_f1"], -current["loss"]) > (best["macro_f1"], -best["loss"])


def _inputs(batch, device):
    return (
        {m: v.to(device) for m, v in batch["features"].items()},
        {m: v.to(device) for m, v in batch["masks"].items()},
    )


@torch.inference_mode()
def predict(
    model: nn.Module, dataset: FeatureDataset, batch_size: int, device: str, condition=None
) -> list[dict]:
    model.eval()
    records = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        features, masks = batch["features"], batch["masks"]
        if condition is not None:
            from .evaluate import corrupt_batch

            features, masks = corrupt_batch(features, masks, batch["id"], condition)
        features = {m: v.to(device) for m, v in features.items()}
        masks = {m: v.to(device) for m, v in masks.items()}
        logits = model(features, masks)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite prediction logits")
        probabilities = logits.softmax(dim=-1).cpu()
        predictions = logits.argmax(dim=-1).cpu()
        losses = nn.functional.cross_entropy(
            logits, batch["label"].to(device), reduction="none"
        ).cpu()
        for i, clip_id in enumerate(batch["id"]):
            records.append(
                {
                    "id": clip_id,
                    "video_id": batch["video_id"][i],
                    "label": int(batch["label"][i]),
                    "score": float(batch["score"][i]),
                    "prediction": int(predictions[i]),
                    "prob_negative": float(probabilities[i, 0]),
                    "prob_positive": float(probabilities[i, 1]),
                    "loss": float(losses[i]),
                }
            )
    return records


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def train_run(data: PreparedData, config: ExperimentConfig, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = config.training
    seed_everything(settings.seed)
    torch.set_num_threads(settings.threads)
    device = choose_device(settings.device)
    model = build_model(config.model, data.feature_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(
        data.splits["train"],
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=settings.num_workers,
        pin_memory=device == "cuda",
    )
    metadata = {
        "config": asdict(config),
        "feature_dims": data.feature_dims,
        "data_fingerprint": data.fingerprint,
        "source_sha256": data.audit["source_sha256"],
        "label_rule": LABEL_RULE,
        "normalization": data.normalization,
        "git_commit": _git_commit(),
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    write_json(output / "config.json", asdict(config))
    write_json(output / "data_audit.json", data.audit)
    history, best, stale = [], None, 0
    started = time.perf_counter()
    best_epoch = 0
    for epoch in range(1, settings.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, seen = 0.0, 0
        for batch in loader:
            features, masks = _inputs(batch, device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, masks)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), settings.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        validation = classification_metrics(
            predict(model, data.splits["valid"], settings.batch_size, device)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / seen,
                "validation": validation,
                "seconds": time.perf_counter() - epoch_start,
            }
        )
        write_json(output / "history.json", history)
        print(
            f"{config.name} seed={settings.seed} epoch={epoch} train_loss={loss_sum / seen:.4f} "
            f"val_macro_f1={validation['macro_f1']:.4f} val_loss={validation['loss']:.4f}",
            flush=True,
        )
        if is_better(validation, best):
            best, best_epoch, stale = validation, epoch, 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "metadata": {
                    **metadata,
                    "best_epoch": best_epoch,
                    "best_validation": best,
                },
            }
            temp = output / "best.pt.tmp"
            torch.save(checkpoint, temp)
            temp.replace(output / "best.pt")
        else:
            stale += 1
        if stale >= settings.patience:
            break
    result = {
        **metadata,
        "best_epoch": best_epoch,
        "best_validation": best,
        "epochs_completed": len(history),
        "training_seconds": time.perf_counter() - started,
    }
    write_json(output / "run.json", result)
    return result


def restore_run(run: Path, device: str = "cpu"):
    device = choose_device(device)
    checkpoint = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["label_rule"] != LABEL_RULE:
        raise ValueError("Checkpoint label protocol is incompatible")
    config = config_from_dict(metadata["config"])
    model = build_model(config.model, metadata["feature_dims"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, metadata
