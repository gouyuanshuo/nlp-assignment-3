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
from .io import read_json, write_json
from .metrics import classification_metrics
from .models import build_model
from .window_data import clip_logits, collate_clips


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
    for batch in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_clips
    ):
        features, masks = batch["features"], batch["masks"]
        if condition is not None:
            from .evaluate import corrupt_batch

            features, masks = corrupt_batch(features, masks, batch["window_ids"], condition)
        features = {m: v.to(device) for m, v in features.items()}
        masks = {m: v.to(device) for m, v in masks.items()}
        logits = clip_logits(
            model(features, masks), batch["clip_indices"], batch["durations"], len(batch["id"])
        )
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


def _save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train_run(data: PreparedData, config: ExperimentConfig, output: Path, *, resume=False) -> dict:
    output = Path(output)
    state = None
    if output.exists():
        if not resume:
            raise FileExistsError(f"Training directory already exists: {output}")
        if read_json(output / "config.json") != asdict(config):
            raise ValueError("Resume configuration differs from the saved run")
        if read_json(output / "data_audit.json")["fingerprint"] != data.fingerprint:
            raise ValueError("Resume data fingerprint differs from the saved run")
        if (output / "run.json").is_file():
            return read_json(output / "run.json")
        if (output / "last.pt").is_file():
            state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            if "best_checkpoint" not in state:
                raise ValueError(
                    "Resume checkpoint predates atomic best-state recovery; start a new run"
                )
            if (
                state["metadata"]["config"] != asdict(config)
                or state["metadata"]["data_fingerprint"] != data.fingerprint
            ):
                raise ValueError("Resume checkpoint configuration/data differs")
        else:
            print(
                "No completed epoch state was saved; restarting this run from its original seed.",
                flush=True,
            )
    else:
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
        collate_fn=collate_clips,
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
    if "feature_manifest" in data.audit:
        metadata["feature_manifest"] = data.audit["feature_manifest"]
        metadata["window_aggregation"] = "duration-weighted mean logits before loss/softmax"
        metadata["smoke_checkpoint"] = bool(
            data.audit.get("smoke_max_clips_per_split")
            or "smoke" in data.audit.get("folds_source", {}).get("purpose", "").lower()
        )
    write_json(output / "config.json", asdict(config))
    write_json(output / "data_audit.json", data.audit)
    history, best, stale = [], None, 0
    completed, elapsed_before = 0, 0.0
    started = time.perf_counter()
    best_epoch = 0
    best_checkpoint = None
    if state is not None:
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        history, best, best_epoch, stale = (
            state[k] for k in ("history", "best", "best_epoch", "stale")
        )
        completed, elapsed_before = state["epoch"], state["elapsed_seconds"]
        best_checkpoint = state["best_checkpoint"]
        # last.pt commits current and best weights together. Repair an export
        # left behind by an interruption on either side of the epoch commit.
        _save_checkpoint(output / "best.pt", best_checkpoint)
        write_json(output / "history.json", history)
        generator.set_state(state["rng"]["loader"])
        torch.set_rng_state(state["rng"]["torch"])
        if device == "cuda" and state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        random.setstate(state["rng"]["python"])
        numpy_rng = state["rng"]["numpy"]
        np.random.set_state(
            (numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:])
        )
        metadata["resumed_from_epoch"] = completed
        metadata["previous_device_name"] = state["metadata"]["device_name"]
        print(f"Resuming {config.name} after epoch {completed}", flush=True)
    final_epoch = completed if stale >= settings.patience else settings.epochs
    for epoch in range(completed + 1, final_epoch + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, seen = 0.0, 0
        for batch in loader:
            features, masks = _inputs(batch, device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = clip_logits(
                model(features, masks), batch["clip_indices"], batch["durations"], len(labels)
            )
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
        improved = is_better(validation, best)
        if improved:
            best, best_epoch, stale = validation, epoch, 0
            best_checkpoint = {
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "metadata": {
                    **metadata,
                    "best_epoch": best_epoch,
                    "best_validation": best,
                },
            }
        else:
            stale += 1
        numpy_rng = np.random.get_state()
        _save_checkpoint(
            output / "last.pt",
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metadata": metadata,
                "epoch": epoch,
                "history": history,
                "best": best,
                "best_epoch": best_epoch,
                "best_checkpoint": best_checkpoint,
                "stale": stale,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                    "loader": generator.get_state(),
                    "python": random.getstate(),
                    "numpy": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                },
            },
        )
        if improved:
            _save_checkpoint(output / "best.pt", best_checkpoint)
        if stale >= settings.patience:
            break
    result = {
        **metadata,
        "best_epoch": best_epoch,
        "best_validation": best,
        "epochs_completed": len(history),
        "training_seconds": elapsed_before + time.perf_counter() - started,
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
