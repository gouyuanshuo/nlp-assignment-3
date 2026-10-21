"""Synthetic shared-cache acceptance for the integrated teammate architectures."""

import numpy as np
import pytest
import torch

from mosei.config import ExperimentConfig, TrainingConfig
from mosei.data import MODALITIES, PreparedData
from mosei.evaluate import evaluate_run
from mosei.io import read_json
from mosei.models import ModelConfig
from mosei.train import predict, restore_run, train_run
from mosei.window_data import WindowDataset


def synthetic_windows(split):
    rng = np.random.default_rng(12)
    scores = np.array([-1, -2, -1, -2, 1, 2, 1, 2], dtype=np.float32)
    return WindowDataset(
        features={m: rng.normal(size=(16, 3, width)).astype(np.float32)
                  for m, width in zip(MODALITIES, [4, 2, 3])},
        masks={m: np.ones((16, 3), dtype=bool) for m in MODALITIES},
        labels=(scores > 0).astype(np.int64), scores=scores,
        ids=[f"{split}-clip-{i}" for i in range(8)],
        video_ids=[f"{split}-video-{i // 2}" for i in range(8)],
        raw_text=["synthetic only"] * 8,
        offsets=np.arange(0, 17, 2), durations=np.tile([1, 3], 8).astype(np.float32),
        intervals=np.tile([[0, 1], [1, 4]], (8, 1)).astype(np.float64),
    )


@pytest.mark.parametrize("name", ["almt", "graph_mfn"])
def test_shared_multiwindow_training_subset_reload_and_corruptions(tmp_path, name):
    audit = {
        "feature_dims": dict(zip(MODALITIES, [4, 2, 3])),
        "fingerprint": "synthetic-hanlin", "source_sha256": "synthetic-only",
        "feature_manifest": {"fixture": True},
    }
    # No test split is available during model selection/training.
    data = PreparedData(tmp_path, {s: synthetic_windows(s) for s in ["train", "valid"]},
                        {"source": "full-training-fixture"}, audit)
    config = ExperimentConfig(
        name=name,
        model=ModelConfig(name=name, d_model=8, hidden_dim=8, n_heads=2, dropout=0,
                          almt_token_count=2, almt_fusion_depth=1),
        training=TrainingConfig(epochs=1, batch_size=2, device="cpu", threads=1,
                                train_fraction=0.5),
    )
    run = tmp_path / name
    result = train_run(data, config, run)
    model, meta = restore_run(run, "cpu")
    assert result["best_epoch"] == 1
    assert meta["feature_manifest"] == audit["feature_manifest"]
    assert meta["normalization"] == data.normalization
    assert meta["train_subset"]["clip_count"] == 4
    selected = set(meta["train_subset"]["video_ids"])
    assert len(selected) == 2
    assert sum(v in selected for v in data.splits["train"].video_ids) == 4
    assert len(data.splits["valid"]) == 8
    a = predict(model, data.splits["valid"], 1, "cpu")
    b = predict(model, data.splits["valid"], 3, "cpu")
    np.testing.assert_allclose([r["prob_positive"] for r in a],
                               [r["prob_positive"] for r in b], atol=1e-6)
    assert len(a) == 8  # One prediction per clip, not per window.
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert train_run(data, config, run, resume=True) == result
    # Recover the atomic best export after a simulated post-epoch interruption.
    (run / "run.json").unlink()
    (run / "best.pt").unlink()
    train_run(data, config, run, resume=True)
    restored, restored_meta = restore_run(run, "cpu")
    assert restored_meta == meta
    c = predict(restored, data.splits["valid"], 3, "cpu")
    assert b == c
    data.splits["test"] = synthetic_windows("test")
    evaluation = evaluate_run(data, run, robustness=True, device="cpu")
    assert len(evaluation["metrics"]) == 13
    assert all(metrics["count"] == 8 for metrics in evaluation["metrics"].values())
    assert read_json(run / "data_audit.json")["fingerprint"] == data.fingerprint
