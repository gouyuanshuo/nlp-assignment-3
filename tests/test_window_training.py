"""Clip-level loss/evaluation for variable window counts; synthetic fixtures only."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mosei.config import ExperimentConfig, TrainingConfig
from mosei.data import MODALITIES, PreparedData
from mosei.models import ModelConfig
from mosei.train import predict, restore_run, train_run
from mosei.window_data import WindowDataset, clip_logits, collate_clips


def dataset():
    values = np.array([[[0, 2]], [[4, 0]], [[2, 2]]], dtype=np.float32)
    return WindowDataset(
        features={m: values.copy() for m in MODALITIES},
        masks={m: np.ones((3, 1), dtype=bool) for m in MODALITIES},
        labels=np.array([0, 1]),
        ids=["a", "b"],
        video_ids=["video-a", "video-b"],
        raw_text=["one two", "three"],
        scores=np.array([-1, 1], np.float32),
        offsets=np.array([0, 2, 3]),
        durations=np.array([1, 3, 1], np.float32),
        intervals=np.array([[0, 1], [1, 4], [0, 1]], np.float64),
    )


def test_weighted_logits_and_gradient_use_all_windows_once():
    batch = collate_clips([dataset()[0], dataset()[1]])
    logits = batch["features"]["text"][:, 0].requires_grad_()
    result = clip_logits(logits, batch["clip_indices"], batch["durations"], 2)
    torch.testing.assert_close(result, torch.tensor([[3.0, 0.5], [2.0, 2.0]]))
    result.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[0.25, 0.25], [0.75, 0.75], [1.0, 1.0]]))
    assert batch["id"] == ["a", "b"]
    assert batch["window_ids"] == ["a::window:0", "a::window:1", "b::window:0"]
    assert batch["label"].tolist() == [0, 1]


@pytest.mark.parametrize("durations", [[0, 3, 1], [-1, 3, 1], [float("nan"), 3, 1]])
def test_invalid_window_weights_fail(durations):
    with pytest.raises(ValueError, match="duration"):
        clip_logits(torch.zeros(3, 2), torch.tensor([0, 0, 1]), torch.tensor(durations), 2)


def test_prediction_is_one_record_per_clip_and_invariant_to_batch_size():
    class Fixed(torch.nn.Module):
        def forward(self, features, masks):
            return features["text"][:, 0]

    first = predict(Fixed(), dataset(), 1, "cpu")
    second = predict(Fixed(), dataset(), 2, "cpu")
    assert first == second
    assert [r["id"] for r in first] == ["a", "b"]
    assert first[0]["prob_positive"] == pytest.approx(torch.tensor([3.0, 0.5]).softmax(0)[1].item())
    assert first[1]["prob_positive"] == 0.5


@pytest.mark.parametrize("name", ["text", "concat", "transformer"])
def test_multiwindow_train_reload_preserves_manifest(tmp_path, name):
    audit = {
        "feature_dims": {m: 2 for m in MODALITIES},
        "fingerprint": "synthetic",
        "source_sha256": "synthetic",
        "feature_manifest": {"version": 123},
    }
    data = PreparedData(tmp_path, {"train": dataset(), "valid": dataset()}, {}, audit)
    config = ExperimentConfig(
        model=ModelConfig(name=name, d_model=8, hidden_dim=8, n_heads=2, ff_dim=16),
        training=TrainingConfig(epochs=1, batch_size=2, device="cpu"),
    )
    train_run(data, config, tmp_path / name)
    model, meta = restore_run(tmp_path / name)
    assert meta["feature_manifest"] == audit["feature_manifest"]
    assert len(predict(model, data.splits["valid"], 2, "cpu")) == 2


def test_invalid_clip_offsets_cannot_drop_windows():
    with pytest.raises(ValueError, match="offsets"):
        replace(dataset(), offsets=np.array([0, 1, 2]))
