from dataclasses import replace

import numpy as np
import pytest
import torch

from mosei.config import ExperimentConfig, TrainingConfig
from mosei.data import load_data, prepare_data
from mosei.models import ModelConfig
from mosei.train import is_better, predict, restore_run, train_run


def test_validation_macro_f1_precedes_cross_entropy():
    assert is_better({"macro_f1": 0.7, "loss": 0.9}, {"macro_f1": 0.6, "loss": 0.1})
    assert is_better({"macro_f1": 0.7, "loss": 0.4}, {"macro_f1": 0.7, "loss": 0.5})
    assert not is_better({"macro_f1": 0.69, "loss": 0.1}, {"macro_f1": 0.7, "loss": 0.9})
    assert not is_better({"macro_f1": 0.7, "loss": 0.5}, {"macro_f1": 0.7, "loss": 0.5})


def test_training_is_reproducible_without_reading_test_labels(tmp_path, feature_file):
    source, sha = feature_file
    prepare_data(source, tmp_path / "data", sha, expected_dims=None)
    data = load_data(tmp_path / "data")
    # Training has no reason to read the test split; removing it catches leakage.
    del data.splits["test"]
    config = ExperimentConfig(
        model=ModelConfig(name="text", hidden_dim=8, dropout=0),
        training=TrainingConfig(
            epochs=3, patience=2, batch_size=2, seed=1111, device="cpu", learning_rate=0.01
        ),
    )
    first = train_run(data, config, tmp_path / "first")
    second = train_run(data, config, tmp_path / "second")
    model_a, meta_a = restore_run(tmp_path / "first", "cpu")
    model_b, _ = restore_run(tmp_path / "second", "cpu")
    a = predict(model_a, data.splits["valid"], 2, "cpu")
    b = predict(model_b, data.splits["valid"], 1, "cpu")
    np.testing.assert_allclose(
        [x["prob_positive"] for x in a], [x["prob_positive"] for x in b], atol=1e-7
    )
    assert first["best_validation"] == second["best_validation"]
    assert meta_a["data_fingerprint"] == data.fingerprint
    assert meta_a["normalization"] == data.normalization
    assert first["trainable_parameters"] > 0
    assert (tmp_path / "first" / "history.json").is_file()
    assert all(torch.isfinite(p).all() for p in model_a.parameters())
    with pytest.raises(FileExistsError):
        train_run(data, config, tmp_path / "first")


@pytest.mark.parametrize("name", ["concat", "transformer"])
def test_each_fusion_model_can_train_and_reload(tmp_path, feature_file, name):
    source, sha = feature_file
    prepare_data(source, tmp_path / "data", sha, expected_dims=None)
    data = load_data(tmp_path / "data")
    config = ExperimentConfig(
        model=ModelConfig(name=name, d_model=8, hidden_dim=8, n_heads=2, ff_dim=16),
        training=TrainingConfig(epochs=1, batch_size=2, device="cpu"),
    )
    result = train_run(data, config, tmp_path / name)
    model, meta = restore_run(tmp_path / name, "cpu")
    predictions = predict(model, data.splits["valid"], 2, "cpu")
    assert result["best_epoch"] == 1
    assert len(predictions) == 2
    assert all(0 <= x["prob_positive"] <= 1 for x in predictions)
    assert meta["config"]["model"]["name"] == name


def test_training_config_rejects_invalid_limits():
    with pytest.raises(ValueError):
        replace(TrainingConfig(), epochs=0)
