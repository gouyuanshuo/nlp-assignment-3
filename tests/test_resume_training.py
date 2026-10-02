from dataclasses import replace

import pytest
import torch

from mosei.config import ExperimentConfig, TrainingConfig
from mosei.data import load_data, prepare_data
from mosei.models import ModelConfig
from mosei.train import restore_run, train_run


def test_resume_after_interruption_reproduces_uninterrupted_training(
    tmp_path, feature_file, monkeypatch
):
    import mosei.train as trainer

    source, sha = feature_file
    prepare_data(source, tmp_path / "data", sha, expected_dims=None)
    data = load_data(tmp_path / "data")
    config = ExperimentConfig(
        model=ModelConfig(name="text", hidden_dim=8, dropout=0.3),
        training=TrainingConfig(epochs=3, patience=3, batch_size=2, device="cpu"),
    )
    train_run(data, config, tmp_path / "full")
    original_predict = trainer.predict
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated runtime disconnect")
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(trainer, "predict", interrupted)
    with pytest.raises(RuntimeError, match="disconnect"):
        train_run(data, config, tmp_path / "partial")
    monkeypatch.setattr(trainer, "predict", original_predict)
    result = train_run(data, config, tmp_path / "partial", resume=True)
    assert result["epochs_completed"] == 3
    full, meta = restore_run(tmp_path / "full")
    resumed, other = restore_run(tmp_path / "partial")
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
    assert meta["best_validation"] == other["best_validation"]
    full_state = torch.load(tmp_path / "full/last.pt", weights_only=True)
    resumed_state = torch.load(tmp_path / "partial/last.pt", weights_only=True)
    for key, value in full_state["state_dict"].items():
        torch.testing.assert_close(value, resumed_state["state_dict"][key], atol=0, rtol=0)
    assert [h["train_loss"] for h in full_state["history"]] == [
        h["train_loss"] for h in resumed_state["history"]
    ]
    assert train_run(data, config, tmp_path / "partial", resume=True) == result


def test_resume_rejects_changed_configuration(tmp_path, feature_file):
    source, sha = feature_file
    prepare_data(source, tmp_path / "data", sha, expected_dims=None)
    data = load_data(tmp_path / "data")
    config = ExperimentConfig(
        model=ModelConfig(name="text", hidden_dim=8), training=TrainingConfig(epochs=1)
    )
    train_run(data, config, tmp_path / "run")
    changed = replace(config, training=replace(config.training, seed=7))
    with pytest.raises(ValueError, match="configuration|config"):
        train_run(data, changed, tmp_path / "run", resume=True)
