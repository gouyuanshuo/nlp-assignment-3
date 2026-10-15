from dataclasses import replace

import pytest
import torch

from mosei.config import ExperimentConfig, TrainingConfig
from mosei.data import load_data, prepare_data
from mosei.io import read_json, write_json
from mosei.models import ModelConfig
from mosei.train import restore_run, train_run


@pytest.mark.parametrize("legacy_files", [(), ("config",), ("checkpoint",), ("config", "checkpoint")])
def test_resume_after_interruption_reproduces_uninterrupted_training(
    tmp_path, feature_file, monkeypatch, legacy_files
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
    # Old runs omit fields introduced by the modality/training-size experiments.
    if "config" in legacy_files:
        saved = read_json(tmp_path / "partial/config.json")
        saved["model"].pop("modalities")
        saved["training"].pop("train_fraction")
        write_json(tmp_path / "partial/config.json", saved)
    if "checkpoint" in legacy_files:
        state = torch.load(tmp_path / "partial/last.pt", weights_only=True)
        for metadata in (state["metadata"], state["best_checkpoint"]["metadata"]):
            metadata["config"]["model"].pop("modalities", None)
            metadata["config"]["training"].pop("train_fraction", None)
        torch.save(state, tmp_path / "partial/last.pt")
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


def test_best_checkpoint_matches_committed_epoch_after_retry_drift(
    tmp_path, feature_file, monkeypatch
):
    import mosei.train as trainer

    source, sha = feature_file
    prepare_data(source, tmp_path / "data", sha, expected_dims=None)
    data = load_data(tmp_path / "data")
    config = ExperimentConfig(
        model=ModelConfig(name="text", hidden_dim=8),
        training=TrainingConfig(epochs=2, patience=3, batch_size=2, device="cpu"),
    )
    losses = iter([0.2, 0.1999999, 0.2000001])
    monkeypatch.setattr(
        trainer, "classification_metrics", lambda _: {"macro_f1": 0.5, "loss": next(losses)}
    )
    save = trainer._save_checkpoint

    def interrupted(path, value):
        if path.name == "last.pt" and value["epoch"] == 2:
            raise RuntimeError("disconnect before epoch commit")
        return save(path, value)

    monkeypatch.setattr(trainer, "_save_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="disconnect"):
        train_run(data, config, tmp_path / "run")
    monkeypatch.setattr(trainer, "_save_checkpoint", save)
    result = train_run(data, config, tmp_path / "run", resume=True)
    _, checkpoint = restore_run(tmp_path / "run")
    assert result["best_epoch"] == checkpoint["best_epoch"] == 1
    assert result["best_validation"] == checkpoint["best_validation"]


@pytest.mark.parametrize("changed_field", [None, "train_fraction", "modalities"])
def test_completed_legacy_run_accepts_defaults_but_rejects_changed_settings(
    tmp_path, feature_file, changed_field
):
    source, sha = feature_file
    prepare_data(source, tmp_path / "data", sha, expected_dims=None)
    data = load_data(tmp_path / "data")
    config = ExperimentConfig(
        model=ModelConfig(name="concat", d_model=8, hidden_dim=8, n_heads=2),
        training=TrainingConfig(epochs=1, device="cpu"),
    )
    directory = tmp_path / "run"
    result = train_run(data, config, directory)
    for filename in ("config.json", "run.json"):
        saved = read_json(directory / filename)
        saved_config = saved if filename == "config.json" else saved["config"]
        saved_config["model"].pop("modalities")
        saved_config["training"].pop("train_fraction")
        write_json(directory / filename, saved)
    if changed_field == "train_fraction":
        config = replace(config, training=replace(config.training, train_fraction=0.5))
    elif changed_field == "modalities":
        config = replace(config, model=replace(config.model, modalities=["text", "audio"]))
    if changed_field:
        with pytest.raises(ValueError, match="configuration"):
            train_run(data, config, directory, resume=True)
    else:
        resumed = train_run(data, config, directory, resume=True)
        assert resumed["best_validation"] == result["best_validation"]
        assert resumed["epochs_completed"] == 1
