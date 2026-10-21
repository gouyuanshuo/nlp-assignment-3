"""Shared experiment matrix contracts; synthetic fixtures, not MOSEI results."""

import runpy
import sys
from dataclasses import asdict

import pytest

from mosei.data import prepare_data
from mosei.io import read_json, write_json


def test_fraction_configs_are_explicit_and_keep_default_names(tmp_path):
    module = runpy.run_path("scripts/run_experiments.py")
    assert "matrix_configs" in module, "The shared runner must expose fraction-aware configuration"
    build = module["matrix_configs"]
    path = tmp_path / "tiny.toml"
    path.write_text('name="tiny"\n[model]\nname="text"\n')
    base = build([path], device="cpu", epochs=1)
    assert [c.name for c in base] == ["tiny"]
    sampled = build([path], device="cpu", epochs=1, train_fractions=[0.5, 1.0])
    assert [c.name for c in sampled] == ["tiny-frac50", "tiny-frac100"]
    assert [c.training.train_fraction for c in sampled] == [0.5, 1.0]
    assert all(c.training.epochs == 1 and c.training.device == "cpu" for c in sampled)


@pytest.mark.parametrize("fractions", [[], [0], [1.1], [0.5, 0.5], [float("nan")]])
def test_invalid_fraction_matrix_fails_before_training(tmp_path, fractions):
    module = runpy.run_path("scripts/run_experiments.py")
    assert "matrix_configs" in module
    path = tmp_path / "tiny.toml"
    path.write_text('name="tiny"\n[model]\nname="text"\n')
    with pytest.raises(ValueError, match="fraction"):
        module["matrix_configs"]([path], device="cpu", train_fractions=fractions)


def test_legacy_matrix_resume_applies_model_defaults(tmp_path, feature_file, monkeypatch):
    source, sha = feature_file
    prepared = tmp_path / "data"
    prepare_data(source, prepared, sha, expected_dims=None)
    config = tmp_path / "tiny.toml"
    config.write_text('name="tiny"\n[model]\nname="text"\nhidden_dim=8\n'
                      '[training]\nepochs=1\nbatch_size=2\ndevice="cpu"\n')
    output = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["runner", "--data", str(prepared), "--configs", str(config),
                                     "--output", str(output), "--seeds", "1111", "--device", "cpu",
                                     "--train-only", "--resume"])
    main = runpy.run_path("scripts/run_experiments.py")["main"]
    main()
    manifest = read_json(output / "manifest.json")
    # Legacy manifests may omit fields whose dataclass defaults are unchanged.
    manifest["configs"][0]["model"].pop("positions")
    write_json(output / "manifest.json", manifest)
    main()
    assert read_json(output / "manifest.json")["status"] == "trained"


def test_hanlin_component_ablations_use_shared_configs():
    module = runpy.run_path("scripts/run_experiments.py")
    assert "matrix_configs" in module
    from pathlib import Path

    configs = module["matrix_configs"](
        [Path("configs/almt.toml"), Path("configs/graph_mfn.toml")],
        device="cpu", include_hanlin_ablations=True,
    )
    assert len(configs) == 10
    assert len({c.name for c in configs}) == 10
    for base in configs[::5]:
        related = [c for c in configs if c.model.name == base.model.name and c != base]
        assert len(related) == 4
        for ablation in related:
            changed = [k for k, v in asdict(base.model).items() if asdict(ablation.model)[k] != v]
            assert len(changed) == 1


def test_report_accepts_mixed_legacy_and_current_seed_configs(tmp_path, feature_file, monkeypatch):
    source, sha = feature_file
    prepared = tmp_path / "data"
    prepare_data(source, prepared, sha, expected_dims=None)
    config = tmp_path / "tiny.toml"
    config.write_text('name="tiny"\n[model]\nname="text"\nhidden_dim=8\n'
                      '[training]\nepochs=1\nbatch_size=2\ndevice="cpu"\n')
    output = tmp_path / "matrix"
    command = ["runner", "--data", str(prepared), "--configs", str(config),
               "--output", str(output), "--seeds", "1111", "1112", "--device", "cpu",
               "--resume", "--bootstrap-samples", "5"]
    monkeypatch.setattr(sys, "argv", command + ["--train-only"])
    main = runpy.run_path("scripts/run_experiments.py")["main"]
    main()
    old_config_path = output / "tiny-seed1111" / "config.json"
    old_config = read_json(old_config_path)
    for field in list(old_config["model"]):
        if field.startswith(("almt_", "graph_")):
            old_config["model"].pop(field)
    write_json(old_config_path, old_config)
    monkeypatch.setattr(sys, "argv", command)
    main()
    assert read_json(output / "manifest.json")["status"] == "evaluated"
    assert (output / "report" / "complete.json").is_file()

    # Canonicalization must not hide a genuinely different model setting.
    from mosei.report import summarize_runs

    old_config["model"]["hidden_dim"] = 16
    write_json(old_config_path, old_config)
    with pytest.raises(ValueError, match="settings must match"):
        summarize_runs([output / f"tiny-seed{seed}" for seed in [1111, 1112]],
                       output / "bad-report", 5)
