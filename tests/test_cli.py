import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


def cli(*args, success=True):
    result = subprocess.run(
        [sys.executable, "-m", "mosei", *map(str, args)], capture_output=True, text=True
    )
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
    return result


def test_cli_end_to_end_on_explicit_synthetic_fixture(tmp_path, raw_features):
    for split in raw_features.values():
        for m, width in {"text": 768, "audio": 74, "vision": 35}.items():
            split[m] = np.repeat(split[m][..., :1], width, axis=-1)
    source = tmp_path / "synthetic.pkl"
    source.write_bytes(pickle.dumps(raw_features))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = tmp_path / "prepared"
    cli("prepare", "--source", source, "--output", data, "--sha256", digest)
    runs = []
    for model in ["text", "transformer"]:
        config = tmp_path / f"{model}.toml"
        config.write_text(f'''name = "{model}"
[model]
name = "{model}"
d_model = 8
n_heads = 2
ff_dim = 16
hidden_dim = 8
[training]
epochs = 1
batch_size = 2
device = "cpu"
''')
        run = tmp_path / model
        cli("train", "--data", data, "--config", config, "--output", run)
        cli("evaluate", "--data", data, "--run", run, "--device", "cpu", "--robustness")
        evaluation = json.loads((run / "evaluation/test/metrics.json").read_text())
        assert len(evaluation["metrics"]) == 13
        assert evaluation["metrics"]["clean"]["count"] == 2
        runs.append(run)
    report = tmp_path / "report"
    cli("report", "--runs", *runs, "--output", report, "--bootstrap-samples", "20")
    assert (report / "summary.csv").is_file()
    assert (report / "transformer_confusion.png").is_file()
    demo = cli(
        "demo",
        "--data",
        data,
        "--run",
        runs[1],
        "--baseline",
        runs[0],
        "--index",
        "0",
        "--device",
        "cpu",
    )
    result = json.loads(demo.stdout)
    assert result["id"] == "testvideo$_$0"
    assert result["annotation"] == "negative"
    assert result["transcript"] == "negative synthetic example"
    assert "text_baseline" in result
    matrix = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "--data",
            str(data),
            "--configs",
            str(config),
            "--seeds",
            "1111",
            "--output",
            str(tmp_path / "matrix"),
            "--train-only",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert matrix.returncode == 0, matrix.stdout + matrix.stderr
    manifest = json.loads((tmp_path / "matrix" / "manifest.json").read_text())
    assert manifest["status"] == "trained"
    assert not (tmp_path / "matrix" / "transformer-seed1111" / "evaluation").exists()
    repeated = cli("train", "--data", data, "--config", config, "--output", runs[1], success=False)
    assert "exists" in repeated.stderr


def test_cli_error_is_actionable_for_missing_dataset(tmp_path):
    result = cli(
        "train",
        "--data",
        tmp_path / "missing",
        "--config",
        Path("configs/text.toml"),
        "--output",
        tmp_path / "out",
        success=False,
    )
    assert "error:" in result.stderr
