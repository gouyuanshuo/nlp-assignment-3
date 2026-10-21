"""Synthetic Colab handoff checks; no dataset download or GPU training."""

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from mosei.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def notebook_cell(marker):
    notebook = json.loads((ROOT / "notebooks/02_train_colab.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and marker in source:
            return source
    raise AssertionError(f"Missing notebook code cell: {marker}")


def notebook_options(include_hanlin=False, include_ablations=True):
    source = notebook_cell("INCLUDE_HANLIN_MODELS =")
    source = source.replace(
        "INCLUDE_HANLIN_MODELS = False", f"INCLUDE_HANLIN_MODELS = {include_hanlin}"
    )
    source = source.replace("INCLUDE_ABLATIONS = True", f"INCLUDE_ABLATIONS = {include_ablations}")
    namespace = {}
    exec(source, namespace)
    return namespace


@pytest.mark.parametrize("include_ablations", [False, True])
def test_colab_preserves_default_matrix_and_explicitly_opts_into_hanlin(include_ablations):
    expected = ["text", "concat", "transformer"]
    if include_ablations:
        expected += ["no_modality", "no_positions", "mean_pooling"]
    default = notebook_options(include_ablations=include_ablations)
    assert default["CONFIG_PATHS"] == [f"configs/{name}.toml" for name in expected]
    assert default["SEEDS"] == [1111, 1112, 1113]
    enabled = notebook_options(include_hanlin=True, include_ablations=include_ablations)
    assert enabled["CONFIG_PATHS"] == [
        f"configs/{name}.toml" for name in expected + ["almt", "graph_mfn"]
    ]


@pytest.mark.parametrize("name", ["almt", "graph_mfn"])
def test_hanlin_configs_use_the_shared_training_protocol(name):
    config = load_config(ROOT / f"configs/{name}.toml")
    reference = load_config(ROOT / "configs/transformer.toml")
    assert config.name == name
    assert config.model.name == name
    assert config.training == reference.training
    if name == "almt":
        assert (config.model.d_model, config.model.n_heads) == (128, 8)
        assert config.model.almt_token_count == 8
        assert config.model.almt_max_input_length == 256
        assert config.model.almt_fusion_depth == 4
    else:
        assert (config.model.hidden_dim, config.model.graph_mfn_heads) == (128, 4)


def test_colab_times_every_selected_configuration_without_reusing_other_models(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(ROOT)
    namespace = notebook_options(include_hanlin=True)
    commands = []
    durations = {}

    def fake_run(command, check):
        assert check
        commands.append(command)
        config_path = command[command.index("--config") + 1]
        directory = Path(command[command.index("--output") + 1])
        directory.mkdir(parents=True)
        seconds = 7.0 * len(commands)
        durations[config_path] = seconds
        (directory / "run.json").write_text(json.dumps({"device_name": "Synthetic GPU"}))
        (directory / "history.json").write_text(json.dumps([{"seconds": seconds}]))

    namespace.update(
        STORE=tmp_path,
        audit={"fingerprint": "synthetic-fingerprint"},
        cache=tmp_path / "cache",
        json=json,
        Path=Path,
        sys=sys,
        subprocess=SimpleNamespace(run=fake_run),
    )
    exec(notebook_cell("estimated_hours ="), namespace)
    assert [command[command.index("--config") + 1] for command in commands] == namespace[
        "CONFIG_PATHS"
    ]
    assert len({command[command.index("--output") + 1] for command in commands}) == len(commands)
    assert all(command[command.index("--epochs") + 1] == "1" for command in commands)
    assert all("--resume" not in command for command in commands)
    expected = sum(
        durations[path] * load_config(path).training.epochs * len(namespace["SEEDS"]) / 3600
        for path in namespace["CONFIG_PATHS"]
    )
    assert namespace["estimated_hours"] == pytest.approx(expected)


def test_colab_matrix_uses_the_exact_measured_configuration_list(tmp_path):
    namespace = notebook_options(include_hanlin=True)
    commands = []
    namespace.update(
        sys=sys,
        cache=tmp_path / "cache",
        RUNS=tmp_path / "runs",
        subprocess=SimpleNamespace(run=lambda command, check: commands.append(command)),
    )
    exec(notebook_cell("matrix ="), namespace)
    command = commands[0]
    start, end = command.index("--configs") + 1, command.index("--seeds")
    assert command[start:end] == namespace["CONFIG_PATHS"]
    assert command[end + 1 : command.index("--train-only")] == [
        str(seed) for seed in namespace["SEEDS"]
    ]
    assert "--include-ablations" not in command


def test_colab_export_explains_missing_model_runs(tmp_path, monkeypatch):
    google = SimpleNamespace(colab=SimpleNamespace(files=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", google.colab)
    namespace = {"EXPORT_MODEL": "almt", "RUNS": tmp_path, "json": json}
    with pytest.raises(ValueError, match="almt.*CONFIG_PATHS"):
        exec(notebook_cell("candidates ="), namespace)


def test_source_bundle_installs_and_imports_both_models_outside_checkout(tmp_path, monkeypatch):
    staged = tmp_path / "bundle-input"
    staged.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, staged / name)
    for name in ("src/mosei", "configs", "scripts"):
        shutil.copytree(ROOT / name, staged / name)
    for name in (
        "data/tools/video-tools.json",
        "data/tools/feature-manifest.json",
        "data/csd/splits.json",
    ):
        path = staged / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    build = runpy.run_path(str(ROOT / "scripts/build_colab_bundle.py"))["build"]
    monkeypatch.setitem(build.__globals__, "ROOT", staged)
    bundle = tmp_path / "mosei-colab-source.zip"
    build(bundle)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle) as archive:
        assert {
            "src/mosei/almt.py",
            "src/mosei/graph_mfn.py",
            "configs/almt.toml",
            "configs/graph_mfn.toml",
        }.issubset(archive.namelist())
        archive.extractall(extracted)
    for name, digest in json.loads((extracted / "bundle-manifest.json").read_text()).items():
        assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == digest
    installed = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(installed),
            str(extracted),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    probe = """
import sys
from pathlib import Path
installed, extracted, checkout = map(Path, sys.argv[1:4])
# -S avoids editable-install .pth files; dependency directories are added explicitly.
sys.path[:0] = [str(installed), *sys.argv[4:]]
assert str(checkout / 'src') not in sys.path
assert str(extracted / 'src') not in sys.path
import mosei.almt
import mosei.graph_mfn
from mosei.config import load_config
from mosei.models import build_model
for name in ('almt', 'graph_mfn'):
    module = sys.modules['mosei.' + name]
    assert Path(module.__file__).is_relative_to(installed)
    model = build_model(load_config(extracted / 'configs' / (name + '.toml')).model,
                        {'text': 4, 'audio': 2, 'vision': 3})
    assert model is not None
print('Isolated ALMT and Graph_MFN installation/import succeeded')
"""
    dependencies = [
        path for path in sys.path if Path(path).name in {"site-packages", "dist-packages"}
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            probe,
            str(installed),
            str(extracted),
            str(ROOT),
            *dependencies,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "installation/import succeeded" in result.stdout
