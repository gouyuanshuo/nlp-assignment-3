"""Synthetic Colab handoff checks; no dataset download or GPU training."""

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, replace
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


@pytest.fixture
def export_matrix(tmp_path, monkeypatch):
    downloads = []
    google = SimpleNamespace(
        colab=SimpleNamespace(files=SimpleNamespace(download=downloads.append))
    )
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", google.colab)
    runs, store = tmp_path / "runs", tmp_path / "store"
    runs.mkdir()
    store.mkdir()
    paths = [str(ROOT / f"configs/{name}.toml") for name in ("almt", "text")]
    configs = []
    for path in paths:
        config = load_config(path)
        configs.append(replace(config, training=replace(config.training, device="cuda")))
    manifest = {
        "status": "trained",
        "configs": [asdict(config) for config in configs],
        "seeds": [1111, 1112],
        "data_fingerprint": "synthetic-fingerprint",
    }
    (runs / "manifest.json").write_text(json.dumps(manifest))
    for config in configs:
        for seed in manifest["seeds"]:
            directory = runs / f"{config.name}-seed{seed}"
            directory.mkdir()
            saved_config = asdict(replace(config, training=replace(config.training, seed=seed)))
            result = {
                "config": saved_config,
                "data_fingerprint": manifest["data_fingerprint"],
                "epochs_completed": 1,
                "best_epoch": 1,
                "best_validation": {"macro_f1": 0.5, "loss": 0.8},
            }
            (directory / "run.json").write_text(json.dumps(result))
            (directory / "config.json").write_text(json.dumps(saved_config))
            (directory / "history.json").write_text(json.dumps([{"epoch": 1}]))
            (directory / "data_audit.json").write_text(
                json.dumps({"fingerprint": manifest["data_fingerprint"]})
            )
            (directory / "best.pt").write_bytes(b"synthetic checkpoint fixture")
    namespace = {
        "EXPORT_MODEL": "almt",
        "RUNS": runs,
        "STORE": store,
        "CONFIG_PATHS": paths,
        "SEEDS": manifest["seeds"].copy(),
        "audit": {"fingerprint": manifest["data_fingerprint"]},
        "json": json,
        "Path": lambda value: (
            tmp_path / "checkpoint.zip"
            if str(value) == "/content/mosei-video-checkpoint.zip"
            else Path(value)
        ),
        "zipfile": zipfile,
        "shutil": shutil,
        "downloads": downloads,
    }
    return namespace


def rewrite_json(path, update):
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("status", [None, "training"])
def test_colab_export_requires_completed_frozen_matrix(export_matrix, status):
    namespace = export_matrix
    manifest = namespace["RUNS"] / "manifest.json"
    if status is None:
        manifest.unlink()
    else:
        rewrite_json(manifest, lambda value: value.update(status=status))
    with pytest.raises(ValueError, match="[Mm]atrix.*(manifest|training|complete)"):
        exec(notebook_cell("candidates ="), namespace)
    assert namespace["downloads"] == []


def test_colab_export_rejects_model_outside_frozen_matrix(export_matrix):
    namespace = export_matrix
    namespace["EXPORT_MODEL"] = "graph_mfn"
    with pytest.raises(ValueError, match="graph_mfn.*frozen"):
        exec(notebook_cell("candidates ="), namespace)


@pytest.mark.parametrize("model", ["almt", "text"])
@pytest.mark.parametrize(
    "missing_file", ["best.pt", "run.json", "config.json", "history.json", "data_audit.json"]
)
def test_colab_export_rejects_any_incomplete_frozen_seed(export_matrix, model, missing_file):
    namespace = export_matrix
    (namespace["RUNS"] / f"{model}-seed1112" / missing_file).unlink()
    with pytest.raises(ValueError, match=f"{model}-seed1112.*{missing_file}"):
        exec(notebook_cell("candidates ="), namespace)
    assert namespace["downloads"] == []


@pytest.mark.parametrize("changed", ["seeds", "configs", "config_contents", "data"])
def test_colab_export_rejects_ui_changes_since_matrix_freeze(export_matrix, changed, tmp_path):
    namespace = export_matrix
    if changed == "seeds":
        namespace["SEEDS"] = [1111]
    elif changed == "configs":
        namespace["CONFIG_PATHS"] = namespace["CONFIG_PATHS"][:1]
    elif changed == "config_contents":
        config = tmp_path / "changed.toml"
        config.write_text(
            Path(namespace["CONFIG_PATHS"][0]).read_text().replace("epochs = 30", "epochs = 29")
        )
        namespace["CONFIG_PATHS"][0] = str(config)
    else:
        namespace["audit"]["fingerprint"] = "different-data"
    with pytest.raises(ValueError, match="[Ff]rozen"):
        exec(notebook_cell("candidates ="), namespace)
    assert namespace["downloads"] == []


@pytest.mark.parametrize(
    "metadata", ["run_seed", "run_data", "saved_config", "data_audit", "incomplete_history"]
)
def test_colab_export_rejects_mismatched_completed_metadata(export_matrix, metadata):
    namespace = export_matrix
    directory = namespace["RUNS"] / "almt-seed1112"
    if metadata == "run_seed":
        rewrite_json(
            directory / "run.json", lambda value: value["config"]["training"].update(seed=9999)
        )
    elif metadata == "run_data":
        rewrite_json(directory / "run.json", lambda value: value.update(data_fingerprint="other"))
    elif metadata == "saved_config":
        rewrite_json(directory / "config.json", lambda value: value["training"].update(seed=9999))
    elif metadata == "data_audit":
        rewrite_json(directory / "data_audit.json", lambda value: value.update(fingerprint="other"))
    else:
        (directory / "history.json").write_text("[]")
    with pytest.raises(ValueError, match="almt-seed1112"):
        exec(notebook_cell("candidates ="), namespace)
    assert namespace["downloads"] == []


@pytest.mark.parametrize("status", ["trained", "evaluated"])
def test_colab_export_uses_only_frozen_seeds_and_validation_scores(export_matrix, status):
    namespace = export_matrix
    runs = namespace["RUNS"]
    rewrite_json(runs / "manifest.json", lambda value: value.update(status=status))
    rewrite_json(
        runs / "almt-seed1111/run.json",
        lambda value: value.update(
            best_validation={"macro_f1": 0.7, "loss": 0.8}, test_macro_f1=1.0
        ),
    )
    rewrite_json(
        runs / "almt-seed1112/run.json",
        lambda value: value.update(
            best_validation={"macro_f1": 0.7, "loss": 0.4}, test_macro_f1=0.0
        ),
    )
    shutil.copytree(runs / "almt-seed1111", runs / "almt-seed9999")
    rewrite_json(
        runs / "almt-seed9999/run.json",
        lambda value: value.update(best_validation={"macro_f1": 1.0, "loss": 0.0}),
    )
    exec(notebook_cell("candidates ="), namespace)
    assert namespace["selected"] == runs / "almt-seed1112"
    assert namespace["downloads"] == [str(namespace["export"])]
    with zipfile.ZipFile(namespace["export"]) as archive:
        result = json.loads(archive.read("selected-run/run.json"))
        assert result["config"]["training"]["seed"] == 1112


def test_colab_export_preserves_missing_default_config_compatibility(export_matrix):
    namespace = export_matrix

    def omit_default_model_fields(config):
        for field in list(config["model"]):
            if field.startswith(("almt_", "graph_")):
                del config["model"][field]

    manifest_path = namespace["RUNS"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for config in manifest["configs"]:
        omit_default_model_fields(config)
        for seed in manifest["seeds"]:
            directory = namespace["RUNS"] / f"{config['name']}-seed{seed}"
            rewrite_json(directory / "config.json", omit_default_model_fields)
            rewrite_json(
                directory / "run.json", lambda result: omit_default_model_fields(result["config"])
            )
    manifest_path.write_text(json.dumps(manifest))
    exec(notebook_cell("candidates ="), namespace)
    assert len(namespace["downloads"]) == 1


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
