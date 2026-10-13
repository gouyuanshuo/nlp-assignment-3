import hashlib
import importlib.machinery
import io
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from mosei.data import prepare_data


def test_notebook_exposes_new_source_to_the_running_kernel(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "notebook_import_probe.py").write_text("ready = True\n")
    kernel = SimpleNamespace(path=[], modules={})
    namespace = {
        "sys": kernel,
        "PROJECT": tmp_path,
        "tools": {"manifest": {"software": {}}},
    }
    setup = runpy.run_path("scripts/build_colab_notebooks.py")["POST_INSTALL"]
    exec(setup, namespace)
    found = importlib.machinery.PathFinder.find_spec("notebook_import_probe", kernel.path)
    assert found is not None
    assert found.origin == str(source / "notebook_import_probe.py")


@pytest.mark.parametrize(
    "distribution,module", [("numpy", "numpy"), ("faster-whisper", "faster_whisper")]
)
def test_notebook_requires_restart_for_stale_loaded_packages(tmp_path, distribution, module):
    setup = runpy.run_path("scripts/build_colab_notebooks.py")["POST_INSTALL"]
    kernel = SimpleNamespace(path=[], modules={module: SimpleNamespace(__version__="1.0")})
    namespace = {
        "sys": kernel,
        "PROJECT": tmp_path,
        "tools": {"manifest": {"software": {distribution: "2.0"}}},
    }
    with pytest.raises(RuntimeError, match="Restart") as error:
        exec(setup, namespace)
    assert distribution in str(error.value)
    kernel.modules[module].__version__ = "2.0"
    exec(setup, namespace)


def test_sdk_download_verifies_bytes_before_writing_folds(tmp_path, monkeypatch):
    module = runpy.run_path("scripts/download_splits.py")
    fetch = module["download_splits"]
    source = (
        b"standard_train_fold=['train']\nstandard_valid_fold=['valid']\n"
        b"standard_test_fold=['test']\nraise RuntimeError('Never execute network code')\n"
    )
    monkeypatch.setitem(fetch.__globals__, "urlopen", lambda *a, **k: io.BytesIO(source))
    destination = tmp_path / "splits.json"
    with pytest.raises(ValueError, match="checksum"):
        fetch(destination)
    assert not destination.exists()
    monkeypatch.setitem(fetch.__globals__, "SDK_SHA256", hashlib.sha256(source).hexdigest())
    fetch(destination)
    assert json.loads(destination.read_text())["splits"] == {
        "train": ["train"],
        "valid": ["valid"],
        "test": ["test"],
    }


def test_matrix_resume_rebuilds_interrupted_report(tmp_path, feature_file, monkeypatch):
    source, sha = feature_file
    prepared = tmp_path / "data"
    prepare_data(source, prepared, sha, expected_dims=None)
    config = tmp_path / "tiny.toml"
    config.write_text("""name = "tiny"
[model]
name = "text"
hidden_dim = 8
[training]
epochs = 1
batch_size = 2
device = "cpu"
""")
    matrix = tmp_path / "matrix"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/run_experiments.py",
            "--data",
            str(prepared),
            "--configs",
            str(config),
            "--seeds",
            "1111",
            "--output",
            str(matrix),
            "--device",
            "cpu",
            "--resume",
            "--bootstrap-samples",
            "5",
        ],
    )
    module = runpy.run_path("scripts/run_experiments.py")
    module["main"]()
    report = matrix / "report"
    # A disconnect during report export can leave JSON but no image or final marker.
    (report / "tiny_confusion.png").unlink()
    (report / "complete.json").unlink(missing_ok=True)
    module["main"]()
    assert (report / "tiny_confusion.png").is_file()
    assert (report / "complete.json").is_file()
    preserved = list(matrix.glob("report-interrupted-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "results.json").is_file()
