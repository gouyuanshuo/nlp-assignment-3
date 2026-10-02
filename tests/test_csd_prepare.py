import json

import h5py
import numpy as np
import pytest

from mosei.csd_prepare import prepare_csd
from mosei.data import load_data
from mosei.io import write_json


class TinyTokenizer:
    model_max_length = 512

    def __call__(self, words, **kwargs):
        return {"input_ids": [0] + list(range(len(words))) + [0]}


class TinyEncoder:
    tokenizer = TinyTokenizer()

    def encode(self, windows):
        return [np.array([[float(w.text)] for w in words], dtype=np.float32) for words in windows]


@pytest.fixture
def csd_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    roots = {"words": "words", "labels": "All Labels", "audio": "COVAREP", "vision": "OpenFace_2"}
    columns = {
        "words": ["word"],
        "labels": ["sentiment"],
        "audio": ["F0"],
        "vision": ["AU01_r", "confidence", "success"],
    }
    files = {}
    for key, root in roots.items():
        files[key] = key + ".csd"
        with h5py.File(source / files[key], "w") as f:
            group = f.create_group(root)
            metadata = group.create_group("metadata")
            metadata["root name"] = [json.dumps(root)]
            metadata["dimension names"] = [json.dumps(columns[key])]
            data = group.create_group("data")
            for split, shift in [("train", 0), ("valid", 100), ("test", 200)]:
                video = data.create_group(split)
                if key == "labels":
                    video["intervals"] = [[0.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
                    video["features"] = [[-1.0], [1.0], [0.0]]
                else:
                    video["intervals"] = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
                    numbers = [1 + shift, 3 + shift, 5 + shift, 999]
                    if key == "words":
                        video["features"] = np.array([[str(n).encode()] for n in numbers])
                    elif key == "vision":
                        video["features"] = [[n, 0.95, 1] for n in numbers]
                    else:
                        video["features"] = np.array(numbers)[:, None]
    manifest = {
        "schema_version": 1,
        "max_words": 1,
        "files": files,
        "roots": roots,
        "text": {"dim": 1},
        "audio": {"columns": ["F0"], "source_columns": ["F0"]},
        "vision": {
            "columns": ["AU01_r"],
            "source_columns": columns["vision"],
            "min_confidence": 0.8,
        },
    }
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "splits.json", {"splits": {s: [s] for s in ("train", "valid", "test")}})
    return source, tmp_path / "manifest.json", tmp_path / "splits.json"


def test_csd_cache_has_clip_offsets_and_train_only_statistics(tmp_path, csd_source):
    source, manifest, splits = csd_source
    audit = prepare_csd(source, tmp_path / "cache", manifest, splits, encoder=TinyEncoder())
    data = load_data(tmp_path / "cache")
    assert audit["cache_version"] == 2
    assert data.splits["train"].offsets.tolist() == [0, 2, 3]
    assert data.splits["train"].labels.tolist() == [0, 1]
    assert audit["splits"]["train"]["excluded_neutral"] == 1
    assert audit["splits"]["train"]["window_count"] == 3
    for m in ("text", "audio", "vision"):
        assert data.normalization[m]["mean"] == [3.0]
        assert data.normalization[m]["scale"] == pytest.approx([np.sqrt(8 / 3)])
        assert data.splits["valid"].features[m][0, 0, 0] > 50
    assert "feature_manifest" in audit
    assert len(audit["source_hashes"]) == 4


def test_split_overlap_rejected_before_encoding(tmp_path, csd_source):
    source, manifest, splits = csd_source
    write_json(splits, {"splits": {"train": ["train"], "valid": ["train"], "test": ["test"]}})
    with pytest.raises(ValueError, match="overlap"):
        prepare_csd(source, tmp_path / "cache", manifest, splits, encoder=None)
    assert not (tmp_path / "cache").exists()


def test_unknown_column_order_fails_before_encoding(tmp_path, csd_source):
    source, manifest, splits = csd_source
    payload = json.loads(manifest.read_text())
    payload["audio"]["source_columns"] = ["other"]
    write_json(manifest, payload)
    with pytest.raises(ValueError, match="column"):
        prepare_csd(source, tmp_path / "cache", manifest, splits, encoder=None)


@pytest.mark.parametrize("filename", ["labels.npy", "text.npy", "clips.json"])
def test_prepared_cache_rejects_modified_training_content(tmp_path, csd_source, filename):
    source, manifest, splits = csd_source
    cache = tmp_path / "cache"
    prepare_csd(source, cache, manifest, splits, encoder=TinyEncoder())
    path = cache / "train" / filename
    if path.suffix == ".json":
        clips = json.loads(path.read_text())
        clips["raw_text"][0] = "changed training transcript"
        write_json(path, clips)
    else:
        values = np.load(path)
        values.flat[0] = 1 if filename == "labels.npy" else 123
        np.save(path, values, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum|content"):
        load_data(cache)


def test_colab_cache_recovers_partial_extraction_and_publication(tmp_path, csd_source, monkeypatch):
    import shutil

    from mosei.colab_cache import persist_cache, restore_cache

    source, manifest, splits = csd_source
    original = tmp_path / "original" / "prepared-csd"
    prepare_csd(source, original, manifest, splits, encoder=TinyEncoder())
    archive = tmp_path / "drive" / "prepared.tar"
    persist_cache(original, archive)
    cache = tmp_path / "runtime" / "prepared-csd"
    cache.mkdir(parents=True)
    shutil.copy2(original / "audit.json", cache / "audit.json")
    assert restore_cache(cache, archive)
    assert load_data(cache).fingerprint == load_data(original).fingerprint
    assert list(cache.parent.glob("prepared-csd-interrupted-*"))
    # A local cache must still be persisted on rerun after Drive publication failed.
    archive.unlink()
    archive.with_suffix(".tar.json").unlink()
    actual_replace = type(archive).replace

    def disconnect(path, destination):
        if str(destination).endswith(".tar.json"):
            raise OSError("simulated publication disconnect")
        return actual_replace(path, destination)

    monkeypatch.setattr(type(archive), "replace", disconnect)
    with pytest.raises(OSError, match="disconnect"):
        persist_cache(cache, archive)
    monkeypatch.setattr(type(archive), "replace", actual_replace)
    assert restore_cache(cache, archive)
    persist_cache(cache, archive)
    fresh = tmp_path / "fresh" / "prepared-csd"
    assert restore_cache(fresh, archive)
    assert load_data(fresh).fingerprint == load_data(original).fingerprint


def test_colab_extraction_becomes_visible_only_after_complete_validation(
    tmp_path, csd_source, monkeypatch
):
    import tarfile

    from mosei.colab_cache import persist_cache, restore_cache

    source, manifest, splits = csd_source
    original = tmp_path / "original"
    prepare_csd(source, original, manifest, splits, encoder=TinyEncoder())
    archive = tmp_path / "drive" / "prepared.tar"
    persist_cache(original, archive)
    extract = tarfile.TarFile.extractall

    def partial(tar, path, **kwargs):
        tar.extract("prepared-csd/audit.json", path=path, filter="data")
        raise RuntimeError("simulated extraction disconnect")

    monkeypatch.setattr(tarfile.TarFile, "extractall", partial)
    cache = tmp_path / "runtime" / "prepared-csd"
    with pytest.raises(RuntimeError, match="disconnect"):
        restore_cache(cache, archive)
    assert not cache.exists()
    monkeypatch.setattr(tarfile.TarFile, "extractall", extract)
    assert restore_cache(cache, archive)
    assert load_data(cache).fingerprint == load_data(original).fingerprint
