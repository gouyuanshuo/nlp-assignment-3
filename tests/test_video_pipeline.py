from pathlib import Path

import numpy as np
import pytest
import torch

from mosei.video_pipeline import VideoClassifier, ensure_compatible, infer_windows


def test_old_checkpoint_and_changed_manifest_cannot_serve_uploads():
    with pytest.raises(ValueError, match="video|manifest"):
        ensure_compatible({"feature_dims": {"text": 768}}, {"schema_version": 1})
    with pytest.raises(ValueError, match="manifest"):
        ensure_compatible(
            {"feature_manifest": {"audio": {"columns": ["F0"]}}}, {"audio": {"columns": ["VUV"]}}
        )


def test_inference_aggregates_logits_and_preserves_window_times():
    class Model(torch.nn.Module):
        def forward(self, features, masks):
            return features["text"][:, 0]

    window = {
        "features": {"text": np.array([[[0.0, 2.0]], [[4.0, 0.0]]], np.float32)},
        "masks": {"text": np.ones((2, 1), bool)},
        "durations": np.array([1.0, 3.0], np.float32),
        "intervals": np.array([[0.0, 1.0], [1.0, 4.0]]),
        "texts": ["first", "last"],
    }
    result = infer_windows(Model(), window, "cpu")
    assert result["prediction"] == "negative"
    assert result["probabilities"]["positive"] == pytest.approx(
        torch.tensor([3.0, 0.5]).softmax(0)[1].item()
    )
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert [r["end"] for r in result["windows"]] == [1.0, 4.0]
    assert result["windows"][1]["text"] == "last"


def test_failed_extraction_cleans_work_directory(tmp_path, monkeypatch):
    # Fault injection at the external extractor boundary; actual cleanup executes.
    import mosei.video_pipeline as pipeline

    classifier = object.__new__(VideoClassifier)
    classifier.manifest = {"audio": {}, "vision": {}}
    classifier.speech = object()
    classifier.temp_root = tmp_path
    created = []
    monkeypatch.setattr(pipeline, "probe_video", lambda p: None)

    def extract(video, output):
        output.write_bytes(b"private temporary audio")
        created.append(output)
        raise ValueError("broken media")

    monkeypatch.setattr(pipeline, "extract_audio", extract)
    with pytest.raises(ValueError, match="broken"):
        classifier.classify(Path("unused.mp4"))
    assert created and not created[0].exists()
    assert list(tmp_path.iterdir()) == []


def test_failed_readiness_check_blocks_model_loading(tmp_path):
    from mosei.io import write_json

    write_json(
        tmp_path / "tools.json",
        {"manifest": {"schema_version": 1}, "fingerprint": "wrong", "paths": {}},
    )
    with pytest.raises(ValueError, match="fingerprint"):
        VideoClassifier(tmp_path / "missing-checkpoint", tmp_path / "tools.json")
