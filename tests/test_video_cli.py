import pytest

from mosei.cli import _parser, main
from mosei.io import write_json
from mosei.web_demo import analyze_upload, empty_result


def test_video_commands_accept_explicit_artifact_paths():
    parser = _parser()
    serve = parser.parse_args(["serve", "--run", "runs/trained", "--tools", "data/tools.json"])
    assert serve.host == "127.0.0.1" and serve.port == 7860
    prepare = parser.parse_args(
        [
            "prepare-csd",
            "--source",
            "data/csd",
            "--output",
            "data/cache",
            "--tools",
            "data/tools.json",
            "--splits",
            "data/splits.json",
        ]
    )
    assert prepare.command == "prepare-csd"
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--run", "x", "--tools", "x", "--host", "0.0.0.0"])


def test_doctor_bad_bundle_reports_error_without_predictions(tmp_path, capsys):
    write_json(tmp_path / "tools.json", {"manifest": {}, "fingerprint": "bad"})
    with pytest.raises(SystemExit) as error:
        main(["doctor", "--tools", str(tmp_path / "tools.json")])
    assert error.value.code == 2
    assert "fingerprint" in capsys.readouterr().err


def test_new_analysis_clears_results_then_returns_real_classifier_response():
    class Classifier:
        def classify(self, path, progress=None):
            return {
                "prediction": "positive",
                "probabilities": {"negative": 0.2, "positive": 0.8},
                "transcript": "Really good.",
                "processing_seconds": 12.5,
                "windows": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Really good.",
                        "prediction": "positive",
                        "prob_negative": 0.2,
                        "prob_positive": 0.8,
                    }
                ],
            }

    values = list(analyze_upload("local.mp4", Classifier()))
    assert values[0][1:] == empty_result()[1:]
    assert values[-1][1] == {"negative": 0.2, "positive": 0.8}
    assert values[-1][2] == "Really good."
    assert values[-1][3] == [[0.0, 2.0, "positive", 0.8, "Really good."]]


def test_failed_new_upload_has_no_stale_prediction_or_transcript():
    class Broken:
        def classify(self, path, progress=None):
            raise ValueError("No speech detected")

    values = list(analyze_upload("bad.mp4", Broken()))
    assert "No speech" in values[-1][0]
    assert values[-1][1:] == (None, "", [])
