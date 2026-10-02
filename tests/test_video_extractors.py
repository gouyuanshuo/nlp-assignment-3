import csv

import numpy as np
import pytest
from scipy.io import savemat, wavfile

from mosei.video_extractors import read_covarep, read_openface, transcribe_audio


def test_covarep_names_and_sampling_are_read_not_guessed(tmp_path):
    path = tmp_path / "audio.mat"
    savemat(
        path,
        {
            "names": np.array(["VUV", "F0"], dtype=object),
            "features": [[1.0, 120.0], [0.0, 0.0]],
            "intervals": [[0.005, 0.005], [0.015, 0.015]],
        },
    )
    result = read_covarep(path, {"columns": ["F0", "VUV"]})
    np.testing.assert_array_equal(result.values, [[120, 1], [0, 0]])
    np.testing.assert_allclose(result.intervals[:, 0], [0.005, 0.015])
    with pytest.raises(ValueError, match="column"):
        read_covarep(path, {"columns": ["invented"]})


def test_face_quality_matches_training_mask_and_faceless_input_fails(tmp_path):
    path = tmp_path / "features.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([" frame", " timestamp", " confidence", " success", " AU01_r"])
        writer.writerows([[1, 0.0, 0.99, 1, 3.0], [2, 0.1, 0.1, 0, 5.0]])
    result = read_openface(path, {"columns": ["AU01_r"], "min_confidence": 0.8})
    assert result.values[0, 0] == 3
    assert np.isnan(result.values[1, 0])
    np.testing.assert_allclose(result.intervals, [[0.0, 0.1], [0.1, 0.2]])
    with pytest.raises(ValueError, match="face"):
        read_openface(path, {"columns": ["AU01_r"], "min_confidence": 1})


def test_silent_audio_stops_before_speech_model_runs(tmp_path):
    path = tmp_path / "audio.wav"
    wavfile.write(path, 16000, np.zeros(16000, dtype=np.int16))
    with pytest.raises(ValueError, match="speech|silent"):
        transcribe_audio(path, None)
