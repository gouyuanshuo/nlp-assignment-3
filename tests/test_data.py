import hashlib
import pickle

import numpy as np
import pytest

from mosei.data import load_data, prepare_data


def prepare(tmp_path, raw, **kwargs):
    source = tmp_path / "input.pkl"
    source.write_bytes(pickle.dumps(raw))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / "prepared"
    audit = prepare_data(source, out, expected_sha256=sha, expected_dims=None, **kwargs)
    return load_data(out), audit


def test_neutral_exclusion_and_train_only_moments(tmp_path, raw_features):
    data, audit = prepare(tmp_path, raw_features)
    assert data.splits["train"].labels.tolist() == [0, 1]
    assert data.splits["test"].ids == ["testvideo$_$0", "testvideo$_$1"]
    assert audit["splits"]["train"]["excluded_neutral"] == 1
    np.testing.assert_allclose(data.normalization["text"]["mean"], [4, 4, 4, 4])
    np.testing.assert_allclose(data.normalization["text"]["scale"], np.sqrt(5))
    np.testing.assert_allclose(data.splits["valid"].features["text"][0, 0], 17 / np.sqrt(5))
    assert not data.splits["train"].features["text"][:, 2].any()


def test_metadata_masks_win_over_nonzero_padding_and_zero_valid_features(tmp_path, raw_features):
    raw_features["train"]["audio"][0, 0] = 0
    data, _ = prepare(tmp_path, raw_features)
    assert data.splits["train"].masks["audio"][0].tolist() == [True, True, False]
    assert data.splits["train"].masks["text"].dtype == np.bool_


def test_nonfinite_values_imputed_with_finite_training_mean(tmp_path, raw_features):
    raw_features["train"]["audio"][0, 0, 0] = -np.inf
    raw_features["valid"]["audio"][0, 0, 0] = np.nan
    data, audit = prepare(tmp_path, raw_features)
    assert data.normalization["audio"]["mean"][0] == 5
    assert data.splits["valid"].features["audio"][0, 0, 0] == 0
    assert audit["splits"]["train"]["nonfinite_valid"]["audio"] == 1
    assert np.isfinite(data.splits["train"].features["audio"]).all()


def test_duplicate_clip_and_cross_split_video_rejected(tmp_path, raw_features):
    raw_features["test"]["id"][0] = "trainvideo$_$99"
    with pytest.raises(ValueError, match="video.*overlap"):
        prepare(tmp_path, raw_features)


def test_neutral_rows_are_included_in_leakage_audit(tmp_path, raw_features):
    raw_features["test"]["id"][2] = "trainvideo$_$2"
    with pytest.raises(ValueError, match="overlap"):
        prepare(tmp_path, raw_features)


def test_ambiguous_or_invalid_masks_rejected(tmp_path, raw_features):
    del raw_features["valid"]["text_bert"]
    with pytest.raises(ValueError, match="mask|length"):
        prepare(tmp_path, raw_features)


def test_incorrect_checksum_rejected_before_deserialization(tmp_path):
    source = tmp_path / "not-pickle"
    source.write_text("not a pickle")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_data(source, tmp_path / "out", expected_sha256="0" * 64)


def test_empty_class_and_out_of_range_labels_rejected(tmp_path, raw_features):
    raw_features["valid"]["regression_labels"][0] = -4
    with pytest.raises(ValueError, match="label"):
        prepare(tmp_path, raw_features)


@pytest.mark.parametrize("lengths", [[1, 2, 1], [999, 2, 1]])
def test_mask_does_not_hide_conflicting_or_invalid_lengths(tmp_path, raw_features, lengths):
    raw_features["train"]["audio_mask"] = raw_features["train"]["text_bert"][:, 1].copy()
    raw_features["train"]["audio_lengths"] = np.array(lengths)
    with pytest.raises(ValueError, match="Conflicting|length"):
        prepare(tmp_path, raw_features)


def test_explicit_text_mask_must_agree_with_bert_attention_mask(tmp_path, raw_features):
    raw_features["train"]["text_mask"] = np.ones((3, 3))
    with pytest.raises(ValueError, match="Conflicting"):
        prepare(tmp_path, raw_features)


def test_consistent_audio_metadata_can_differ_from_text_metadata(tmp_path, raw_features):
    raw_features["train"]["audio_lengths"] = np.array([1, 1, 1])
    raw_features["train"]["audio_mask"] = np.array([[1, 0, 0]] * 3)
    data, _ = prepare(tmp_path, raw_features)
    assert data.splits["train"].masks["audio"][0].tolist() == [True, False, False]
    assert data.splits["train"].masks["text"][0].tolist() == [True, True, False]
