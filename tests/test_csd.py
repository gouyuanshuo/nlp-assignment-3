"""Tiny HDF5 fixtures exercise the real CSD reader, not network services."""

import json

import h5py
import numpy as np
import pytest


def write_csd(path, values, intervals, names, root="COVAREP"):
    with h5py.File(path, "w") as handle:
        group = handle.create_group(root)
        meta = group.create_group("metadata")
        for key, value in {"root name": root, "dimension names": names}.items():
            meta.create_dataset(key, data=[json.dumps(value).encode()])
        clip = group.create_group("data/video1")
        clip.create_dataset("features", data=values)
        clip.create_dataset("intervals", data=intervals)
    return path


def test_named_columns_and_timestamps_survive_csd_read(tmp_path):
    from mosei.csd import CSDStream

    path = write_csd(
        tmp_path / "a.csd", [[100, 1], [120, 0]], [[0, 0.1], [0.1, 0.2]], ["F0", "VUV"]
    )
    with CSDStream(path, expected_root="COVAREP") as stream:
        intervals, values = stream.read("video1")
        assert stream.feature_names == ["F0", "VUV"]
        assert stream.video_ids == ["video1"]
        np.testing.assert_array_equal(values, [[100, 1], [120, 0]])
        np.testing.assert_allclose(intervals, [[0, 0.1], [0.1, 0.2]])


@pytest.mark.parametrize("intervals", [[[1, 0]], [[float("nan"), 1]], [[1, 2], [0, 1]]])
def test_invalid_intervals_are_rejected(tmp_path, intervals):
    from mosei.csd import CSDStream

    path = write_csd(tmp_path / "a.csd", [[1]] * len(intervals), intervals, ["F0"])
    with CSDStream(path) as stream, pytest.raises(ValueError, match="interval"):
        stream.read("video1")


def test_column_width_mismatch_is_rejected(tmp_path):
    from mosei.csd import CSDStream

    path = write_csd(tmp_path / "a.csd", [[1, 2]], [[0, 1]], ["F0"])
    with CSDStream(path) as stream, pytest.raises(ValueError, match="column"):
        stream.read("video1")


def test_wrong_stream_cannot_be_used_as_another_modality(tmp_path):
    from mosei.csd import CSDStream

    path = write_csd(tmp_path / "a.csd", [[1]], [[0, 1]], ["sentiment"], root="All Labels")
    with pytest.raises(ValueError, match="root"):
        CSDStream(path, expected_root="COVAREP")


def test_words_decode_and_point_measurements_are_supported(tmp_path):
    from mosei.csd import CSDStream

    path = write_csd(
        tmp_path / "words.csd",
        np.array([[b"great"], [b"movie"]]),
        [[0, 0], [1, 1]],
        ["word"],
        root="words",
    )
    with CSDStream(path) as stream:
        _, words = stream.read("video1")
        assert words.tolist() == [["great"], ["movie"]]


def test_split_parser_rejects_overlap_and_does_not_execute_source():
    from mosei.csd import parse_sdk_splits

    source = "standard_train_fold=['a']; standard_valid_fold=['b']; standard_test_fold=['c']\nraise RuntimeError('must not execute')"
    assert parse_sdk_splits(source) == {"train": ["a"], "valid": ["b"], "test": ["c"]}
    with pytest.raises(ValueError, match="overlap"):
        parse_sdk_splits(source.replace("['b']", "['a']"))
    with pytest.raises(ValueError, match="literal"):
        parse_sdk_splits(source.replace("['a']", "list('a')"))
