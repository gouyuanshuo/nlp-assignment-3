import numpy as np
import pytest

from mosei.config import TrainingConfig
from mosei.subset import training_subset


class _Clips:
    def __init__(self):
        self.labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
        self.video_ids = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"]

    def __len__(self):
        return len(self.labels)


def test_video_subset_is_deterministic_and_keeps_whole_videos():
    data = _Clips()
    first, first_record = training_subset(data, 0.5, 1111)
    second, second_record = training_subset(data, 0.5, 1111)
    assert first_record["video_ids"] == second_record["video_ids"]
    assert list(first.indices) == list(second.indices)
    selected = set(first_record["video_ids"])
    for video_id in data.video_ids:
        positions = [i for i, value in enumerate(data.video_ids) if value == video_id]
        assert set(positions).issubset(first.indices) or set(positions).isdisjoint(first.indices)
        if video_id in selected:
            assert set(positions).issubset(first.indices)
    assert first_record["class_counts"]["negative"] > 0
    assert first_record["class_counts"]["positive"] > 0
    assert first_record["clip_count"] < len(data)
    assert {data.video_ids[i] for i in first.indices} == selected


def test_full_fraction_uses_every_training_clip():
    data = _Clips()
    subset, record = training_subset(data, 1, 1111)
    assert subset is data
    assert record is None


def test_train_fraction_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        TrainingConfig(train_fraction=0)
    with pytest.raises(ValueError):
        TrainingConfig(train_fraction=1.5)
