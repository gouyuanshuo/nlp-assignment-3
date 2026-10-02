"""Small synthetic fixtures; none of these are MOSEI experimental results."""

import hashlib
import pickle

import numpy as np
import pytest


@pytest.fixture
def raw_features():
    # Hand-checkable mean of eligible training tokens: (1 + 3 + 5 + 7) / 4 = 4.
    data = {}
    for split, offset in [("train", 0), ("valid", 20), ("test", 40)]:
        mask = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.int64)
        values = np.array([[1, 3, 999], [5, 7, 999], [100, 999, 999]], np.float32)
        values = values + offset
        bert = np.zeros((3, 3, 3), dtype=np.int64)
        bert[:, 1] = mask
        data[split] = {
            "text": np.repeat(values[..., None], 4, axis=2),
            "audio": np.repeat(values[..., None], 2, axis=2),
            "vision": values[..., None],
            "text_bert": bert,
            "regression_labels": np.array([[-1], [2], [0]], dtype=np.float32),
            "id": [f"{split}video$_${i}" for i in range(3)],
            "raw_text": ["negative synthetic example", "positive synthetic example", "neutral"],
        }
    return data


@pytest.fixture
def feature_file(tmp_path, raw_features):
    path = tmp_path / "fixture.pkl"
    path.write_bytes(pickle.dumps(raw_features))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def limit_torch_threads():
    import importlib.util

    if importlib.util.find_spec("torch"):
        import torch

        torch.set_num_threads(1)
