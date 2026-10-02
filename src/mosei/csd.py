"""Read CMU SDK computational sequences without executing their metadata."""

import ast
import json
from pathlib import Path

import numpy as np


def _metadata_value(value):
    value = np.asarray(value).reshape(-1).tolist()
    value = [v.decode("utf-8") if isinstance(v, bytes) else v for v in value]
    if len(value) == 1:
        value = value[0]
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
    return value


class CSDStream:
    def __init__(self, path: Path, expected_root: str | None = None):
        import h5py

        self.handle = h5py.File(path, "r")
        try:
            roots = list(self.handle)
            if len(roots) != 1 or (expected_root is not None and roots[0] != expected_root):
                raise ValueError(f"Unexpected CSD root: {roots}; expected {expected_root}")
            self.root_name = roots[0]
            root = self.handle[self.root_name]
            if set(root) != {"metadata", "data"}:
                raise ValueError("CSD root must contain metadata and data")
            self.metadata = {k: _metadata_value(v[()]) for k, v in root["metadata"].items()}
            if self.metadata.get("root name") != self.root_name:
                raise ValueError("CSD metadata root differs from the stored root")
            self.feature_names = self.metadata.get("dimension names")
            if not isinstance(self.feature_names, list) or not all(
                isinstance(n, str) and n.strip() for n in self.feature_names
            ):
                raise ValueError("CSD requires documented feature column names")
            self.feature_names = [n.strip() for n in self.feature_names]
            if len(set(self.feature_names)) != len(self.feature_names):
                raise ValueError("CSD feature column names must be unique")
            self.data = root["data"]
            self.video_ids = sorted(self.data)
        except BaseException:
            self.handle.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.handle.close()

    def read(self, video_id: str) -> tuple[np.ndarray, np.ndarray]:
        group = self.data[video_id]
        intervals = np.asarray(group["intervals"], dtype=np.float64)
        values = np.asarray(group["features"])
        if (
            intervals.ndim != 2
            or intervals.shape[1] != 2
            or not np.isfinite(intervals).all()
            or np.any(intervals[:, 1] < intervals[:, 0])
            or (self.root_name != "All Labels" and np.any(np.diff(intervals[:, 0]) < 0))
        ):
            raise ValueError(f"Invalid intervals in {self.root_name}/{video_id}")
        if values.ndim != 2 or values.shape != (len(intervals), len(self.feature_names)):
            raise ValueError(f"Feature row/column mismatch in {self.root_name}/{video_id}")
        if values.dtype.kind in "SO":
            values = np.array(
                [
                    [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in row]
                    for row in values
                ],
                dtype=str,
            ).reshape(values.shape)
        return intervals, values


def parse_sdk_splits(source: str) -> dict[str, list[str]]:
    """Read official literal fold assignments; never import/execute network code."""
    names = {f"standard_{split}_fold": split for split in ("train", "valid", "test")}
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise ValueError("SDK folds must be literal lists") from exc
                if (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(v, str) and v for v in value)
                ):
                    raise ValueError("SDK folds must be nonempty literal lists of video IDs")
                if len(set(value)) != len(value):
                    raise ValueError("Duplicate video in SDK split")
                result[names[target.id]] = value
    if set(result) != {"train", "valid", "test"}:
        raise ValueError("SDK source does not contain all official fold lists")
    all_ids = [v for values in result.values() for v in values]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("SDK source video overlap between splits")
    return result
