"""Shared timestamp alignment and window preparation for CSD and uploaded media."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float

    def __post_init__(self):
        if (
            not self.text.strip()
            or not np.isfinite([self.start, self.end]).all()
            or not 0 <= self.start < self.end
        ):
            raise ValueError("A word requires nonempty text and a finite, positive interval")


@dataclass
class TimedFeatures:
    intervals: np.ndarray
    values: np.ndarray
    names: list[str]

    def __post_init__(self):
        self.intervals = np.asarray(self.intervals, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float32)
        if self.intervals.shape != (len(self.values), 2) or self.values.shape != (
            len(self.values),
            len(self.names),
        ):
            raise ValueError("Feature columns and intervals must match the values")
        if not np.isfinite(self.intervals).all() or np.any(
            self.intervals[:, 1] < self.intervals[:, 0]
        ):
            raise ValueError("Invalid feature intervals")
        if np.any(np.diff(self.intervals[:, 0]) < 0) or len(set(self.names)) != len(self.names):
            raise ValueError("Feature intervals must be ordered and column names unique")


def align_words(words: list[Word], stream: TimedFeatures) -> tuple[np.ndarray, np.ndarray]:
    result = np.full((len(words), len(stream.names)), np.nan, dtype=np.float32)
    starts, ends = stream.intervals.T
    for index, word in enumerate(words):
        selected = (starts < word.end) & (ends >= word.start)
        values = stream.values[selected]
        low, high = starts[selected], ends[selected]
        weights = np.maximum(0, np.minimum(high, word.end) - np.maximum(low, word.start))
        points = (low == high) & (low >= word.start) & (low < word.end)
        weights = np.where(points, 1.0, weights)[:, None]
        finite = np.isfinite(values)
        totals = (weights * finite).sum(axis=0)
        sums = (weights * np.where(finite, values, 0)).sum(axis=0)
        np.divide(sums, totals, out=result[index], where=totals > 0)
    return result, np.isfinite(result).any(axis=1)


def make_windows(words: list[Word], tokenizer, max_words: int = 50) -> list[list[Word]]:
    if not words or max_words < 1:
        raise ValueError("No speech words available for classification")
    if any(b.start < a.start for a, b in zip(words, words[1:])):
        raise ValueError("Speech word intervals must be ordered")
    limit = min(int(tokenizer.model_max_length), 512)
    windows, current = [], []
    for word in words:
        candidate = current + [word]
        tokens = tokenizer(
            [w.text for w in candidate],
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        if len(candidate) > max_words or len(tokens) > limit:
            if not current:
                raise ValueError("One word exceeds the text encoder token limit")
            windows.append(current)
            current = []
            if (
                len(
                    tokenizer(
                        [word.text],
                        is_split_into_words=True,
                        add_special_tokens=True,
                        truncation=False,
                    )["input_ids"]
                )
                > limit
            ):
                raise ValueError("One word exceeds the text encoder token limit")
        current.append(word)
    if current:
        windows.append(current)
    return windows


def normalize_features(values: np.ndarray, mask: np.ndarray, stats: dict) -> np.ndarray:
    mean, scale = (np.asarray(stats[k], dtype=np.float32) for k in ("mean", "scale"))
    if (
        mean.shape != (values.shape[-1],)
        or scale.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Normalization does not match feature columns")
    filled = np.where(np.isfinite(values), values, mean)
    return np.where(mask[..., None], (filled - mean) / scale, 0).astype(np.float32)


def prepare_windows(words, audio, vision, encoder, manifest, normalization=None) -> dict:
    for name, stream in (("audio", audio), ("vision", vision)):
        if stream.names != manifest[name]["columns"]:
            raise ValueError(f"{name} feature columns differ from checkpoint manifest")
    width = manifest.get("max_words", 50)
    windows = make_windows(words, encoder.tokenizer, width)
    texts = encoder.encode(windows)
    raw = {"text": texts, "audio": [], "vision": []}
    masks = {"text": [np.ones(len(w), dtype=bool) for w in windows], "audio": [], "vision": []}
    for words_in_window in windows:
        for name, stream in (("audio", audio), ("vision", vision)):
            values, valid = align_words(words_in_window, stream)
            raw[name].append(values)
            masks[name].append(valid)
    features, valid = {}, {}
    for name, arrays in raw.items():
        features[name] = np.zeros((len(windows), width, arrays[0].shape[1]), dtype=np.float32)
        valid[name] = np.zeros((len(windows), width), dtype=bool)
        for i, (values, mask) in enumerate(zip(arrays, masks[name], strict=True)):
            features[name][i, : len(values)] = values
            valid[name][i, : len(values)] = mask
        if normalization is not None:
            features[name] = normalize_features(features[name], valid[name], normalization[name])
    intervals = np.array([[w[0].start, w[-1].end] for w in windows], dtype=np.float64)
    return {
        "features": features,
        "masks": valid,
        "intervals": intervals,
        "durations": (intervals[:, 1] - intervals[:, 0]).astype(np.float32),
        "texts": [" ".join(w.text for w in window) for window in windows],
    }
