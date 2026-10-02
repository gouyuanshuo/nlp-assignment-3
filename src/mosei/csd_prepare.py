"""Prepare timestamped CSD streams without dropping later windows or leaking folds."""

import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .csd import CSDStream
from .data import LABEL_RULE, MODALITIES, SPLITS, _moments
from .io import fingerprint, read_json, sha256_file, write_json
from .video_features import TimedFeatures, Word, normalize_features, prepare_windows


def select_features(intervals, values, names, section):
    """Select named channels and apply identical OpenFace quality masks on both paths."""
    wanted = section["columns"]
    if len(set(wanted)) != len(wanted) or not set(wanted) <= set(names):
        raise ValueError("Requested feature columns are missing or duplicated")
    selected = np.asarray(values[:, [names.index(n) for n in wanted]], dtype=np.float32).copy()
    if "min_confidence" in section:
        if not {"confidence", "success"} <= set(names):
            raise ValueError("OpenFace confidence/success columns are missing")
        confidence = values[:, names.index("confidence")]
        success = values[:, names.index("success")]
        valid = np.isfinite(confidence) & (confidence >= section["min_confidence"]) & (success == 1)
        selected[~valid] = np.nan
    return TimedFeatures(intervals, selected, wanted)


def _clip_words(intervals, values, start, end):
    # Assign words by midpoint, then clip their bounds to the labelled interval.
    # Non-overlapping adjacent labels cannot double-count a boundary word.
    # Source annotations that genuinely overlap retain their original clip labels.
    keep = ((intervals[:, 0] + intervals[:, 1]) / 2 >= start) & (
        (intervals[:, 0] + intervals[:, 1]) / 2 < end
    )
    words = []
    for (low, high), row in zip(intervals[keep], values[keep], strict=True):
        text = str(row[0]).strip()
        low, high = max(float(low), float(start), 0), min(float(high), float(end))
        if text and text.lower() not in {"sp", "sil", "<sil>", "<sp>"} and low < high:
            words.append(Word(text, low, high))
    return words


def _write_split(split, videos, streams, folder, manifest, encoder, max_clips):
    folder.mkdir()
    clips = {"ids": [], "video_ids": [], "raw_text": []}
    offsets, durations, intervals, scores = [0], [], [], []
    audit = {
        "original_count": 0,
        "excluded_neutral": 0,
        "excluded_no_words": 0,
        "videos_missing_streams": [],
        "missing_modality_clips": {m: 0 for m in MODALITIES},
    }
    dims = {
        "text": manifest["text"]["dim"],
        **{m: len(manifest[m]["columns"]) for m in ("audio", "vision")},
    }
    with ExitStack() as stack:
        handles = {m: stack.enter_context((folder / f"{m}.raw").open("wb")) for m in MODALITIES}
        masks = {m: stack.enter_context((folder / f"{m}_mask.raw").open("wb")) for m in MODALITIES}
        for video in videos:
            if any(video not in s.data for s in streams.values()):
                audit["videos_missing_streams"].append(video)
                continue
            label_times, label_values = streams["labels"].read(video)
            sentiment = label_values[:, streams["labels"].feature_names.index("sentiment")]
            if not np.isfinite(sentiment).all() or np.any(np.abs(sentiment) > 3):
                raise ValueError(f"Invalid sentiment labels in {video}")
            word_times, word_values = streams["words"].read(video)
            modalities = {}
            for m in ("audio", "vision"):
                times, values = streams[m].read(video)
                modalities[m] = select_features(
                    times, values, streams[m].feature_names, manifest[m]
                )
            for i, ((start, end), score) in enumerate(zip(label_times, sentiment, strict=True)):
                if max_clips is not None and len(scores) >= max_clips:
                    break
                audit["original_count"] += 1
                if score == 0:
                    audit["excluded_neutral"] += 1
                    continue
                words = _clip_words(word_times, word_values, start, end)
                if not words:
                    audit["excluded_no_words"] += 1
                    continue
                window = prepare_windows(
                    words, modalities["audio"], modalities["vision"], encoder, manifest
                )
                for m in MODALITIES:
                    if window["features"][m].shape[-1] != dims[m]:
                        raise ValueError(f"{m} width differs from the feature manifest")
                    window["features"][m].tofile(handles[m])
                    window["masks"][m].tofile(masks[m])
                    audit["missing_modality_clips"][m] += int(not window["masks"][m].any())
                count = len(window["durations"])
                offsets.append(offsets[-1] + count)
                durations.extend(window["durations"].tolist())
                intervals.extend(window["intervals"].tolist())
                scores.append(float(score))
                clips["ids"].append(f"{video}::segment:{i}")
                clips["video_ids"].append(video)
                clips["raw_text"].append(" ".join(w.text for w in words))
            if max_clips is not None and len(scores) >= max_clips:
                break
            if len(scores) and len(scores) // 500 != (len(scores) - len(sentiment)) // 500:
                print(f"Preparing {split}: {len(scores)} clips, {offsets[-1]} windows", flush=True)
    if not scores or set(np.sign(scores).tolist()) != {-1.0, 1.0}:
        raise ValueError(f"{split} must contain both classes after filtering")
    write_json(folder / "clips.json", clips)
    for name, value, dtype in [
        ("offsets", offsets, np.int64),
        ("durations", durations, np.float32),
        ("intervals", intervals, np.float64),
        ("scores", scores, np.float32),
        ("labels", np.array(scores) > 0, np.int64),
    ]:
        np.save(folder / f"{name}.npy", np.array(value, dtype=dtype), allow_pickle=False)
    audit.update(
        eligible_count=len(scores),
        window_count=offsets[-1],
        class_counts={
            "negative": int((np.array(scores) < 0).sum()),
            "positive": int((np.array(scores) > 0).sum()),
        },
    )
    return audit, dims


def prepare_csd(
    source: Path,
    output: Path,
    manifest: Path,
    splits: Path,
    *,
    encoder=None,
    model_path: Path | None = None,
    device="cpu",
    max_clips: int | None = None,
) -> dict:
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Prepared directory already exists: {output}")
    if max_clips is not None and max_clips < 2:
        raise ValueError("max_clips must be at least two and is only for smoke checks")
    manifest, folds_document = read_json(manifest), read_json(splits)
    folds = folds_document["splits"]
    if set(folds) != set(SPLITS) or any(
        not isinstance(folds[s], list) or not folds[s] for s in SPLITS
    ):
        raise ValueError("All three nonempty video folds are required")
    all_videos = [v for s in SPLITS for v in folds[s]]
    if len(set(all_videos)) != len(all_videos):
        raise ValueError("Source video overlap or duplicate in splits")
    with ExitStack() as stack:
        streams = {
            key: stack.enter_context(CSDStream(source / filename, manifest["roots"][key]))
            for key, filename in manifest["files"].items()
        }
        if set(streams) != {"audio", "vision", "words", "labels"}:
            raise ValueError("Four CSD streams are required: audio, vision, words, labels")
        for m in ("audio", "vision"):
            expected = manifest[m].get("source_columns_sha256")
            if expected is None:
                expected = fingerprint(manifest[m]["source_columns"])
            if fingerprint(streams[m].feature_names) != expected:
                raise ValueError(f"{m} source column order differs from the manifest")
        if (
            streams["words"].feature_names != ["word"]
            or "sentiment" not in streams["labels"].feature_names
        ):
            raise ValueError("Unrecognized word or sentiment columns")
        hashes = {}
        for key, filename in manifest["files"].items():
            hashes[key] = sha256_file(source / filename)
            record = source / (filename + ".download.json")
            if record.exists():
                recorded = read_json(record)
                if (
                    recorded["sha256"] != hashes[key]
                    or recorded["bytes"] != (source / filename).stat().st_size
                ):
                    raise ValueError(f"Downloaded CSD file changed: {filename}")
        if encoder is None:
            from .text_features import BertWordEncoder

            if model_path is None:
                raise ValueError("A local pinned BERT model directory is required")
            encoder = BertWordEncoder(model_path, device=device)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
        try:
            audits, dims = {}, None
            for split in SPLITS:
                audits[split], dims = _write_split(
                    split, folds[split], streams, staging / split, manifest, encoder, max_clips
                )
            normalization = {}
            for split in SPLITS:
                folder = staging / split
                n = audits[split]["window_count"]
                for m in MODALITIES:
                    shape = (n, manifest["max_words"], dims[m])
                    raw = np.memmap(folder / f"{m}.raw", mode="r", dtype=np.float32, shape=shape)
                    mask = np.memmap(
                        folder / f"{m}_mask.raw", mode="r", dtype=bool, shape=shape[:2]
                    )
                    if split == "train":
                        normalization[m] = _moments(raw, mask, np.arange(n))
                    target = np.lib.format.open_memmap(
                        folder / f"{m}.npy", mode="w+", dtype=np.float32, shape=shape
                    )
                    for start in range(0, n, 128):
                        target[start : start + 128] = normalize_features(
                            raw[start : start + 128], mask[start : start + 128], normalization[m]
                        )
                    np.save(folder / f"{m}_mask.npy", mask, allow_pickle=False)
                    target.flush()
                    del raw, mask, target
                    (folder / f"{m}.raw").unlink()
                    (folder / f"{m}_mask.raw").unlink()
            audit = {
                "cache_version": 2,
                "dataset": "CMU-MOSEI",
                "language": "English",
                "source_hashes": hashes,
                "source_sha256": fingerprint(hashes),
                "feature_dims": dims,
                "feature_manifest": manifest,
                "label_rule": LABEL_RULE,
                "splits": audits,
                "folds_source": folds_document,
                "video_overlap": [],
                "smoke_max_clips_per_split": max_clips,
                "word_assignment": "midpoint in [label_start,label_end), bounds clipped; silence removed",
                "normalization": "finite valid non-neutral training tokens only; nonfinite imputed with training mean",
                "window_aggregation": "duration-weighted mean logits before loss/softmax",
            }
            audit["fingerprint"] = fingerprint({"audit": audit, "normalization": normalization})
            write_json(staging / "normalization.json", normalization)
            write_json(staging / "audit.json", audit)
            staging.rename(output)
        except BaseException:
            shutil.rmtree(staging)
            raise
    return audit
