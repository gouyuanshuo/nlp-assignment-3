"""Local speech, COVAREP and OpenFace adapters with explicit output schemas."""

import csv
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np

from .csd_prepare import select_features
from .video_features import Word
from .video_media import canonical_video, run_command


def transcribe_audio(audio: Path, model) -> list[Word]:
    from scipy.io import wavfile

    rate, samples = wavfile.read(audio)
    if rate != 16000 or samples.ndim != 1 or samples.dtype != np.int16:
        raise ValueError("Speech recognition requires mono 16 kHz signed 16-bit audio.")
    signal = samples.astype(np.float32) / 32768
    if len(signal) < rate // 4 or np.sqrt(np.mean(signal**2)) < 1e-5:
        raise ValueError("No speech detected: the audio is silent or too short.")
    # Decode with our checked WAV contract. Passing arrays also avoids PyAV
    # decoder-version differences affecting channel order or sample rate.
    segments, _ = model.transcribe(
        signal,
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    words = []
    for segment in segments:
        for w in segment.words or []:
            text = w.word.strip()
            if text and 0 <= w.start < w.end <= len(signal) / rate + 0.1:
                words.append(Word(text, float(w.start), min(float(w.end), len(signal) / rate)))
    if not words:
        raise ValueError(
            "No English speech could be recognized. Use a clip with clear spoken audio."
        )
    return words


def run_container(image, args, mounts, *, label, timeout=600, entrypoint=None):
    name = "mosei-video-" + uuid.uuid4().hex
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--cpus",
        "4",
        "--memory",
        "4g",
        "--pids-limit",
        "256",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    if entrypoint:
        command += ["--entrypoint", entrypoint]
    for source, target, readonly in mounts:
        command += [
            "--mount",
            f"type=bind,source={Path(source).resolve()},target={target}"
            + (",readonly" if readonly else ""),
        ]
    command += [image, *args]
    try:
        return run_command(command, timeout=timeout, label=label)
    finally:
        # Killing a docker CLI on timeout does not stop its container.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20, check=False)


def read_covarep(path: Path, section):
    from scipy.io import loadmat

    data = loadmat(path, simplify_cells=True)
    names = [str(n).strip() for n in np.asarray(data["names"]).reshape(-1)]
    values = np.asarray(data["features"], dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, len(names))
    intervals = np.asarray(data["intervals"], dtype=np.float64).reshape(-1, 2)
    if values.shape[1] != len(names) or not np.isfinite(values).any():
        raise ValueError("Audio extraction returned invalid feature columns.")
    return select_features(intervals, values, names, section)


def extract_covarep(audio, workdir, manifest):
    folder = Path(workdir) / "acoustic"
    folder.mkdir()
    shutil.copyfile(audio, folder / "audio.wav")
    run_container(manifest["image"], [], [(folder, "/work", False)], label="Voice analysis")
    if not (folder / "audio.mat").exists():
        raise ValueError("Voice analysis could not extract acoustic features from this recording.")
    return read_covarep(folder / "audio.mat", manifest)


def read_openface(path: Path, section):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        names = [n.strip() for n in next(reader)]
        values = np.array([[float(v) for v in row] for row in reader], dtype=np.float32)
    if values.ndim != 2 or not len(values) or "timestamp" not in names:
        raise ValueError("No usable face was found in the video.")
    times = values[:, names.index("timestamp")].astype(np.float64)
    # Released CSD stores each frame as [timestamp, next timestamp), not as
    # a point measurement. Use the same support for overlap-weighted pooling.
    step = (
        1 / section["canonical_fps"]
        if "canonical_fps" in section
        else (float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30)
    )
    ends = np.concatenate([times[1:], [times[-1] + step]])
    result = select_features(np.stack([times, ends], axis=1), values, names, section)
    if not np.isfinite(result.values).any():
        raise ValueError("No usable face was found. Use a well-lit clip with one visible speaker.")
    return result


def extract_openface(video, workdir, manifest):
    folder = Path(workdir) / "face"
    folder.mkdir()
    canonical = canonical_video(video, Path(workdir) / "frames.mp4")
    run_container(
        manifest["image"],
        [
            "-f",
            "/input/video.mp4",
            "-out_dir",
            "/output",
            "-of",
            "features",
            "-pose",
            "-aus",
            "-gaze",
        ],
        [(canonical, "/input/video.mp4", True), (folder, "/output", False)],
        entrypoint=manifest["executable"],
        label="Facial expression analysis",
    )
    if not (folder / "features.csv").exists():
        raise ValueError("Facial expression analysis did not produce features.")
    return read_openface(folder / "features.csv", manifest)
