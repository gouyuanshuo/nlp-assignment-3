"""Bounded local media inspection and conversion, with no shell interpolation."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    size_bytes: int


def run_command(args, *, timeout, label, cwd=None):
    try:
        result = subprocess.run(
            [str(a) for a in args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{label} timed out; try a shorter video.") from exc
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is not installed. Run the prototype readiness check.") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1200:]
        raise ValueError(f"{label} could not read/process the video: {detail}")
    return result.stdout


def probe_video(path: Path, *, max_seconds=60, max_bytes=200_000_000) -> MediaInfo:
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() not in {".mp4", ".mov", ".webm"}:
        raise ValueError("Please upload an MP4, MOV or WebM video.")
    size = path.stat().st_size
    if not 0 < size <= max_bytes:
        raise ValueError(f"Video size must be between 1 byte and {max_bytes / 1e6:g} MB.")
    output = run_command(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
        timeout=20,
        label="Video inspection",
    )
    try:
        metadata = json.loads(output)
        videos = [
            s
            for s in metadata["streams"]
            if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")
        ]
        audios = [s for s in metadata["streams"] if s["codec_type"] == "audio"]
        if not videos or not audios:
            raise ValueError(
                "The video must contain a video stream and an audio stream with speech."
            )
        duration = float(metadata["format"]["duration"])
        width, height = int(videos[0]["width"]), int(videos[0]["height"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read video duration and stream information.") from exc
    if not np.isfinite(duration) or not 0 < duration <= max_seconds:
        raise ValueError(
            f"Video duration must be greater than zero and at most {max_seconds:g} seconds."
        )
    if min(width, height) < 32 or max(width, height) > 4096:
        raise ValueError("Use a video between 32 and 4096 pixels on each side.")
    return MediaInfo(duration, width, height, size)


def extract_audio(video: Path, output: Path) -> Path:
    run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            Path(video).resolve(),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "aresample=16000:async=1:first_pts=0",
            "-c:a",
            "pcm_s16le",
            output,
        ],
        timeout=90,
        label="Audio extraction",
    )
    return Path(output)


def canonical_video(video: Path, output: Path) -> Path:
    # A bounded 30 Hz image stream; fps also fills leading timestamp gaps from t=0.
    run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            Path(video).resolve(),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "fps=30:start_time=0,scale=w='min(960,iw)':h=-2",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-threads",
            "2",
            output,
        ],
        timeout=120,
        label="Video frame preparation",
    )
    return Path(output)
