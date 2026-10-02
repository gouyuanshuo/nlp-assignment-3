import subprocess
import sys

import pytest
from scipy.io import wavfile

from mosei.video_media import extract_audio, probe_video, run_command


def make_media(path, duration=1, audio=True):
    args = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=64x64:r=10",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=300:sample_rate=16000"]
    args += [
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(args, check=True, capture_output=True)
    return path


def test_real_video_probe_and_audio_conversion(tmp_path):
    path = make_media(tmp_path / "with space;video.mp4")
    info = probe_video(path, max_seconds=1, max_bytes=path.stat().st_size)
    assert info.duration == 1
    assert (info.width, info.height) == (64, 64)
    wav = extract_audio(path, tmp_path / "audio.wav")
    rate, samples = wavfile.read(wav)
    assert rate == 16000 and samples.ndim == 1
    assert 15000 <= len(samples) <= 17000


def test_duration_size_and_missing_audio_fail_before_extraction(tmp_path):
    path = make_media(tmp_path / "sample.mp4")
    with pytest.raises(ValueError, match="duration|seconds"):
        probe_video(path, max_seconds=0.5)
    with pytest.raises(ValueError, match="size|MB"):
        probe_video(path, max_bytes=path.stat().st_size - 1)
    with pytest.raises(ValueError, match="audio"):
        probe_video(make_media(tmp_path / "silent.mp4", audio=False))


def test_corrupt_and_unsupported_media_are_clear_errors(tmp_path):
    path = tmp_path / "corrupt.mp4"
    path.write_text("this is not a video")
    with pytest.raises(ValueError, match="read|video"):
        probe_video(path)
    path = tmp_path / "video.txt"
    path.write_text("text")
    with pytest.raises(ValueError, match="MP4|MOV|WebM"):
        probe_video(path)


def test_timeout_is_bounded_and_explained():
    with pytest.raises(ValueError, match="timed out"):
        run_command(
            [sys.executable, "-c", "import time;time.sleep(5)"],
            timeout=0.05,
            label="Test extractor",
        )
