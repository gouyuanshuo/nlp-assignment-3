"""Checkpoint-bound inference from a new local video, with disposable intermediates."""

import importlib.metadata
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .io import fingerprint, read_json, sha256_file
from .text_features import BertWordEncoder
from .train import choose_device, restore_run
from .video_extractors import extract_covarep, extract_openface, transcribe_audio
from .video_features import prepare_windows
from .video_media import extract_audio, probe_video, run_command
from .window_data import clip_logits


def ensure_compatible(metadata, manifest):
    saved = metadata.get("feature_manifest")
    if saved is None:
        raise ValueError(
            "This checkpoint has no video feature manifest. Train on the new CSD cache before serving uploads."
        )
    if fingerprint(saved) != fingerprint(manifest):
        raise ValueError(
            "The checkpoint and installed video tools have different feature manifests. Use the matching tool bundle or retrain."
        )


def verify_models(manifest, paths, sections=("text", "speech")):
    for section in sections:
        for filename, expected in manifest[section]["files_sha256"].items():
            path = Path(paths[section]) / filename
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Pinned {section} model file is missing or changed: {filename}")


def check_tools(tools: Path, *, native=True):
    record = read_json(tools)
    manifest = record["manifest"]
    if fingerprint(manifest) != record["fingerprint"]:
        raise ValueError("Video tool manifest fingerprint mismatch")
    for package, expected in manifest["software"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Installed {package} differs from the feature manifest ({expected}).")
    for filename, expected in manifest["preprocessing_sha256"].items():
        if sha256_file(Path(__file__).parent / filename) != expected:
            raise ValueError(
                f"Preprocessing code changed: {filename}. Regenerate the tool manifest and prepared data."
            )
    verify_models(manifest, record["paths"], sections=("text", "speech") if native else ("text",))
    if native:
        ffmpeg = run_command(["ffmpeg", "-version"], timeout=10, label="FFmpeg").splitlines()[0]
        if ffmpeg != manifest["ffmpeg_version"]:
            raise ValueError("FFmpeg version differs from the feature manifest.")
        run_command(["ffprobe", "-version"], timeout=10, label="FFprobe")
        for m in ("audio", "vision"):
            actual = run_command(
                ["docker", "image", "inspect", manifest[m]["image"], "--format", "{{.Id}}"],
                timeout=20,
                label=f"{m} extractor readiness",
            ).strip()
            if actual != manifest[m]["image"]:
                raise ValueError(f"The pinned {m} extraction image is unavailable.")
    return record


@torch.inference_mode()
def infer_windows(model, windows, device):
    model.eval()
    features = {
        m: torch.from_numpy(np.asarray(a)).to(device) for m, a in windows["features"].items()
    }
    masks = {m: torch.from_numpy(np.asarray(a)).to(device) for m, a in windows["masks"].items()}
    logits = model(features, masks)
    if not torch.isfinite(logits).all():
        raise ValueError("The model produced a non-finite result for this video.")
    merged = clip_logits(
        logits, torch.zeros(len(logits), dtype=torch.long), torch.as_tensor(windows["durations"]), 1
    )
    probabilities = merged.softmax(-1)[0].cpu().tolist()
    names = ["negative", "positive"]
    rows = []
    for i, probability in enumerate(logits.softmax(-1).cpu().tolist()):
        rows.append(
            {
                "start": float(windows["intervals"][i, 0]),
                "end": float(windows["intervals"][i, 1]),
                "text": windows["texts"][i],
                "prediction": names[int(np.argmax(probability))],
                "prob_negative": probability[0],
                "prob_positive": probability[1],
            }
        )
    return {
        "prediction": names[int(np.argmax(probabilities))],
        "probabilities": dict(zip(names, probabilities, strict=True)),
        "windows": rows,
    }


class VideoClassifier:
    def __init__(self, run: Path, tools: Path, device="auto", *, temp_root=None):
        record = check_tools(tools)
        self.manifest = record["manifest"]
        self.device = choose_device(device)
        self.model, self.metadata = restore_run(run, self.device)
        ensure_compatible(self.metadata, self.manifest)
        self.encoder = BertWordEncoder(record["paths"]["text"], device=self.device)
        from faster_whisper import WhisperModel

        self.speech = WhisperModel(
            record["paths"]["speech"],
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            local_files_only=True,
        )
        self.temp_root = temp_root

    def classify(self, video: Path, progress=None):
        notify = progress if progress is not None else lambda fraction, message: None
        started = time.perf_counter()
        info = probe_video(video)
        with tempfile.TemporaryDirectory(prefix="mosei-video-", dir=self.temp_root) as temporary:
            workdir = Path(temporary)
            # Gradio's preview cache can expire while CPU feature extraction runs.
            # Own the input for the entire request, alongside its other intermediates.
            local_video = workdir / ("input" + Path(video).suffix.lower())
            shutil.copyfile(video, local_video)
            video = local_video
            notify(0.05, "Reading audio")
            audio_file = extract_audio(video, workdir / "audio.wav")
            notify(0.15, "Recognizing speech")
            words = transcribe_audio(audio_file, self.speech)
            notify(0.35, "Analyzing vocal expression")
            audio = extract_covarep(audio_file, workdir, self.manifest["audio"])
            notify(0.6, "Analyzing facial expression")
            vision = extract_openface(video, workdir, self.manifest["vision"])
            notify(0.85, "Combining speech, voice and expression")
            windows = prepare_windows(
                words, audio, vision, self.encoder, self.manifest, self.metadata["normalization"]
            )
            coverage = {
                m: float(windows["masks"][m].sum() / windows["masks"]["text"].sum())
                for m in ("audio", "vision")
            }
            for m in coverage:
                if coverage[m] < self.manifest[m]["min_word_coverage"]:
                    raise ValueError(
                        f"Not enough usable {'voice' if m == 'audio' else 'face'} information during speech. Try a clearer clip."
                    )
            result = infer_windows(self.model, windows, self.device)
            result.update(
                transcript=" ".join(w.text for w in words),
                words=[{"text": w.text, "start": w.start, "end": w.end} for w in words],
                duration_seconds=info.duration,
                processing_seconds=time.perf_counter() - started,
                smoke_checkpoint=self.metadata.get("smoke_checkpoint", False),
                coverage=coverage,
                feature_fingerprint=fingerprint(self.manifest),
                task="Positive versus negative sentiment; neutral excluded from training",
            )
            notify(1, "Analysis complete")
            return result
