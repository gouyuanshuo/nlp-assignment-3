# Video Upload Prototype Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. Track evidence and decisions in the plan workspace and `docs/video-prototype-progress.md`.

**Goal:** Classify a newly uploaded English video with a trained CMU-MOSEI multimodal Transformer in a local browser interface.

**Architecture:** A CSD importer and live video extractors produce the same timestamped streams, which share alignment, windowing and normalization code. Training aggregates window logits per source clip. Saved feature manifests constrain inference; the UI loads a compatible trained checkpoint and invokes the real extraction pipeline.

**Tech stack:** Existing Python 3.12/PyTorch project; h5py, Transformers/BERT, faster-whisper, FFmpeg, COVAREP/Octave, OpenFace 2, Gradio, pytest. Heavy native tools may run in pinned Docker images.

**Spec:** `docs/superpowers/specs/2026-10-03-video-upload-prototype-design.md`, approved 2026-10-03.

## Global constraints

- Work in `/home/gys/Projects/nlp-assignment-3`; preserve the existing MMSA cache and evidence.
- Preferred source: `samarwarsi/cmu-mosei`, version 1. Distinguish calculated file hashes from independently published hashes.
- English MP4/MOV/WebM, at most 60 seconds and 200 MB, one visible speaker; local processing and localhost UI.
- Binary labels: negative `y < 0`, positive `y > 0`, exact zero excluded. Preserve official video splits and prevent video overlap.
- Frozen BERT word features, COVAREP audio and documented OpenFace 2 features. Do not replace feature families just to satisfy expected shapes.
- At most 50 words per window, with the BERT token limit enforced too. Duration-weighted mean window logits before clip loss/evaluation; each source clip has one label.
- Pin extractor/model versions, column order, units, aggregation rules and normalization. Train-only normalization; validation-only checkpoint selection.
- Actual video extraction and browser upload checks are required for demo readiness. Mock predictions and the old MMSA checkpoint cannot satisfy acceptance.

## Review focus

- Corrupt/incomplete downloads and unknown CSD columns must be rejected before training (Task 1).
- Boundary words, gaps, overlapping measurements and tokenizer splitting must align consistently without dropping late windows (Task 2).
- Long clips must contribute one training example and one evaluation record, regardless of window count (Task 3).
- Invalid, silent, over-limit or faceless media must produce a useful error, without stale results or silent text-only fallback (Task 4/5).
- A valid checkpoint with the wrong extraction manifest must fail before predicting; uploaded media must be cleaned up (Task 5).

## Task 1: Reproducible CSD source and extractor compatibility

**Files:** create `src/mosei/csd.py`, `scripts/download_csd.py`, `scripts/setup_video_tools.py`, `tools/video/`, `tests/test_csd.py`; extend optional dependencies and ignore downloaded tools/data.

**Interfaces:** `CSDStream(path: Path)` exposes `metadata`, `feature_names`, `video_ids`, and `read(video_id: str) -> (intervals: ndarray, features: ndarray)` with schema/time validation. `download_csd(output: Path, files: list[str] | None = None) -> dict` records the pinned catalogue and local hashes. An extraction manifest is a JSON object fingerprinted through `mosei.io.fingerprint`.

- [ ] Write tests using small real HDF5 fixtures: a feature stream exposes ordered columns; invalid intervals and width/name mismatch fail; a word stream decodes UTF-8; unexpected root names fail.
- [ ] Run `.venv/bin/python -m pytest tests/test_csd.py -q` and observe missing-reader failures.
- [ ] Implement the reader and resumable selected-file downloader. Pin the SDK split file at a specific commit and parse literal split lists without executing downloaded code.
- [ ] Inspect real CSD metadata. Install isolated native extractors, process a real spoken clip, and establish their names/units/timing contract. Save versions and compatibility evidence. Stop dependent final training if the chosen representation cannot be reproduced reliably.
- [ ] Run reader tests and baseline suite; record the evidence and commit.

## Task 2: Shared timestamp alignment, windows and BERT

**Files:** create `src/mosei/video_features.py`, `src/mosei/text_features.py`, `tests/test_video_features.py`.

**Interfaces:** `Word(text: str, start: float, end: float)`; `TimedFeatures(intervals: ndarray, values: ndarray, names: list[str])`; `align_words(words, stream) -> (values, valid_mask)`; `make_windows(words, tokenizer, max_words=50) -> list[list[Word]]`; `BertWordEncoder.encode(windows) -> list[ndarray]`; `prepare_windows(words, audio, vision, encoder, manifest, normalization=None) -> dict` returns modality arrays, masks, positive durations and window timestamps.

- [ ] Write tests with hand-calculated interval overlap means, empty intervals, a wordpiece split, 51-word input retaining two windows, and tokenizer limits forcing a split before 50 words.
- [ ] Run `.venv/bin/python -m pytest tests/test_video_features.py -q` and observe the intended failures.
- [ ] Implement overlap-weighted finite-feature aggregation, explicit missing masks, deterministic wordpiece-to-word pooling and shared normalization. Feature-name checks run before alignment.
- [ ] Verify training-style and upload-style equivalent streams produce identical arrays, including padding; run the suite and commit.

## Task 3: CSD preparation and clip-level multi-window training

**Files:** create `src/mosei/csd_prepare.py`, `src/mosei/window_data.py`, `tests/test_window_training.py`; modify `data.py`, `train.py`, `evaluate.py` only where needed for cache dispatch, collation and manifest metadata.

**Interfaces:** `prepare_csd(source: Path, output: Path, manifest: Path, splits: Path) -> dict`; `WindowDataset` returns one clip with variable window count; `collate_clips(items) -> dict` flattens windows and retains clip indices/durations; `clip_logits(window_logits, clip_indices, durations, clip_count) -> Tensor` returns `[clips, 2]`. Existing `load_data`, `train_run`, `predict` and reports accept both the original and new caches.

- [ ] Write tests proving one label/record per clip, exact duration-weighted logits, nonzero gradients through all windows, normalization unaffected by validation outliers, and rejection of source-video overlap.
- [ ] Observe failures, then implement a new cache version with clip offsets, window tensors, manifests and a complete audit. Preserve the original cache loader and CLI behavior.
- [ ] Train on clip-level aggregated logits. Apply matched corruption per window using stable clip/window IDs, then aggregate for evaluation; preserve the public record schema used by reports.
- [ ] Run full tests and a small real-data train/restore check for all three models; record actual counts and commit.

## Task 4: Real uploaded-video extraction

**Files:** create `src/mosei/video_media.py`, `src/mosei/video_extractors.py`, `src/mosei/video_pipeline.py`, `tests/test_video_media.py`, `tests/test_video_pipeline.py`.

**Interfaces:** `probe_video(path: Path) -> MediaInfo`; `extract_audio(video: Path, output: Path) -> Path`; `transcribe(audio: Path, model: str, revision: str) -> list[Word]`; `extract_covarep(audio, workdir, manifest) -> TimedFeatures`; `extract_openface(video, workdir, manifest) -> TimedFeatures`; `VideoClassifier(run: Path, tools: Path, device='auto').classify(video: Path, progress=None) -> dict`.

- [ ] Create FFmpeg-generated media fixtures to assert duration/size/stream validation. Test exact boundary acceptance, corrupt media, timeout and silent speech errors. Test missing face and mismatched manifest through extractor-result fixtures.
- [ ] Observe failures. Implement subprocess argument lists with timeouts, per-request temporary directories, explicit column selection and clear tool errors. Models load locally from pinned cached revisions.
- [ ] Extract all three modalities from a real English spoken video. Compare schemas and preprocessing contracts with the real CSD stream metadata; document remaining train/upload distribution differences.
- [ ] Verify actual checkpoint inference, probability sum and finite outputs, transcript/timestamp preservation and cleanup; run tests and commit.

## Task 5: Browser demo and CLI commands

**Files:** create `src/mosei/web_demo.py`, `tests/test_video_cli.py`; modify `cli.py`, `pyproject.toml`, README and config examples.

**Interfaces:** add `prepare-csd`, `video`, `doctor` and `serve` commands. `build_demo(classifier) -> gradio.Blocks`; `serve --run RUN --tools TOOLS --host 127.0.0.1 --port 7860` starts a local interface with no public sharing.

- [ ] Write tests that a failed readiness check blocks inference; an old MMSA checkpoint is rejected for uploads; returned labels/probabilities/window rows match the classifier response; new requests clear stale results.
- [ ] Observe failures. Implement upload/preview, Analyse, progress, transcript, sentiment probability bars and per-window timestamp rows. Keep tool names and internal manifests out of the normal user flow.
- [ ] Run CLI help and validation tests, then launch the real interface. Upload a spoken video and inspect the successful result; exercise an invalid/silent input and inspect its error. Record actual latency.
- [ ] Run the full suite, format/lint and commit.

## Task 6: Research runs, evidence and proposal update

**Files:** update `scripts/run_experiments.py` only if new cache integration needs it; add `docs/video-prototype-verification.md`; update README, progress, and the existing Assignment 3 DOCX proposal using the document skill.

- [ ] Freeze the new feature manifest/configuration after validation checks. Train the three core models across the existing three seeds, then ablations, using the revised cache.
- [ ] Evaluate fixed checkpoints and generate the existing clean/robustness reports. Keep old smoke measurements separate; report model training and upload extraction latency separately.
- [ ] Write evidence covering dataset provenance, extractor compatibility, sample counts, software tests, new-video verification, recognition errors and observed limitations. Do not report success for an unexecuted stage.
- [ ] Update the proposal for the approved prototype and selected package; render and visually verify it. Keep source code and data in Projects.
- [ ] Run the final required checks and one independent whole-branch review under `superpowers:executing-plans`. Fix consequential findings, record final results, and deliver working launch commands.

## Execution decisions

- The user approved the written design and already requested implementation. Continue the established native execution method rather than asking again for authorization to do this work.
- This is a dedicated local repository on an implementation branch; work in its current Projects location. No remote publish, deployment or shared-branch merge is requested.
- Feature compatibility is a real engineering gate. If it fails, complete independent useful work and report the concrete incompatibility rather than inventing substitute features or calling a mock demo finished.
