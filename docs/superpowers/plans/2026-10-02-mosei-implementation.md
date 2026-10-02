# CMU-MOSEI Transformer Fusion Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. Track completion below and in `docs/implementation-log.md`.

**Goal:** Deliver a runnable, reproducible English sentiment experiment and CLI demo implementing the approved Assignment 3 proposal.

**Architecture:** One audited data cache supplies text-only, pooled concatenation, and sequence Transformer classifiers. Shared train-only normalization, training, evaluation, corruption, and reporting code enforce fair comparisons. A CLI exposes preparation, training, evaluation, paired reporting, and a held-out clip demo.

**Tech stack:** Python 3.12, PyTorch, NumPy, scikit-learn, matplotlib, pytest; TOML configurations; JSON/CSV outputs.

**Spec:** `/home/gys/Course/2026-Spring-42173-Advanced-Natural-Language-Processing/04_Assignments/Assignment_03_Cross_Modal_Conflict/Plan/Assignment3_Transformer_Fusion_CMU_MOSEI_Project_Plan.docx` (approved by the user's request to implement it).

## Global constraints

- English CMU-MOSEI only. MMSA aligned_50.pkl SHA-256: `45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791`.
- Preserve released train/valid/test partitions and clip IDs. Negative = y < 0, positive = y > 0; exclude y = 0 from every split, recording counts.
- Cached frozen text/audio/vision features, widths 768/74/35 for this release. Preserve documented masks; never infer text padding from BERT vectors being zero.
- Fit feature normalization using valid training tokens only. Test labels never select checkpoints or configurations.
- Core Transformer: width 256, two layers, four heads, FFN 512, dropout 0.1, modality embeddings, position encoding, CLS pooling, two-logit classifier.
- Every classifier uses cross-entropy. Checkpoints selected by validation macro F1, then validation cross-entropy; argmax prediction.
- Core seeds 1111/1112/1113. Matched missing-modality and Gaussian noise SD 0.1/0.3/0.5 conditions; padding unchanged.
- Report macro F1, accuracy, weighted F1, per-class precision/recall, confusion matrices, runtime, parameter count, run variability and paired source-video bootstrap differences.
- Core ablations: no modality embeddings, no positions, masked-mean pooling. Optional published comparators and raw-media encoders are outside this implementation.
- Generated data, checkpoints and private transcripts stay out of version control. Synthetic fixtures are clearly labelled software checks, never experimental evidence.

## Review focus

- Metadata can disagree with feature lengths; preparation must reject ambiguous masks instead of silently treating padding as data (Task 1).
- Non-finite audio values occur in released features; count and replace affected feature values using training-only statistics while leaving padding zero (Task 1).
- Fully missing text or other streams must give finite logits and be insensitive to their hidden values (Task 2).
- Evaluations with different batching must apply identical corruption to each clip; pairing must fail on mismatched IDs/labels/protocols (Task 4).
- A rerun must not silently overwrite a trained run, and evaluation must restore preprocessing/configuration from the checkpoint (Tasks 3–5).

## Task 1: Audited dataset preparation

**Files:** `src/mosei/data.py`, `src/mosei/io.py`, `tests/test_data.py`, `tests/conftest.py`, `pyproject.toml`, `.gitignore`.

**Interfaces:** `prepare_data(source: Path, output: Path, expected_sha256: str) -> dict`; `load_data(directory: Path) -> PreparedData`; `PreparedData.splits` contains `FeatureDataset` with tensors, boolean valid masks, binary labels, IDs, video IDs, raw text and original labels. Prepared arrays saved without pickle, plus audit/normalization metadata.

- [x] Write and run failing tests for neutral exclusion, train-only normalization, metadata masks, padding, non-finite values and duplicate/cross-split video IDs.
- [x] Implement strict aligned-release schema validation, streaming SHA-256 verification before pickle loading, source-video audit, per-feature finite moments and safe prepared cache.
- [x] Verify: `.venv/bin/python -m pytest tests/test_data.py -q` → all pass.
- [x] Record evidence and commit the independently usable preparation layer.

## Task 2: Three models and ablation switches

**Files:** `src/mosei/models.py`, `tests/test_models.py`.

**Interfaces:** `ModelConfig`; `build_model(config: ModelConfig, feature_dims: dict[str,int]) -> nn.Module`; model forward accepts modality tensor dictionary and valid-mask dictionary and returns [batch,2] logits.

- [x] Write/run failing tests for padding invariance, stream removal, gradients and ablation execution across all models.
- [x] Implement masked mean, text MLP, pooled projected-concatenation MLP, and independently initialized Transformer layers with CLS, sinusoidal positions and optional modality embeddings.
- [x] Verify: `.venv/bin/python -m pytest -q` → all pass, including finite loss/gradients under every missing stream.
- [x] Record evidence and commit model layer.

## Task 3: Reproducible training and checkpoint restoration

**Files:** `src/mosei/config.py`, `src/mosei/train.py`, `src/mosei/metrics.py`, `configs/*.toml`, `tests/test_training.py`.

**Interfaces:** `ExperimentConfig`; `train_run(data: PreparedData, config: ExperimentConfig, output: Path) -> dict`; `restore_run(run: Path, device: str) -> (model, metadata)`; `predict(model, dataset, batch_size, device, condition) -> prediction records`.

- [x] Write/run failing tests for validation tie-breaks, repeatable toy training, checkpoint round-trip, train/valid-only access, and overwrite rejection.
- [x] Implement deterministic seeds, AdamW, clipping, early stopping, saved best checkpoint, per-epoch logs and provenance; expose CPU/CUDA/auto.
- [x] Verify: `.venv/bin/python -m pytest -q` → all pass, including a tiny real train/reload cycle.
- [x] Record evidence and commit training layer.

## Task 4: Matched robustness and paired reporting

**Files:** `src/mosei/evaluate.py`, `src/mosei/report.py`, `tests/test_evaluation.py`.

**Interfaces:** `Condition`; `corrupt_batch(features, masks, ids, condition) -> (features,masks)`; `evaluate_run(data, run, split, robustness) -> dict`; `summarize_runs(run_dirs, output, bootstrap_samples) -> dict`.

- [x] Write/run failing tests for per-ID/batch-invariant corruption, untouched padding, literal confusion/F1 values, source-video bootstrap and pairing guards.
- [x] Implement clean + 12 corruption conditions, per-clip predictions, degradation from clean, CSV/JSON/PNG exports, seed mean/std and paired video-cluster intervals.
- [x] Verify: `.venv/bin/python -m pytest -q` → all pass with missing modality and noise reproducible across models.
- [x] Record evidence and commit evaluation/report layer.

## Task 5: CLI, experiment matrix, demo and documented run

**Files:** `src/mosei/cli.py`, `src/mosei/__main__.py`, `scripts/download_features.py`, `scripts/run_experiments.py`, `README.md`, `tests/test_cli.py`.

**Interfaces:** `python -m mosei {prepare,train,evaluate,report,demo}`. Core/ablation configs share training defaults. Matrix script runs chosen configs/seeds and evaluates only after training/selection. Demo compares a chosen held-out clip with text-only baseline.

- [x] Write/run an end-to-end CLI test against a clearly synthetic MMSA-format fixture: prepare → train → reload → robustness evaluation → report → demo.
- [x] Implement CLI and safe download helper, pin dependencies, document data source/feature provenance/label rule and exact commands.
- [x] Download and audit the actual feature release if accessible; run real-data train/validation and demo smoke checks on available GPU. Record experimental scope honestly.
- [x] Verify: full pytest suite, ruff, package install, CLI smoke, CUDA availability and actual-data checks when available.
- [x] Request one independent final code review under the executing-plans skill, fix consequential findings with regression tests, and record final verification.

## Execution decisions

- The user approved the written proposal and explicitly requested implementation. Continue directly in this session; no duplicate design approval is required.
- No enclosing Git repository exists. Use `/home/gys/Projects/nlp-assignment-3`; the user requires code in Projects, outside Course.
- The isolated Python environment is `.venv` in the project root.
- Full report prose and presentation require observed experiment results and group contributions; this software generates their evidence, not invented outcomes.
