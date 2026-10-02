# NLP Assignment 3: English video sentiment with Transformer fusion

Upload an English video, extract text, voice and facial features, and classify its sentiment with a compact multimodal Transformer. Text-only and pooled-concatenation baselines use the same prepared data and training protocol.

**Current status:** the real video pipeline and local upload API work with a small integration checkpoint. The four selected CSD streams (28.4 GB) are downloaded in local `data/csd/`. Full research training is prepared for **Google Colab** and has not been run there yet. The initial real upload took 121.4 seconds for a 15-second clip; a subsequent API upload took 98.1 seconds. These are extraction/inference measurements, not model-training times. See [verification evidence](docs/video-prototype-verification.md).

**Task:** negative (`y < 0`) versus positive (`y > 0`). Exactly neutral clips are excluded. The model has no neutral class, and its softmax probabilities are not calibrated measures of a person's internal emotion.

## Train in Google Colab

All local code and data stay in `/home/gys/Projects/nlp-assignment-3`.

1. Open [01_download_csd.ipynb](notebooks/01_download_csd.ipynb) in Colab on a **CPU runtime**. Upload `dist/mosei-colab-source.zip` when prompted and mount your Drive. It downloads about **28.4 GB** of selected CSD streams into `MyDrive/NLP_Assignment_3/csd`; interrupted transfers resume.
2. Open [02_train_colab.ipynb](notebooks/02_train_colab.ipynb), select an available GPU (A100 preferred), and mount the same Drive. The notebook copies the source to runtime storage, regenerates frozen BERT words and saves a reusable prepared cache to Drive.
3. Run the timing cell and enter the **units/hour shown in your Colab resource panel** and your remaining balance. It measures one epoch on the allocated GPU before estimating the matrix. Download, preparation, evaluation and idle time are separate costs; 200 units do not represent a fixed number of GPU hours. Availability and limits vary ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).
4. Run the frozen training matrix. Each completed epoch is checkpointed to Drive. Rerunning the same cells resumes after a disconnect; a partial epoch repeats. The default is six configurations × three seeds; `INCLUDE_ABLATIONS = False` runs the three core models first, but use a new matrix directory if changing a matrix already started.
5. Once modeling choices are fixed, evaluate held-out data and export the Transformer checkpoint selected by **validation** performance. Extract the exported `selected-run/` folder into local `runs/colab/`, then launch the demo below.

The notebooks require enough Drive space for source, prepared arrays and runs, and at least 55 GB free runtime disk during initial preparation. They have been checked locally but not executed in an authenticated Colab session. The bundle contains source and pinned feature manifests, not datasets, weights or credentials. Build it again after an intentional source update:

```bash
.venv/bin/python scripts/build_colab_bundle.py
```

Keep the local extractor bundle that produced the feature manifest. Changing feature code, model versions or normalization requires compatible new preparation and training.

## Run the video prototype locally

The existing checkout has the pinned extractors/models installed. After importing the Colab checkpoint:

```bash
cd /home/gys/Projects/nlp-assignment-3
.venv/bin/python -m mosei doctor --run runs/colab/selected-run --tools data/tools/video-tools.json
.venv/bin/python -m mosei serve --run runs/colab/selected-run --tools data/tools/video-tools.json
```

Open **http://127.0.0.1:7860**. Upload an English MP4, MOV or WebM, up to 60 seconds and 200 MB, with one visible speaker. The app shows a preview, processing progress, transcript, positive/negative probabilities and predictions for each speech window. Silent input, insufficient face/voice coverage and incompatible checkpoints produce errors with cleared results. Processing stays local; each analysis owns a temporary copy until completion, and the preview cache expires within about an hour.

For a local pipeline demonstration before full training, replace the run path with `runs/video-integration-v2/transformer-seed1111`. This tiny model is explicitly marked as an integration checkpoint. It is **not suitable for accuracy claims**. New uploads were exercised through the real Gradio API; visual browser review is still pending because no browser was connected to the coding session.

A command-line upload is also available:

```bash
.venv/bin/python -m mosei video --run runs/colab/selected-run \
  --tools data/tools/video-tools.json --input /path/to/clip.mp4 --output results/upload.json
```

## Reproduce setup and data preparation

Python 3.12 and dependencies are pinned through `.python-version` and `uv.lock`. The local upload pipeline requires Docker, FFmpeg and FFprobe. Colab training uses the released acoustic/visual streams and does not need Docker.

```bash
uv sync --frozen --extra dev --extra video
source .venv/bin/activate
python scripts/setup_video_tools.py
python scripts/download_splits.py
python scripts/download_csd.py
python -m mosei prepare-csd --source data/csd --output data/prepared-csd \
  --tools data/tools/video-tools.json --splits data/csd/splits.json
python -m pytest -q
```

The selected [samarwarsi/cmu-mosei package, version 1](https://www.kaggle.com/datasets/samarwarsi/cmu-mosei/versions/1) contains timestamped features and annotations, not original videos. We use timestamped words, sentiment labels, COVAREP and OpenFace 2 (28,365,272,301 bytes). The complete seven-file package is about 31.6 GB; GloVe, phones and FACET are not model inputs. The downloader validates pinned file sizes, resumes chunks and records locally calculated SHA-256 hashes. Those hashes are not independently published source checksums.

Official source-video folds come from a pinned CMU Multimodal SDK revision, parsed as literals without executing downloaded Python. Their source hash is verified before use. Preparation audits roots, feature columns, interval order, exclusions, missing streams and source-video overlap. Label segments retain their original IDs even when stored out of chronological order. Statistics are fitted on valid, eligible training tokens only; all other splits and uploads use those saved statistics.

| Modality | Shared representation |
|---|---|
| Text | 768 frozen BERT-base-uncased dimensions; final wordpieces averaged per word |
| Audio | 73 named COVAREP channels; mono 16 kHz input and 10 ms bins |
| Vision | 40 OpenFace 2 channels: gaze angles, head rotations, AU intensities/presence; success and confidence ≥ 0.8 |

The MATLAB-toolbox-dependent `creak` channel is omitted on both paths. Native COVAREP extraction uses pinned upstream algorithms and tested Octave compatibility adapters. Camera-dependent landmark coordinates and head translations are omitted from vision. Historical CSD extractor commits/settings are not published: matching names, units and processing conventions does **not** establish numerical identity or identical train/upload distributions. Automatic transcripts introduce another difference from the released words. These limitations belong in the report.

Overlap-weighted alignment produces word-level streams. All windows are retained (at most 50 words and 512 BERT tokens each). The duration-weighted mean of window logits is computed **before clip loss and softmax**, so every original clip contributes one label and one evaluation record. Cache version 2 stores offsets, masks, durations and fingerprints for this protocol. Every prepared array and clip-metadata file is hashed and verified before reuse; labels are checked against preserved sentiment scores. Interrupted cache extraction is staged before an atomic rename. Incomplete Drive publication is retried even when the local cache already exists.

## Train the models

```bash
python -m mosei train --data data/prepared-csd --config configs/text.toml --output runs/text-seed1111
python -m mosei train --data data/prepared-csd --config configs/concat.toml --output runs/concat-seed1111
python -m mosei train --data data/prepared-csd --config configs/transformer.toml --output runs/transformer-seed1111
```

| Config | Model |
|---|---|
| `text.toml` | Masked mean of cached text → MLP → two logits |
| `concat.toml` | Per-modality linear projections → masked means → concatenation → MLP |
| `transformer.toml` | Projections + modality/position embeddings → joint self-attention → CLS → MLP |
| `no_modality.toml` | Transformer without modality embeddings |
| `no_positions.toml` | Transformer without positional encoding |
| `mean_pooling.toml` | Transformer with masked mean pooling instead of CLS pooling |

The Transformer starts with width 256, 2 layers, 4 heads, feed-forward width 512 and dropout 0.1. Positions restart within each aligned stream. They preserve the release's token order; equal embedding widths alone do not create temporal alignment. Each Transformer layer is initialized independently.

All models use cross-entropy, AdamW, gradient clipping and identical default training budgets: learning rate 0.0003, batch size 32, at most 30 epochs, patience 6. The saved checkpoint maximizes **validation macro F1**, breaking ties with lower validation cross-entropy. Predictions use argmax. Run directories must be new unless `--resume` is supplied. Resume restores the last completed epoch, optimizer, loader and random-number states; changed configurations or data are rejected.

`--seed 1112`, `--epochs 1` and `--device cpu` override individual training settings. A one-epoch run checks execution; it does not establish the hypotheses. Record validation-based tuning choices and give each model the same tuning budget before freezing the final configurations. Parameter counts and training time are reported so capacity differences remain visible.

## Run the planned experiment matrix

```bash
# Nine core runs: three models × seeds 1111, 1112, 1113.
python scripts/run_experiments.py --data data/prepared-csd --output runs/core-v1

# Eighteen runs: the core models plus all three planned ablations.
python scripts/run_experiments.py --data data/prepared-csd --include-ablations --output runs/full-v1
```

The runner saves a manifest before starting, completes training for every run, then evaluates the fixed checkpoints and builds the report. `--train-only` postpones held-out evaluation; `--configs` and `--seeds` choose an explicit matrix. Use `--resume` to continue an existing frozen matrix. Interrupted evaluations and report exports are preserved before recomputation. Each completed run remains usable if a later run fails; incomplete runs are not treated as completed experiments.

The committed `last.pt` contains current weights, optimizer/RNG state and the best weights/metrics together. Resume repairs `best.pt` from that committed state before continuing.

Run artifacts include `config.json`, `data_audit.json`, epoch-by-epoch `history.json`, `best.pt` and `run.json`. Checkpoints retain model configuration, normalization, source/data fingerprints, software versions, seed and validation selection. The inference CLI checks preprocessing compatibility before prediction.

## Evaluate and export evidence

```bash
python -m mosei evaluate --data data/prepared-csd --run runs/text-seed1111 --robustness
python -m mosei evaluate --data data/prepared-csd --run runs/concat-seed1111 --robustness
python -m mosei evaluate --data data/prepared-csd --run runs/transformer-seed1111 --robustness
python -m mosei report --runs runs/text-seed1111 runs/concat-seed1111 runs/transformer-seed1111 --output results/first-comparison
```

The primary metric is macro F1. Secondary outputs include accuracy, weighted F1, class precision/recall/F1/support, cross-entropy and a confusion matrix. With `--robustness`, each run evaluates 13 conditions:

- Clean data.
- Missing text, audio or vision (one stream at a time).
- Gaussian noise SD 0.1, 0.3 or 0.5 in normalized feature units, separately for each stream.

Noise uses a fixed seed and clip ID, making perturbations identical across models, seeds, ordering and batch sizes. Padding is unchanged. A missing stream is masked; the Transformer retains a valid CLS token. An unavailable pooled stream contributes a zero vector. These are **feature corruption** experiments, not demonstrations of robustness to every kind of recording noise.

Each evaluation saves per-clip CSVs and `metrics.json` in `evaluation/test/` (or `evaluation/valid/` when `--split valid` is specified). Existing evaluation directories are protected from overwrite. Reports contain:

- `results.md`, `summary.csv`, `results.json`: clean and corrupted metrics, degradation from clean, seed mean/sample SD, parameters and runtime.
- `paired_differences.csv`: matched model differences with 95% source-video cluster bootstrap intervals.
- `*_confusion.png`: mean confusion counts across the provided seeds.
- `error_examples.json`: clips corrected or newly misclassified by one model relative to another.

Pairing requires identical IDs, annotations, videos, protocols and seed sets. Bootstrap intervals measure variation over source videos conditional on fitted runs; the separate sample SD measures variation across training seeds. With only one seed, SD is explicitly unavailable. A negative `macro_f1_drop` means a corrupted condition happened to score higher than clean data.

## Earlier aligned-feature experiment

The initial implementation used the separate MMSA `aligned_50.pkl` package (768/74/35 dimensions), whose [published hash](https://github.com/thuiar/MMSA#2-datasets) is verified before pickle loading. Its cached-example workflow is retained for reproducibility:

```bash
python scripts/download_features.py
python -m mosei prepare --source data/raw/aligned_50.pkl --output data/prepared
python -m mosei demo --data data/prepared --run runs/real-data-smoke-20261002/transformer-seed1111 --index 0
```

See [original experiment evidence](docs/verification.md). Those cache/checkpoint files use another feature representation and cannot serve uploaded videos. Their tiny training timings do not measure the larger CSD preparation or full video pipeline.

## Project files

- `src/mosei/`: data preparation, shared features, video extraction, models, training, evaluation and app.
- `configs/`: three core classifiers, three ablations and the feature contract template.
- `notebooks/`: Colab download and training handoff.
- `scripts/` and `tools/video/`: reproducible downloads, native tool setup, source bundling and experiments.
- `tests/`: behavioral tests, real media fixtures and interruption/recovery checks.
- `docs/`: design, implementation decisions, evidence and `contributions.csv`.
- `data/`, `runs/`, `results/`, `dist/`: ignored local datasets, checkpoints and deliverables.

The proposal remains in the course Assignment 3 `Plan` folder; source code stays in Projects. Group members should record actual owned tasks, commits and experiments in `docs/contributions.csv`. Use saved research results for the final report and presentation.

Method references: [CMU-MOSEI](https://aclanthology.org/P18-1208/), [MulT](https://aclanthology.org/P19-1656/), [modality robustness](https://aclanthology.org/2022.naacl-main.50/). This concatenated-sequence Transformer is an application implementation, not an exact reproduction of MulT.
