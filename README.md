# NLP Assignment 3: multimodal sentiment with Transformer fusion

A PyTorch implementation of the approved CMU-MOSEI proposal. It compares text-only prediction, pooled concatenation and a compact Transformer using the same frozen English multimodal features.

**Task:** negative sentiment (`y < 0`) versus positive sentiment (`y > 0`). Exactly neutral clips (`y == 0`) are excluded from train, validation and test. This model has no neutral class.

## Get started

The local checkout is `/home/gys/Projects/nlp-assignment-3`. Its environment, data, checkpoints and outputs stay in this project.

```bash
cd /home/gys/Projects/nlp-assignment-3
uv sync --frozen --extra dev
source .venv/bin/activate
python -m pytest -q
```

Python 3.12 is pinned in `.python-version`, and `uv.lock` pins dependencies. CUDA is selected automatically when available; every training/evaluation command also accepts `--device cpu`. A Python environment without uv can install the package with `python -m pip install -e '.[dev]'`, though the uv lock is the reproducible setup.

## Download and prepare CMU-MOSEI

The downloader defaults to [this Kaggle mirror, version 1](https://www.kaggle.com/datasets/neffow/cmu-mosei/versions/1). It retrieves only `aligned_50.pkl` and verifies the [MMSA authors' published SHA-256](https://github.com/thuiar/MMSA#2-datasets):

```text
45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791
```

```bash
python scripts/download_features.py
python -m mosei prepare --source data/raw/aligned_50.pkl --output data/prepared
```

The archive is about 4.7 GB. Preparation needs several GB of free RAM and another several GB of disk space; the prepared arrays are memory-mapped during training. Interrupted downloader chunks are retained and reused on rerun. `--source authors` selects the original public Google Drive file. No Kaggle credentials are needed for this public mirror when its download endpoint is available.

Preparation verifies the checksum **before loading the pickle**. It preserves released partitions, excludes neutral clips, audits all IDs (including excluded rows), rejects video overlap, and validates feature dimensions and masks. It fits each feature's mean and standard deviation using finite, valid, eligible training tokens only. Non-finite feature values are counted and imputed with the training mean; normalized padding is exactly zero. A constant feature uses scale 1.

`data/prepared/audit.json` records the source hash, masks, original/eligible counts, class balance, non-finite counts and schema; `normalization.json` holds the fitted statistics. Never use a model-dependent subset as the primary test set. A manually supplied `--sha256` is for a separately trusted archive; the default verifies the exact proposed release.

Cached text has width 768, audio 74 and vision 35. The release documents BERT text features. Audio and visual extractor provenance must be cited from the originating release/metadata, rather than inferred solely from these dimensions. This code starts from cached sequences; it does not run raw-media encoders.

## Train the models

```bash
python -m mosei train --data data/prepared --config configs/text.toml --output runs/text-seed1111
python -m mosei train --data data/prepared --config configs/concat.toml --output runs/concat-seed1111
python -m mosei train --data data/prepared --config configs/transformer.toml --output runs/transformer-seed1111
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

All models use cross-entropy, AdamW, gradient clipping and identical default training budgets: learning rate 0.0003, batch size 32, at most 30 epochs, patience 6. The saved checkpoint maximizes **validation macro F1**, breaking ties with lower validation cross-entropy. Predictions use argmax. Run directories must be new, protecting previous experiment results.

`--seed 1112`, `--epochs 1` and `--device cpu` override individual training settings. A one-epoch run checks execution; it does not establish the hypotheses. Record validation-based tuning choices and give each model the same tuning budget before freezing the final configurations. Parameter counts and training time are reported so capacity differences remain visible.

## Run the planned experiment matrix

```bash
# Nine core runs: three models × seeds 1111, 1112, 1113.
python scripts/run_experiments.py --data data/prepared --output runs/core-v1

# Eighteen runs: the core models plus all three planned ablations.
python scripts/run_experiments.py --data data/prepared --include-ablations --output runs/full-v1
```

The runner saves a manifest before starting, completes training for every run, then evaluates the fixed checkpoints and builds the report. `--train-only` postpones held-out evaluation; `--configs` and `--seeds` choose an explicit matrix. The runner refuses an existing output directory. Each completed run remains usable if a later run fails; incomplete runs are not treated as completed experiments.

Run artifacts include `config.json`, `data_audit.json`, epoch-by-epoch `history.json`, `best.pt` and `run.json`. Checkpoints retain model configuration, normalization, source/data fingerprints, software versions, seed and validation selection. The inference CLI checks preprocessing compatibility before prediction.

## Evaluate and export evidence

```bash
python -m mosei evaluate --data data/prepared --run runs/text-seed1111 --robustness
python -m mosei evaluate --data data/prepared --run runs/concat-seed1111 --robustness
python -m mosei evaluate --data data/prepared --run runs/transformer-seed1111 --robustness
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

## Demonstrate a held-out clip

```bash
python -m mosei demo --data data/prepared \
  --run runs/transformer-seed1111 --baseline runs/text-seed1111 --index 0
```

Use `--clip-id 'VIDEO_ID$_$CLIP_ID'` to inspect an error-example ID. Output shows the transcript, annotation, sentiment probabilities and the text-only comparison. Probabilities are model outputs; the demo does not infer a person's true internal emotion. Cases of apparent sarcasm or disagreement remain qualitative observations because this dataset does not provide independent sentiment truth for each modality.

## Project files

```text
src/mosei/       data preparation, models, training, evaluation, reporting, CLI
configs/         three core configurations and three ablations
scripts/         dataset download and complete experiment runner
tests/           behavioral and end-to-end checks on labelled synthetic fixtures
docs/            implementation plan, progress/evidence and contribution tracker
data/            ignored raw archive and prepared arrays
runs/            ignored checkpoints, predictions and experiment manifests
results/         ignored report exports
```

The original proposal remains in the course Assignment 3 `Plan` folder. Group members can record owned tasks, commits and experiments in `docs/contributions.csv`. The assignment report and presentation should use actual saved results, describe negative findings, and distinguish exploratory validation runs from the final held-out comparison.

Method references: [CMU-MOSEI](https://aclanthology.org/P18-1208/), [MulT](https://aclanthology.org/P19-1656/), [modality robustness](https://aclanthology.org/2022.naacl-main.50/). This concatenated-sequence Transformer is an application implementation, not an exact reproduction of MulT.
