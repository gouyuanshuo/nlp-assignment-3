# ALMT and Graph-MFN: shared-pipeline integration

Hanlin's original work is preserved in the history of
[`implement_Hanlin/ALMT-and-Graph_MFN`](https://github.com/gouyuanshuo/nlp-assignment-3/tree/implement_Hanlin/ALMT-and-Graph_MFN)
at `2d379dc33f977312a9325469bbaf67d85b58f11a`. The runnable implementations now
live in `src/mosei/almt.py` and `src/mosei/graph_mfn.py`. The incompatible root
loaders/trainers and shared-protocol copy were retired in favor of the existing
installed `mosei` package. No missing `mosei_dataset.py` was invented.

## What is preserved and what changes

- ALMT keeps the submitted latent modality tokens, three language/AHL scales,
  cross-modal fusion and five-node graph classifier. This is Hanlin's custom
  MSALMT variant, not a claim of an exact reproduction of the ALMT paper.
- Graph-MFN keeps the submitted text/audio/vision order, unimodal LSTM cells,
  dynamic multihead graph, view/fusion memories and temporal attention.
- Both return two logits through `model(features, masks)`, using feature widths
  from the prepared cache. ALMT's old `image` key is mapped to shared `vision`.
- ALMT previously inverted an all-false latent padding mask, masking every
  audio/visual attention key and producing zero AHL attention gradients. The
  integration fixes this and tests nonzero audio and visual projection gradients.
- Masked values are cleared before projection. Entirely absent streams and
  fully padded examples have finite outputs/gradients; missing stream contexts
  are not allowed to leak projection biases into ALMT's audio/visual attention.
- All training/evaluation uses existing official cached splits, non-neutral
  binary labels, full-training normalization, duration-weighted clip logits,
  validation macro-F1/loss checkpoint selection, resumable optimizer/RNG state,
  matched corruption conditions and source-video bootstrap reporting.
- Sample-size runs use the existing whole-source-video sampler, not the old
  clip-wise stratified sampler. Validation/test stay complete. Fractions target
  a clip count and can overshoot because complete videos must stay together.
  Sample membership is saved with each run; fractions are not guaranteed nested.
- Existing shared-pipeline checkpoints remain loadable. The old standalone
  teammate checkpoint formats are not automatically converted: retrain with
  the shared runner rather than renaming checkpoint keys by hand.

## Run locally or in Colab

Install the project using its existing setup instructions. With your verified
prepared CSD cache available, first perform a real-data smoke run:

```bash
python scripts/run_experiments.py --data data/prepared-csd \
  --configs configs/almt.toml configs/graph_mfn.toml --seeds 1111 \
  --epochs 1 --train-only --device auto --output runs/hanlin-real-smoke
```

This checks one epoch on real data but is not a research result. Use a new
directory for the full frozen matrix; never reuse the smoke output as a final
run. Training requires explicit execution and may consume substantial compute.

```bash
python scripts/run_experiments.py --data data/prepared-csd \
  --configs configs/almt.toml configs/graph_mfn.toml \
  --seeds 1111 1112 1113 --train-only --device auto --output runs/hanlin-full
```

Only after freezing model decisions, repeat the same command without
`--train-only` and with `--resume` to evaluate fixed checkpoints and report.
`--resume` also recovers interrupted training; the frozen configuration and data
fingerprint must match. Older shared-run manifests receive current defaults for
new model fields, without changing their effective architectures.

Optional experiment expansions (each increases training cost):

- `--include-hanlin-ablations`: full model plus four independent component
  removals for each selected ALMT/Graph-MFN model. Use full-model configs as input.
- `--train-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1`:
  ten whole-video training-size conditions, named with `-frac10` through
  `-frac100`. Pair both models with identical seeds/fractions.
- Combining the two flags crosses all ablations with all fractions. Do not do
  this unintentionally: two models × five variants × ten fractions × three
  seeds means 300 runs.

The Colab notebook keeps the existing model matrix by default. Set
`INCLUDE_HANLIN_MODELS = True` in the timing/configuration cell to opt into the
two new full models. Each selected configuration is timed separately before
the full matrix; the estimates exclude preparation, test evaluation and idle
time and are not spending limits. The source bundle includes the installed
modules and both TOML configurations automatically.

## Configuration mapping

ALMT uses `d_model` for its hidden width, `n_heads`, `almt_token_count`,
`almt_max_input_length` and `almt_fusion_depth`. The three AHL scales remain
fixed. Its switches are `modality_embeddings`, `almt_use_ahl`,
`almt_use_cross_modal_fusion` and `almt_use_graph_fusion`.

Graph-MFN uses `hidden_dim` and `graph_mfn_heads`. Its switches are
`graph_use_dynamic_graph`, `graph_use_view_memory`, `graph_use_fusion_memory`
and `graph_use_temporal_attention`. All switches default to enabled.

See [integration verification](hanlin-integration-verification.md) for the
checks actually run and the remaining real-data/CUDA acceptance boundary.
