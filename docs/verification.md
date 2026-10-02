# Verification evidence

Project: `/home/gys/Projects/nlp-assignment-3`.

## Software checks completed

- `uv sync --frozen --extra dev`: isolated environment installed from `uv.lock`.
- Full pytest suite: 46 passed after the independent review fix.
- Ruff: checks and formatting passed.
- End-to-end CLI test: prepare → train → restore → 13 evaluation conditions → report → demo → matrix train-only.
- Synthetic fixtures are explicitly labelled in tests; their scores are never research evidence.
- Independent code review: no critical issues, one important mask/length ambiguity fixed with failing-then-passing regression tests.
- Default Transformer CUDA execution: 10 optimization steps, batch 32, 50 positions per stream, 1,347,330 trainable parameters, approximately 253 MB peak allocated GPU memory.
- Exported confusion-matrix figures inspected; readable cell contrast and wrapped titles verified.

## Real-release verification completed

Downloaded `aligned_50.pkl` from Kaggle mirror `neffow/cmu-mosei`, version 1. The 4,683,341,880-byte file matches the authors' published SHA-256:

`45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791`

The full loader audit passed: expected feature shapes, valid BERT-derived aligned masks, no duplicate clip IDs or cross-split source-video overlap, and no non-finite values among eligible valid feature elements. Normalization uses training tokens only.

| Split | Released clips | Neutral excluded | Eligible | Negative | Positive |
|---|---:|---:|---:|---:|---:|
| train | 16,326 | 3,540 | 12,786 | 4,738 | 8,048 |
| valid | 1,871 | 433 | 1,438 | 506 | 932 |
| test | 4,659 | 1,025 | 3,634 | 1,350 | 2,284 |

## GPU execution check on real data

Each core model trained for **one epoch**, seed **1111**, on all 12,786 eligible training clips and evaluated all 13 conditions on the 1,438 eligible **validation** clips. These are software checks, not the final research comparison. No test-set metrics were computed.

| Model | Validation macro F1 | Validation accuracy | Trainable parameters | Training seconds |
|---|---:|---:|---:|---:|
| text | 0.8222 | 0.8387 | 197,378 | 1.98 |
| concat | 0.8213 | 0.8387 | 422,658 | 1.91 |
| transformer | 0.8357 | 0.8547 | 1,347,330 | 7.03 |

Saved artifacts: `runs/real-data-smoke-20261002/`. This includes the three trained checkpoints, configurations, logs, 39 condition prediction CSVs, comparison report, confusion figures, error-example IDs, and `demo.json` from validation index 0. The demo successfully reloads the Transformer and text-only checkpoints with the same preprocessing.

Prepared data: `data/prepared/`. Audit fingerprint: `d48cc4525ded122b8fa05840ba0d4a3ce40ca0ca19aa207953a15a6d51862782`.

The final three-seed test comparison and ablation study have **not** been run. Their executable runner and configurations are ready in `scripts/run_experiments.py` and `configs/`. Freeze any validation-based tuning decisions before running that final matrix; report these one-epoch checks separately.
