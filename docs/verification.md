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

## Real-release verification

The pinned Kaggle aligned archive is downloading. A queued verification job will check its published SHA-256, prepare all released splits, and run all three models for one epoch on the full eligible training set. It will evaluate the 13 conditions on validation, without using test metrics for development.

The final three-seed test comparison and ablation study require their own frozen experiment runs. The executable runner and configurations are provided in `scripts/run_experiments.py` and `configs/`.
