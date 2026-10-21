# ALMT and Graph-MFN integration plan

## Scope

Integrate Hanlin's branch at `2d379dc33f977312a9325469bbaf67d85b58f11a`
with the existing shared MOSEI pipeline at master
`fae2ef9f910087bd16de90cb2b88412b3baea3e8`. Preserve the teammate branch
and its history. The user approved planning, integration, validation and merge.

1. Port the two existing model architectures into installable flat `mosei`
   modules. Adapt only the feature/mask interface and two-logit classification
   contract; preserve the custom MSALMT graph head and Graph-MFN components.
2. Test and fix masked-input handling, including ALMT's inverted all-padding
   attention mask. Test finite forward/backward behavior and checkpoint reload.
3. Route training, evaluation, component ablations and training-size experiments
   through the existing shared trainer, official prepared folds, full-training
   normalization, clip-level duration aggregation and whole-video sampling.
   Retire incompatible duplicate loaders/trainers rather than inventing
   `mosei_dataset.py` or silently changing data preparation.
4. Add explicit opt-in Colab model choices, architecture-specific timing and
   clean extracted source-bundle import/install validation.
5. Run existing regressions, synthetic multi-window train/reload/evaluation,
   configuration/resume tests, lint and independent whole-change review.
6. Publish an integration commit/PR based on Hanlin's branch and merge into
   master only after code-level acceptance; verify remote commit and CI state.

## Review focus and acceptance boundary

- Older Transformer/text/concat configurations and checkpoints remain readable.
- Padding, absent modalities and ablations do not produce non-finite gradients.
- Train fractions never subsample validation/test or split source videos.
- Model/config files work from an installed extracted Colab source bundle.
- No data, credentials, model weights or experiment results are committed.
- Synthetic fixtures establish code integration, not MOSEI accuracy, CUDA
  throughput, real-data memory requirements or a successful Colab training run.
  The real dataset is only on the user's currently offline computer. Provide
  reproducible smoke commands for that final environment-specific acceptance.
