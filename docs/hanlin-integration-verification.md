# Integration verification, 21 October 2026

## Tested revision and environment

Integration based on Hanlin commit `2d379dc33f977312a9325469bbaf67d85b58f11a`,
which was 21 commits ahead of and zero behind master
`fae2ef9f910087bd16de90cb2b88412b3baea3e8`. Original source bytes were checked
against GitHub blob hashes before edits. The teammate branch was not rewritten.

Cloud CPU environment: Python 3.12, PyTorch 2.14.1+cpu, NumPy 2.3.5,
Transformers 4.57.6. No user dataset, CUDA device or authenticated Colab runtime
was accessed. Tests use explicitly synthetic fixtures.

## Results

- Full `python -m pytest -q`: **203 passed, 2 failed**. Both failures are
  pre-existing Docker-launcher environment tests,
  `test_launcher_uses_daemon_socket_and_correct_user_namespace[rootless]` and
  `[desktop]`. This sandbox denies creation of their temporary Unix sockets
  with `PermissionError: [Errno 1] Operation not permitted`. The same failures
  occurred before integration; they are not model failures. An approved-context
  retry could not launch due to the environment's read-only runtime mount.
- Excluding only those two environment-blocked tests: **203 passed,
  2 deselected**. No product test was disabled or modified to hide a failure.
- `ruff check src scripts tests`, `python -m compileall -q src scripts tests`,
  and `git diff --check`: passed.
- Both new architectures: forward/backward with partial/all missing streams;
  NaN/Inf/large padded-junk invariance; extra-padding invariance; two-logit
  checkpoint round trip; all eight component ablations and hooks proving each
  disabled component is actually skipped. Masked softmax is checked in fp16,
  bfloat16 and fp32.
- Synthetic multi-window training for both models: one prediction per clip,
  full-video 50% training selection, full validation data, no test split during
  training, preserved normalization/manifests, validation-selected checkpoints,
  completed-run resume and atomic best-checkpoint recovery after interruption.
  Both then passed clean plus all 12 missing/noisy-modality evaluations.
- Independent review found and fixed a mixed-generation reporting regression:
  resumed matrices combining completed legacy seed configs and newly saved seed
  configs now compare canonical effective settings. The original failing matrix
  was rerun unchanged and completed; meaningful model changes still fail.
- Old shared matrix configurations apply defaults on resume. Existing
  text/concat/Transformer model checkpoints load strictly and produce
  bitwise-identical outputs in direct old-versus-new probes.
- Direct source-equivalence probes: Graph-MFN full model and all four ablations
  match the submitted model after strict weight loading. ALMT full model and
  all four ablations match the submitted model after correcting only its
  all-keys-masked AHL bug for present streams. The uncorrected source had zero
  audio and visual attention-projection gradients in all three AHL layers;
  the new regression requires nonzero gradients.
- Colab: default matrix unchanged; new models explicitly opt in; each selected
  configuration is calibrated independently; validation-only export selection;
  source-bundle hashes and required files checked. A bundle containing synthetic
  tool manifests was extracted and pip-installed offline into a separate target.
  An isolated Python process imported and constructed both models without the
  checkout or extracted source directory on `sys.path`.
- All generated notebook code cells compile. Repeated generation is
  byte-identical; notebook 01 is unchanged.

## Remaining acceptance boundary

### Export completion follow-up

Post-integration review identified an inherited notebook edge case: exporting
after an interrupted matrix could select from only the completed seed folders.
Export now requires `trained`/`evaluated` manifest status, the exact frozen
configuration/seed/data identity, and matching completion metadata plus export
files for every frozen run. It ignores extra seed folders and ranks the chosen
experiment using validation metrics only. Legacy missing-default configs remain
compatible. Independent review found no blocker in the guard.

After this fix, the full suite reports **227 passed, 2 failed**, with only the
same two Unix-socket environment failures described above. All **32 Colab
tests** pass; the added export regression cases were first observed failing
against the previous implementation. Ruff, notebook compilation, deterministic
regeneration and diff checks pass, and notebook 01 remains unchanged.

### Data and runtime limits

This establishes code integration, not real-data performance or published-paper
reproduction. No new MOSEI accuracy, F1, training time, GPU memory requirement or
Colab cost is claimed. Official splits and full-training normalization are
reused unchanged through the existing cache loader; the user's actual cache
was not available here to inspect or train on. Run the real-cache smoke command
in the [integration guide](hanlin-integration.md) before committing to the full
training matrix. The two sandbox-blocked Docker tests should be rerun on a host
that permits local Unix sockets.
