# Video prototype implementation progress

Plan: `docs/superpowers/plans/2026-10-03-video-upload-prototype.md`.
Approved design: `docs/superpowers/specs/2026-10-03-video-upload-prototype-design.md`.

- Task 1: complete — all four selected CSD files downloaded (28,365,272,301 bytes), with recorded local checksums and readable schemas. Reader/schema tests and all real label/word rows pass; native tools processed a spoken clip.
- Task 2: complete — shared alignment, masks, windows, BERT and normalization. All windows are retained.
- Task 3: complete — full CSD cache prepared from official folds; all six configurations trained across three seeds on an A100. Eligible train/validation/test counts are 12,789/1,438/3,637 clips.
- Task 4: complete implementation and real extraction/inference check — English speech, voice and face features reach the validation-selected full-data Transformer in `runs/colab/selected-run`.
- Task 5: trained-model video upload and silent-error API checks passed. CLI and localhost interface work. Initial desktop layout and missing-video handling were inspected in a browser; upload/playback and progress still need manual visual review because the browser tool could not access the Projects fixture path.
- Task 6: complete training/evaluation/export — 18 runs, 234 held-out evaluations and paired source-video bootstrap comparisons saved. Full models/cache and exports persist in Drive; the selected demo checkpoint, research report and training records are also local. Assignment report writing, presentation and evaluation on fresh labelled uploads remain follow-up work.

## Decisions and costs

1. Continue in the existing dedicated Projects implementation branch. The user already authorized implementation; code and data stay out of Course. The existing Course proposal is edited in place with a preserved backup. No remote publish or shared-branch merge is included.
2. Exclude COVAREP `creak` from both training and uploads because it requires a MATLAB toolbox unavailable in Octave. Cost: one acoustic channel. Preserve actual upstream feature families; never fabricate substitutes.
3. Preserve label segments in source storage order, because real CSD annotations are not chronologically sorted. Cost: overlapping source annotations may share speech context; they retain separate original IDs within the same video fold. Feature streams still require ordered timestamps.
4. Use Colab for the full matrix following the user's confirmation of 200 compute units and GPU access. On 11 October 2026 the A100 session completed at a displayed rate of 6.77 units/hour; the balance changed from 197.24 to 189.00 units. Recorded model training totalled 21.4 minutes; preparation, evaluation, transfers and idle time also consumed runtime. The runtime was stopped and the final usage rate was 0.00/hour.
5. Use real Gradio API uploads for automated end-to-end checks. Browser inspection covered the initial layout and empty-input error, while file access restrictions prevented a browser-driven upload. Cost: playback and progress appearance still require manual review.

Historical extractor settings are not fully documented by the CSD release. Schema, feature families, units, masks and alignment are verified; numerical identity to historical features is not claimed. The report must distinguish released-transcript benchmark performance from automatic-transcript uploads.

See [execution evidence](video-prototype-verification.md) and [Colab instructions](../README.md).

Independent review completed with three important findings fixed in one test-led
pass: cache transfer recovery, prepared-file integrity and epoch/best-checkpoint
consistency. That suite passed 107 tests with Ruff passing. The Colab execution
subsequently exposed editable-source discovery and stale loaded package issues;
the notebook setup now detects them, with 110 tests and Ruff passing.

The final real-upload repeat exposed a preview-cache expiry race. Each request
now owns its input copy until completion; regression and full-suite checks pass.

The current demo uses `transformer-seed1113`, best epoch 3, selected on validation
before held-out evaluation. Its spoken-video API check took 82.7 seconds for a
15-second clip; silent input cleared all prediction fields. The local app is at
http://127.0.0.1:7860. Clean test macro F1 means are 82.49% (text), 82.42% (concat)
and 82.66% (Transformer). Both baseline comparison intervals include zero, so
the small clean-data improvement is inconclusive. All original hypotheses and
the frozen experiment matrix are retained.
