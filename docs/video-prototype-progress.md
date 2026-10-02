# Video prototype implementation progress

Plan: `docs/superpowers/plans/2026-10-03-video-upload-prototype.md`.
Approved design: `docs/superpowers/specs/2026-10-03-video-upload-prototype-design.md`.

- Task 1: complete implementation and extractor checks; full selected CSD download is still running. Reader/schema tests and all real label/word rows pass; native tools processed a spoken clip.
- Task 2: complete — shared alignment, masks, windows, BERT and normalization. All windows are retained.
- Task 3: complete implementation — CSD cache and clip-level training tested; all three models trained and restored on a small real integration subset. Full-fold research runs remain pending.
- Task 4: complete implementation and real extraction/inference check — English speech, voice and face features reach the trained integration checkpoint.
- Task 5: API upload and silent-error checks passed. CLI and localhost interface implemented. Visual browser inspection remains pending: no browser was available through the session's computer-use tool.
- Task 6: Colab notebooks, durable epoch/matrix resume, source bundle and updated proposal prepared. Full Colab training, frozen held-out evaluation and final research conclusions remain pending.

## Decisions and costs

1. Continue in the existing dedicated Projects implementation branch. The user already authorized implementation; code and data stay out of Course. The existing Course proposal is edited in place with a preserved backup. No remote publish or shared-branch merge is included.
2. Exclude COVAREP `creak` from both training and uploads because it requires a MATLAB toolbox unavailable in Octave. Cost: one acoustic channel. Preserve actual upstream feature families; never fabricate substitutes.
3. Preserve label segments in source storage order, because real CSD annotations are not chronologically sorted. Cost: overlapping source annotations may share speech context; they retain separate original IDs within the same video fold. Feature streams still require ordered timestamps.
4. Use Colab for the full matrix following the user's confirmation of 200 compute units and GPU access. Local tiny runs establish integration only. Cost: final performance remains unknown until the authenticated cloud run and evaluation finish.
5. Use real Gradio API uploads for the available automated check; retain visual playback/layout review as an explicit acceptance item. Cost: UI appearance has not been verified in a connected browser.

Historical extractor settings are not fully documented by the CSD release. Schema, feature families, units, masks and alignment are verified; numerical identity to historical features is not claimed. The report must distinguish released-transcript benchmark performance from automatic-transcript uploads.

See [execution evidence](video-prototype-verification.md) and [Colab instructions](../README.md).
