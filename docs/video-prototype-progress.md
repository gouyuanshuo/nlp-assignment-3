# Video prototype implementation progress

Plan: `docs/superpowers/plans/2026-10-03-video-upload-prototype.md`.
Spec approved by the user on 2026-10-03.

- Task 1: in progress — real CSD reader and resumable downloader pass tests;
  native extractor compatibility is being checked while large files download.
- Task 2: complete — shared alignment, windows and frozen BERT wordpiece pooling;
  11 focused tests passed after observed missing-module failures.
- Task 3: in progress — preparation and clip-level training.
- Task 4: pending — uploaded-video extraction.
- Task 5: pending — browser interface and CLI.
- Task 6: pending — revised experiments, evidence and proposal.

Ruling: continue native implementation in the existing dedicated Projects repo
after approval of the written design — the user already requested implementation
and the governing collaboration instructions prohibit redundant authorization
requests. No remote publication or shared-branch merge is included.

Pre-flight: Tasks 1/2 establish named timestamped streams and a fingerprinted
manifest; Tasks 3/4 consume exactly those shared preprocessing interfaces. Task 3
retains one label per clip despite multiple windows; Tasks 4/5 use the same
duration-weighted logit aggregation. Existing caches remain on their original
loader path and cannot be used by the video-upload command.

Verification so far: 67 tests passed; Ruff check and format check passed.
Real labels/words load; official SDK folds have 2249/300/678 source videos.
CSD metadata confirms 74 named COVAREP features and the OpenFace 2 columns.
BERT and Whisper model snapshots are downloaded at pinned revisions.

Task 1: Ruling: exclude COVAREP `creak` from both training and upload inference —
upstream documents a MATLAB Neural Networks Toolbox dependency unavailable in
Octave — costs one acoustic feature; never replace it with fabricated zeros.
Native smoke testing also exposed obsolete `wavread`, missing Octave `lpc`, and
different `findpeaks` semantics; these remain compatibility work, not a passed
extraction check.
