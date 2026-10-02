# Video prototype implementation progress

Plan: `docs/superpowers/plans/2026-10-03-video-upload-prototype.md`.
Spec approved by the user on 2026-10-03.

- Task 1: pending — CSD source, metadata and real extractor compatibility.
- Task 2: pending — shared alignment, windowing and BERT.
- Task 3: pending — preparation and clip-level training.
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
