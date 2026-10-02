# Video prototype verification

This record separates implemented and exercised behavior from research work that has not yet run. Code and data live in `/home/gys/Projects/nlp-assignment-3`.

## Verified locally

Final local suite after review fixes: **107 passed in 29.14 seconds**. Ruff
check and format check passed (57 files). The real checkpoint/tool readiness
command passed after regenerating the small real-data cache and retraining all
three integration models. All nine rendered proposal pages were visually inspected after
the final document edit.

- Real CSD reader validated 23,259 label rows and 1,084,005 timestamped word rows. Official SDK video-fold lists contain 2,249 training, 300 validation and 678 test video IDs; these list lengths are not final eligible clip counts.
- Full source download selects timestamped words, labels, COVAREP and OpenFace 2 from `samarwarsi/cmu-mosei`, version 1: 28,365,272,301 bytes. Large audio/vision transfers are still in progress at this checkpoint. Sizes come from the pinned catalog; saved SHA-256 values are calculated locally.
- A separate nine-video subset (three videos from each official fold) was downloaded and prepared for integration only. After exact-neutral exclusions, it contains 25/8/13 train/validation/test clips and 25/9/13 windows. Training class counts are 8 negative and 17 positive. All three model families trained for one epoch and restored successfully.
- Every clip contributes one label, irrespective of window count. Tests cover duration-weighted logits, gradients through all windows, stable corruption, training-only normalization, held-out source-video separation and old/new cache compatibility.
- A simulated disconnect during training followed by resume produced exactly the same CPU model weights and losses as uninterrupted training on the test fixture. Resume refuses changed data/configuration. A separate interrupted-report test confirms incomplete exports are preserved and rebuilt.
- The pinned SDK download was exercised against the real source. Network text is checksum-verified and parsed as literal assignments, never executed.

## Actual new-video check

The spoken example is a 15-second excerpt, starting 15 seconds into the public-domain [24 January 2009 US presidential weekly address](https://commons.wikimedia.org/wiki/File:20090124_WeeklyAddress.ogv), resized to 640 pixels wide. Original paired speech and face were retained.

| Check | Observed result |
|---|---|
| Audio extraction | 1,500 intervals × 73 named channels |
| Face extraction | 450 intervals × 40 named channels |
| Face quality | 100% of extracted frames passed the fixed threshold |
| Shared preparation | Finite normalized text/audio/vision tensors; one speech window |
| CLI prediction | Finite positive/negative probabilities summing to one; transcript/timestamps retained |
| First full analysis | 121.42 seconds for the 15-second video |
| Subsequent Gradio upload API | 98.1 seconds of analysis; about 98.3 seconds including the client call |
| Final API repeat after recovery/lifetime fixes | 88.2 seconds of analysis; finite probabilities sum to one |
| Silent upload | Explicit “No speech detected” error; probability, transcript and window outputs cleared |

Artifacts: `data/video-smoke/prediction.json`, `web-api-result.json`, `web-api-silent-result.json` and `runs/video-integration-v2/`. These are ignored local outputs. The integration classifier predicted positive sentiment for the economic-news excerpt; this is pipeline evidence, not an accuracy claim. It was trained on only 25 clips for one epoch, and the interface labels it accordingly.

FFmpeg-generated fixture tests cover media boundaries, malformed input and stream errors. Pipeline tests also cover insufficient face/voice coverage, cleanup after success/failure and incompatible manifests. Actual native COVAREP compatibility tests were executed in its Docker image.

## Feature contract and limits

- Frozen BERT revision: `86b5e0934494bd15c9632b12f734a8a67f723594`.
- Whisper small.en revision: `d1d751a5f8271d482d14ca55d9e2deeebbae577f`.
- COVAREP revision: `5a2be5d6b776f14a0b275c69fde90eb13849e60d`; MATLAB-only `creak` excluded consistently.
- OpenFace base image digest: `sha256:f43ad4e7fa4530143c7a9e0e8eca7e4f2b45599c1ef19680b68ad1eebba05197`; a local derived image makes the supplied model weights readable to the unprivileged runtime UID.
- Current verified feature fingerprint: `00a27ecbeb038294e895791f0ecca2be9302a1308bfd161eaf4b42571688a60a`. Check `data/tools/video-tools.json` for the current manifest after any intentional change.

CSD measurement intervals are bins. Live audio uses 10 ms intervals and canonical video uses 30 fps bins. Alignment weights finite overlapping measurements by interval overlap. OpenFace includes two gaze angles, three head rotations, 17 action-unit intensities and 18 presence indicators; tracking must succeed with confidence at least 0.8. Inputs are windowed without discarding later words, and normalized with saved training statistics.

The original CSD release does not identify exact historical extraction commits/settings. We established named features, units and a shared preprocessing contract, not identical numerical features. Human training transcripts and automatic upload transcripts differ too. A labelled evaluation of the complete video pipeline is still required to assess its accuracy and those distribution differences.

## Colab handoff and remaining checks

The two notebooks and small `dist/mosei-colab-source.zip` are prepared. All notebook code cells pass Python syntax checks. Source entries are hashed; the prepared cache is transferred with its hash. GPU allocation is printed, a separate one-epoch timing run estimates training time, and the current displayed compute-unit rate must be entered. Epoch checkpoints, prepared arrays and results persist in Drive. No Colab GPU session or cloud training was started from this environment.

Pending acceptance items:

- Finish the full selected CSD download and prepare the complete research folds.
- Execute and validate the notebooks in the user's authenticated Colab runtime.
- Train the frozen core models and ablations across three seeds, evaluate fixed checkpoints and generate the report. No final performance or full-training-time claim exists yet.
- Export the validation-selected compatible Transformer and test it with fresh labelled English clips.
- Inspect playback, progress and layout in a connected browser. The real Gradio API was tested, but the computer-use tool had no available browser surface.

The proposal is updated in its existing Course location with the upload pipeline, feature selection and Colab plan. Its prior version is backed up at `data/proposal-backup/before-video-prototype.docx`. Rendered proposal pages are internal layout checks, not new assignment deliverables.

## Independent code review and fixes

A separate reviewer inspected `48c3a97..e3f6bcd` and independently ran the original
99-test suite. No critical findings or minor items were raised. Three important
issues were reproduced and corrected with regression tests that failed first:

- Interrupted cache extraction/publication: restore into a temporary directory,
  verify complete content, then rename. Always check publication independently of
  local cache existence. Interrupted local directories are preserved for inspection.
- Prepared-data integrity: include all arrays and clip metadata in the fingerprint
  through per-file hashes, verify on load, and enforce labels against original
  non-neutral sentiment scores. Changed labels, features and transcripts are rejected.
- Checkpoint consistency: commit current and best model state in one `last.pt`,
  then export `best.pt`; resume repairs that export from the committed epoch.
  A test injects a disconnect and slight retry loss drift at the previous failure
  boundary and confirms checkpoint metrics match the reported selected epoch.

The real prepared-cache archive also completed a local write/restore round trip.
The generated notebook calls those tested recovery functions. Full Colab execution,
full-fold accuracy/cost and visual browser acceptance remain pending; historical
extractor numerical parity remains unknown. These limits are retained rather than
inferred from software tests.

A repeated real upload exposed preview-cache expiration during the slower CPU
extraction stage. The pipeline now copies the upload into its per-request
temporary directory before extraction, and the preview expires after about an
hour. A regression test deletes the original preview during analysis, confirms
the face stage still reads intact input, and verifies request cleanup. The test
failed before the fix and passed afterward; the full suite passed 107 tests.

Final real API repeat passed with the refreshed checkpoint: 88.2 seconds of
analysis for the same 15-second clip (88.27 seconds including the client call).
The following silent upload again cleared every result field. Final response
records are `data/video-smoke/final-web-api-result.json` and
`final-web-api-silent-result.json`. The source ZIP contains 42 entries with
verified hashes and a clean ZIP integrity check.
