# Video prototype verification

This record separates implemented and exercised behavior from research work that has not yet run. Code and data live in `/home/gys/Projects/nlp-assignment-3`.

## Verified locally

Final local suite before independent review: **99 passed in 31.07 seconds**. Ruff
check and format check passed (56 files). The real checkpoint/tool readiness
command passed. All nine rendered proposal pages were visually inspected after
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
| Silent upload | Explicit “No speech detected” error; probability, transcript and window outputs cleared |

Artifacts: `data/video-smoke/prediction.json`, `web-api-result.json`, `web-api-silent-result.json` and `runs/video-integration/`. These are ignored local outputs. The integration classifier predicted positive sentiment for the economic-news excerpt; this is pipeline evidence, not an accuracy claim. It was trained on only 25 clips for one epoch, and the interface labels it accordingly.

FFmpeg-generated fixture tests cover media boundaries, malformed input and stream errors. Pipeline tests also cover insufficient face/voice coverage, cleanup after success/failure and incompatible manifests. Actual native COVAREP compatibility tests were executed in its Docker image.

## Feature contract and limits

- Frozen BERT revision: `86b5e0934494bd15c9632b12f734a8a67f723594`.
- Whisper small.en revision: `d1d751a5f8271d482d14ca55d9e2deeebbae577f`.
- COVAREP revision: `5a2be5d6b776f14a0b275c69fde90eb13849e60d`; MATLAB-only `creak` excluded consistently.
- OpenFace base image digest: `sha256:f43ad4e7fa4530143c7a9e0e8eca7e4f2b45599c1ef19680b68ad1eebba05197`; a local derived image makes the supplied model weights readable to the unprivileged runtime UID.
- Initial verified feature fingerprint: `639612bceda8b4e1da07fe77aa17f598a776eabe5e1bae3f89b5fb5aa589f110`. Check `data/tools/video-tools.json` for the current manifest after any intentional change.

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
