# Video prototype verification

This record preserves the earlier integration checks and the completed full-data Colab experiment on 11 October 2026. Code and local data live in `/home/gys/Projects/nlp-assignment-3`.

## Verified locally

Earlier local suite after review fixes: **107 passed in 29.14 seconds**. Ruff
check and format check passed (57 files). The real checkpoint/tool readiness
command passed after regenerating the small real-data cache and retraining all
three integration models. All nine rendered proposal pages were visually inspected after
the final document edit.

- Real CSD reader validated 23,259 label rows and 1,084,005 timestamped word rows. Official SDK video-fold lists contain 2,249 training, 300 validation and 678 test video IDs; these list lengths are not final eligible clip counts.
- Full source download selects timestamped words, labels, COVAREP and OpenFace 2 from `samarwarsi/cmu-mosei`, version 1: 28,365,272,301 bytes. All four transfers completed; each assembled file has its size/hash record and opens as the expected CSD stream. Acoustic/visual source column hashes match the feature manifest. Sizes come from the pinned catalog; saved SHA-256 values are calculated locally.
- A separate nine-video subset (three videos from each official fold) was downloaded and prepared for integration only. After exact-neutral exclusions, it contains 25/8/13 train/validation/test clips and 25/9/13 windows. Training class counts are 8 negative and 17 positive. All three model families trained for one epoch and restored successfully.
- Every clip contributes one label, irrespective of window count. Tests cover duration-weighted logits, gradients through all windows, stable corruption, training-only normalization, held-out source-video separation and old/new cache compatibility.
- A simulated disconnect during training followed by resume produced exactly the same CPU model weights and losses as uninterrupted training on the test fixture. Resume refuses changed data/configuration. A separate interrupted-report test confirms incomplete exports are preserved and rebuilt.
- The pinned SDK download was exercised against the real source. Network text is checksum-verified and parsed as literal assignments, never executed.

## Earlier integration-checkpoint video check

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

## Full Colab execution — 11 October 2026

The installed authenticated Colab CLI 0.7.4 ran the preparation, timing, training
and evaluation/export workflow on an NVIDIA A100-SXM4-80GB. The user completed
the interactive Drive authorization. CLI-adapted notebooks replace only the
browser upload/mount interactions and preserve the original pipeline and model
configurations. Each phase uses one code cell: this CLI version otherwise
continues after cell errors and can exit zero. All four completed output
notebooks were inspected and contain no error outputs; completion artifacts
were verified independently.

Full source CSD checksums match the local originals. The prepared cache retained
all windows, used source-video folds and fitted normalization on training data
only. Exactly neutral clips were excluded under the frozen binary protocol.

| Split | Eligible clips | Windows | Negative | Positive | Neutral exclusions |
|---|---:|---:|---:|---:|---:|
| Train | 12,789 | 13,242 | 4,739 | 8,050 | 3,538 |
| Validation | 1,438 | 1,484 | 506 | 932 | 433 |
| Test | 3,637 | 3,763 | 1,350 | 2,287 | 1,025 |

No clips were excluded for missing words. Two test-fold videos,
`7l3BNtSE0xc` and `dZFV0lyedX4`, lacked required streams. Clips with no usable
vision were retained with masks: 211 training, 14 validation and 58 test clips.
The prepared-data fingerprint is
`50eba4dec0f14e7c2afaa9c60f5519366165b013ac08788e05353d6d71519682`.

All six configurations completed seeds 1111, 1112 and 1113, using the frozen
30-epoch maximum and six-epoch patience. The timing run measured 11.69 seconds
for one full-data Transformer epoch. The sum of recorded training time across
the 18 research runs was **1,284.27 seconds (21.4 minutes)**. This excludes
preparation, evaluation and transfers. BERT representations are cached and
frozen; the compact classifiers are trained on those prepared features, so
source-file size alone does not determine epoch duration.

The displayed runtime rate was 6.77 compute units/hour. Balance moved from
197.24 to 189.00 units, approximately **8.24 units for the entire session**.
Drive was flushed/unmounted after export, then the runtime was stopped.
The final CLI check reported no active sessions, zero active assignments and
0.00 units/hour.

Recorded environment: Python 3.13.15, PyTorch 2.11.0+cu130, CUDA 13.0,
Transformers 4.57.6, Tokenizers 0.22.2, faster-whisper 1.2.1,
CTranslate2 4.8.2, NumPy 2.5.3 and SciPy 1.18.1. See the saved
`environment.json` for the machine-readable record.

### Held-out results and interpretation

There are **234 completed evaluations**: 18 runs × 13 clean/missing/noise
conditions. The matrix manifest was frozen before evaluation and checked
against the final manifest. The table shows means and sample SD across three
seeds; SD is expressed in percentage points.

| Configuration | Macro F1 (%) | F1 SD (pp) | Accuracy (%) |
|---|---:|---:|---:|
| Text-only | 82.49 | 0.05 | 83.84 |
| Concatenation | 82.42 | 0.38 | 83.76 |
| Transformer | 82.66 | 0.69 | 83.61 |
| Without modality embeddings | 82.63 | 0.19 | 83.55 |
| Without positions | 82.61 | 0.39 | 83.81 |
| Mean pooling | 82.06 | 0.13 | 83.16 |

The Transformer's mean macro F1 difference from text-only is **+0.174
percentage points**, with a 95% paired source-video bootstrap interval of
**[-0.617, +0.973]**. Against concatenation it is **+0.245 points**, with
interval **[-0.584, +1.114]**. Both intervals include zero; these runs do not
establish a clear clean-data improvement. Accuracy is slightly higher for the
text-only baseline. The intervals use 2,000 resamples of 639 source-video
clusters, bootstrap seed 2026, holding the fitted training seeds fixed.

When text features are masked, mean macro F1 is 58.83% for the Transformer,
46.27% for concatenation and 38.61% for text-only. This supports stronger
performance under this specific missing-text intervention. Feature corruption
experiments do not establish robustness to all real recording conditions.

The demo checkpoint was selected **before test evaluation** from Transformer
runs using validation macro F1, with validation loss as the tie-breaker:
`transformer-seed1113`, best epoch 3, validation macro F1 0.8297993. It is not
the seed with the highest observed test score. Its local run records explicitly
identify it as a full-data checkpoint, and its feature fingerprint matches the
installed upload extractors.

### Exports and full-data demo acceptance

Drive location: `MyDrive/NLP_Assignment_3`. The full prepared cache and all
research runs remain there, with run directory
`runs-csd-50eba4dec0f14e7c`. Local exports are:

- [Research report](../results/colab-20261011/results.md), including CSV/JSON
  metrics, paired differences, error examples and six confusion plots.
- Selected model and normalization in `runs/colab/selected-run/`.
- `runs/colab-20261011/`: input/output notebooks, environment, data audit,
  frozen manifest, timing, training records, logs and acceptance records.
- Three final ZIP archives, verified against the remote SHA-256 manifest,
  plus `execution-evidence.zip` with the operational notebooks and executed
  source bundle. Copies of the execution evidence and final exports were
  saved in Drive before shutdown.

The selected checkpoint ZIP SHA-256 is
`b18c355d04ed4e88b5e6eb3f2b311a481e1a4889da4da9f135c50d6d1af00d50`.
The executed source bundle is preserved separately from subsequent notebook
setup fixes, so a rebuilt handoff bundle is not confused with the code used
for the experiment.

The full-data model passed readiness checks and real Gradio API uploads at
http://127.0.0.1:7860:

| Check | Observed result |
|---|---|
| Spoken 15-second example | Negative sentiment; negative probability 0.9085; transcript and one speech window populated |
| Processing time | 82.7 seconds; 82.92 seconds including the client request |
| Silent example | Explicit no-speech error; probabilities, transcript and window rows cleared |
| Checkpoint status | Full-data seed 1113, epoch 3; no integration/pending warning |

These clips establish pipeline operation, not accuracy on newly recorded
videos. Gradio serializes an empty probability Label as `{}`; the acceptance
assertion was corrected to accept that representation and the silent check
was repeated successfully. The authoritative results are
`runs/colab-20261011/demo-acceptance.json` and the two saved response JSON files.

Initial desktop layout and missing-video handling were inspected in the
connected browser. Browser upload was prevented by the tool's allowed file
roots, which did not include Projects; the upload checks above used the real
local Gradio API. Playback and progress appearance still need manual review.

Colab setup exposed two problems: the running notebook kernel did not discover
the newly installed editable package, and it retained an older NumPy in memory
after installation. A kernel restart resolved the actual run. The generated
notebooks now expose the source path immediately and stop with a restart
instruction when loaded package versions disagree with the pinned manifest.
Regression tests failed before this fix, then **110 tests passed in 25.26
seconds**; Ruff and notebook syntax checks passed.

Remaining project work is to test fresh labelled English uploads, visually
review upload/playback and progress, and use the frozen evidence to write the
assignment report and presentation. The historical extractor and automatic
transcript distribution differences described above remain limitations.

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
The generated notebook calls those tested recovery functions. At that review,
full Colab execution, full-fold accuracy/cost and visual browser acceptance were
pending. The completed cloud run is documented above; historical extractor
numerical parity remains unknown.

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

Completed source inventory: 3,837 word-stream videos, 3,293 label-stream videos,
3,836 acoustic-stream videos and 3,837 visual-stream videos. These are source
container inventories; official-fold membership, missing streams and neutral
exclusions determine the final prepared research counts. The downloader exited
successfully after assembling the final visual file.
