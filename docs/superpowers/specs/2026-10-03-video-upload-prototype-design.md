# CMU-MOSEI video upload prototype

Status: approved by the user's "good" on 2026-10-03. This document defines the
next project stage; the video upload implementation and larger dataset download
are not yet complete. The existing code implements the earlier cached-feature
experiment.

## Intended outcome

The user wants an Assignment 3 prototype in which a person uploads a previously
unseen English video and receives a sentiment classification. The preferred
training source is `samarwarsi/cmu-mosei` on Kaggle, rather than the MMSA
`aligned_50.pkl` mirror used for the initial implementation.

Success requires a working path from video bytes through text, audio and visual
feature extraction to a trained Transformer classifier. Selecting a cached
dataset example does not satisfy this requirement. All code, environments,
downloads and outputs belong under `/home/gys/Projects/nlp-assignment-3`.

Proposed first-demo scope: a local browser application accepting MP4, MOV or
WebM, at most 60 seconds and 200 MB, containing English speech and one visible
speaker. These input limits are proposed assumptions, not previously confirmed
requirements. The task remains positive versus negative sentiment; a neutral
class would require a separate label-protocol change and retraining.

## Evidence and approach selection

The preferred Kaggle package contains seven CSD feature/annotation files,
totalling 31,632,114,007 bytes. It provides timestamped words, word vectors and
phones, COVAREP acoustics, FACET and OpenFace visual features, and labels. It
does not supply the original video/audio recordings.

Recommended approach: use its timestamped words and labels, COVAREP acoustics,
and an explicitly documented subset of its OpenFace 2 features. Generate BERT
word representations from the supplied words during preparation. At inference,
transcribe the uploaded video and run the corresponding COVAREP, OpenFace and
BERT extraction. Verify the feature contract before training final models.

Alternative: acquire original CMU-MOSEI media and extract a new feature set with
modern speech and vision encoders for both training and uploads. This would
provide greater control over feature consistency, but requires additional media
acquisition and feature extraction beyond the selected 31.63 GB package. It is
not the default for this proposal.

The existing MMSA checkpoint is tied to its original feature representation.
Matching array widths is not sufficient to reuse it with a different extractor.
Retrain the three model families on the new, shared prepared cache.

## Compatibility check before full training

Implement this check first, before committing to expensive training:

1. Inspect the selected CSD files' metadata, feature names, dimensions,
   timestamps and available extractor/version information. Save the evidence.
2. Verify an OpenFace 2 executable can process a real spoken video and produce
   the chosen named features in their recorded order. Exclude identity-dependent
   pixel coordinates and administrative fields from model inputs; select the
   documented action-unit, gaze and pose features supported by both sources.
3. Verify COVAREP extraction with its required runtime and dependencies on the
   same video's audio. Establish the dataset-compatible feature ordering,
   sample rate, temporal sampling and treatment of non-finite values.
4. Pin tool/model versions, selected columns, units and all transformation rules
   in a feature manifest. Include the manifest fingerprint in every checkpoint.
5. Where a corresponding source recording can be obtained, compare extracted
   features with its released features to quantify compatibility. Schema checks
   alone do not establish numerical or distributional equivalence. Document any
   remaining provenance uncertainty and validate the upload pipeline separately.

An incompatible extractor must fail clearly; do not substitute MediaPipe for
OpenFace, generic audio embeddings for COVAREP, or pad unrelated features into
the expected width. If provenance or extraction cannot be made sufficiently
reliable, report the concrete issue and revisit the raw-media alternative.

The host has FFmpeg, Docker and an RTX 4060 Laptop GPU with 8 GB VRAM. OpenFace,
COVAREP/Octave, the transcription model, BERT and the web UI are not yet installed
in this project. COVAREP's documentation states that most functions support
Octave; the complete required extraction path still needs execution testing.

## Training pipeline

- Download the selected Kaggle release with resumable transfers. Record the
  uploader, version, file sizes and per-file hashes. A locally calculated hash
  supports reproducibility; it is not an independently published checksum.
- Read CSD files as HDF5. Audit video IDs, annotations, timestamp ordering,
  missing streams and feature schemas. Use the official CMU-MOSEI video splits
  from a pinned SDK revision and reject cross-split video overlap. Do not assign
  random clip splits or assume the old pickle's row counts apply.
- Form clips using annotation time intervals. Preserve original sentiment
  scores. Apply the existing binary rule (`y < 0` negative, `y > 0` positive,
  exclude exact zero) consistently to every split and record resulting counts.
- Extract frozen BERT representations from the timestamped transcript words
  using a pinned English model/tokenizer. Associate wordpieces back to words
  explicitly and average their representations per word.
- Aggregate valid COVAREP and OpenFace measurements into the corresponding word
  intervals using the same defined overlap/aggregation rule in both training
  and inference. Preserve masks when an interval lacks valid measurements.
- Use windows of at most 50 words. Clips that need multiple windows keep one
  source clip ID and one label. Take the duration-weighted mean of window logits
  per clip before loss and evaluation, weighting each window by its nonzero
  first-word-start to last-word-end duration. Thus long clips do not gain
  multiple independent labels or appear in multiple splits. The text encoder's
  token limit is enforced independently of
  the word count; any additional window boundary is recorded.
- Fit normalization using eligible training data only. Store normalization,
  feature manifests, masks, clip IDs and preparation counts with the cache.
- Reuse text-only, concatenation and Transformer model families. Retain the
  validation-selected checkpoints, fixed seeds, robustness comparisons and
  reporting approach. All comparisons must use the same new feature cache.

## Uploaded-video pipeline

1. Validate file size, media decodability, duration and audio/video streams with
   FFprobe. Reject unsupported or over-limit input with an actionable message.
2. Decode audio through FFmpeg. Run a pinned local speech-recognition model
   (proposed: faster-whisper `small.en`) to obtain the transcript and word times.
   Empty speech must not produce a confident sentiment prediction.
3. Extract BERT, COVAREP and OpenFace features using the checkpoint's manifest.
   Share word aggregation, windowing, normalization and masking code with
   training. Process heavy components sequentially where needed to fit VRAM.
4. Apply the trained model to the windows and aggregate logits using the same
   clip rule as evaluation. Display the corresponding positive/negative model
   probabilities and the transcript. Probabilities are not claimed to be
   calibrated certainty or measurements of a person's internal emotional state.
5. Show a clear failure when audio/vision processing is unavailable or no valid
   face is found. Do not silently turn the requested multimodal demo into a
   text-only system.

Human transcripts in the training package and automatic transcripts at upload
time introduce an additional source of error. Dataset test performance must be
reported separately from a small, labelled evaluation of the complete video
pipeline. Recognition quality and extraction latency belong in that evaluation.

## Local application

Use a small Gradio interface bound to localhost. The user uploads a video,
previews it and selects Analyse. Display processing stages, the generated
transcript, the predicted sentiment, and positive/negative probability bars.
For multi-window clips, show window predictions with timestamps so a mixed
opinion can be inspected alongside the aggregate result.

Load only a checkpoint whose extraction dependencies and feature manifest pass
startup checks. No fabricated predictions, untrained placeholder classifier or
automatic substitution of the old MMSA checkpoint is permitted. Process media
in per-request temporary directories and clean up after processing. Public
hosting, accounts, live webcam capture and multiple-speaker analysis are outside
the proposed first version.

## Validation and acceptance

- Existing experiment tests continue to pass.
- CSD fixture tests cover label/time association, split leakage, missing frames,
  feature-column mismatches and normalization fitted only on training data.
- Shared processing tests demonstrate that equivalent timestamped features yield
  identical tensors through training and inference paths, including windowing.
- Run the actual transcription/audio/visual toolchain on a real English spoken
  video; mocks and synthetic arrays alone do not establish demo readiness.
- Train and restore all three model families on the selected real-data cache;
  run the final experiment protocol after validation choices are frozen.
- In the browser, upload a new video, inspect its playback/transcript, obtain
  finite probabilities from the saved model, and verify error handling on an
  invalid or silent file. Record the sample source and measured latency.
- Update the proposal and README to describe the selected feature source,
  preprocessing, upload flow, model scope, measured results and limitations.
  The old cached-feature timings must not be presented as upload latency or as
  measured training time for this revised representation.

## Delivery order

First establish extraction compatibility and the CSD feature manifest. Then
build the shared preparation/inference transformations, retrain the models,
connect the local video interface, and verify a real uploaded video. The final
report and demonstration should use evidence from this revised pipeline.

## Sources

- [Selected Kaggle package](https://www.kaggle.com/datasets/samarwarsi/cmu-mosei)
- [Kaggle file manifest](https://www.kaggle.com/api/v1/datasets/list/samarwarsi/cmu-mosei)
- [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)
- [COVAREP](https://github.com/covarep/covarep)
- [COVAREP feature extraction](https://github.com/covarep/covarep/blob/master/feature_extraction/COVAREP_feature_extraction.m)
- [OpenFace output format](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format)
- [OpenFace Docker setup](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Docker)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [MMSA feature release](https://github.com/thuiar/MMSA#2-datasets)
