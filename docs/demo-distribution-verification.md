# Demo distribution verification

Verified on 13 October 2026, Ubuntu 26.04 x86_64 with Docker Engine 29.8.2 and the containerd image store.

## Automated checks

- Full repository suite: **119 passed** (`.venv/bin/python -m pytest -q`).
- Ruff checks, `git diff --check` and `bash -n demo.sh`: passed.
- Nine distribution tests cover relocated model paths, unchanged feature contracts, out-of-bundle path rejection, missing setup, allowlisted exports, Docker Desktop socket selection, rootless UID mapping, unverified archive parts and incompatible image stores.
- Independent code review identified socket, UID and checksum-list issues; these were corrected and covered by tests.

## Runtime and installation

- Built the `linux/amd64` runtime from the pinned Ubuntu image, exact FFmpeg version, locked dependencies and PyTorch 2.14.1 CPU wheel. The Dockerfile records the CPU wheel SHA-256. CUDA-only dependencies are excluded from the final image.
- Created a separate source-only checkout in a directory containing a space, without a Python environment, research data or trained runs.
- Relocated the model assets into that checkout and passed the actual containerized readiness check. The training feature manifest and fingerprint were unchanged.
- Exported the runtime and both extractor images, BERT/Whisper weights, `best.pt` and run metadata. The downloadable assets total **5,218,221,050 bytes**, plus the checksum text file; image archives are split into three parts smaller than 2 GiB each.
- Ran `demo.sh setup` against the complete exported bundle in the relocated checkout. All five checksums passed; Docker loaded the images and the readiness check passed. This reused the local Docker engine, rather than provisioning another physical computer.
- The test app published only `127.0.0.1:17861` on the host. It was stopped after verification.

## Real upload checks

- Sent the existing 15-second spoken-video fixture through the Gradio upload API to the containerized app. It produced a transcript, a timestamped window prediction and a negative classification with probability approximately 0.9084.
- Processing took approximately **145 seconds** while the image export was also running. This is an integration observation, not an isolated performance benchmark.
- Sent the silent-video fixture through the same API. The app reported no speech and cleared probabilities, transcript and window rows.
- The tests exercised the shared Docker socket and actual COVAREP/OpenFace subprocess containers, not mocked extractors.

## Platform limits

macOS Intel/Apple Silicon, Windows WSL2 and rootless Docker branches have launch-argument regression coverage and documented setup paths. They have **not** been tested on separate physical machines. Apple Silicon requires x86 emulation and may encounter platform-specific performance or instruction-set limitations. Native Windows without WSL2 is outside this release's supported launch path.
