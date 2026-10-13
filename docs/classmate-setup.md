# Run the pretrained video sentiment demo

This route runs the trained Transformer on your own computer. It includes the sentiment checkpoint, BERT and Whisper model weights, and the exact Linux runtime and audio/face extractor images used by the feature contract. No training dataset, retraining, GPU, Python installation or Colab account is required.

## Requirements

- Access to `gouyuanshuo/nlp-assignment-3` on GitHub. It is currently public; if its visibility changes to private, accept the owner’s invitation first.
- Recent Docker Desktop on macOS or Windows, or Docker Engine/Desktop on Linux, with the **containerd image store** enabled. The bundle preserves OCI image identities, so the legacy image store is not supported. Start Docker before setup; `docker info` must work without `sudo`.
- GitHub CLI (`gh`) for authenticated clone and download, or a manually downloaded release bundle.
- Recommended: 16 GB system RAM, at least 8 GB assigned to Docker Desktop, 4 CPUs, and 50 GB free disk for the download, extracted models and Docker images. The first download is about **5.22 GB**; OpenFace accounts for most of the image size. These are starting allocations, not measurements of every supported computer.

### Linux

Use a local Docker daemon with access to your project directory. Rootless Docker is supported through the active Docker context's Unix socket. Remote Docker contexts are not supported because feature extractors read local temporary files.

Docker Engine 29 or later enables the containerd image store on fresh installations. For an upgraded installation, follow [Docker's containerd setup instructions](https://docs.docker.com/engine/storage/containerd/). Switching image stores does not transfer your existing images between them.

### Windows

Install Docker Desktop with its WSL2 backend and enable integration with your Ubuntu WSL distribution. Run every command below in the Ubuntu WSL terminal, not PowerShell or Command Prompt. Clone into a directory such as `~/projects`, inside the Linux filesystem. This avoids Windows filesystem permission and performance differences.

### macOS

Use Terminal and Docker Desktop. Both Intel and Apple Silicon Macs use the same `linux/amd64` images. On Apple Silicon, enable x86 emulation; Docker Desktop's Apple Virtualization framework with Rosetta can improve speed. Allow Docker to share the folder containing the project. Start with a short clip because emulation can make extraction much slower.

In Docker Desktop Settings, enable **Use containerd for pulling and storing images** if it is not already enabled. This also applies to Docker Desktop on Windows.

The packaged image is built and tested on Linux x86_64. See [verification evidence](demo-distribution-verification.md). The launch flow accommodates Docker Desktop on macOS and WSL2, but those platforms have not been physically tested for this release. CPU instruction emulation or Docker file-sharing settings can still require machine-specific adjustment.

## First run

After installing Docker and [GitHub CLI](https://cli.github.com/), run:

```bash
gh auth login
gh repo clone gouyuanshuo/nlp-assignment-3
cd nlp-assignment-3
./demo.sh setup
./demo.sh
```

Setup downloads the `demo-v1` release into `dist/demo-v1`, verifies SHA-256 checksums, loads the images and extracts model files into `.demo/assets`. It then verifies the saved model, feature code, package versions, FFmpeg and Docker image identities. The final command starts the application at **http://localhost:7860**. Startup can take a little time while models load.

Upload an English MP4, MOV or WebM with one visible speaker, at most 60 seconds and 200 MB. The app shows its transcript and positive/negative predictions. Exactly neutral examples were excluded during training, so there is no neutral prediction. Silent clips and clips without enough usable speech or facial information are rejected.

Stop the app with **Ctrl+C**. For subsequent runs, use only `./demo.sh`. Once installed, the runtime needs no model downloads or GitHub authentication to launch.

```bash
./demo.sh doctor                 # Recheck installed assets and environment
MOSEI_DEMO_PORT=7861 ./demo.sh    # Use another localhost port
```

The application container uses the local Docker socket to launch the two feature-extraction containers. Docker socket access is powerful: run the launcher only from a checkout you trust. The web port is published on `127.0.0.1`, not your local network. Videos are processed locally and temporary analysis files are cleaned after each request.

## Manual bundle download

Download **every asset**, including `SHA256SUMS`, `demo-release.json`, `demo-assets.tar.gz`, and all `demo-images.tar.gz.part-*` files, from the [demo-v1 release](https://github.com/gouyuanshuo/nlp-assignment-3/releases/tag/demo-v1). Keep them together in one folder. Then, from the repository:

```bash
./demo.sh setup /absolute/path/to/downloaded-files
./demo.sh
```

This also allows the project owner to share the same files directly with classmates who have the source checkout. Keep the checksum file with the bundle. GitHub's source ZIP alone contains no trained model or runtime images.

## Troubleshooting

| Message or symptom | Action |
|---|---|
| Repository or release not found | Check the repository URL and `gh auth status`. If the repository has become private, accept your invitation first. |
| Docker is not ready | Start Docker Desktop or the Docker service, then run `docker info`. On Windows, enable WSL integration. |
| Checksum fails or download was interrupted | Remove only the failed download file from `dist/demo-v1`, then rerun setup. Downloads already present are not overwritten automatically. |
| Port is already allocated | Run `MOSEI_DEMO_PORT=7861 ./demo.sh` and open port 7861. |
| Mount denied on macOS | Allow the project directory under Docker Desktop file sharing. Use a normal local folder, not a remote drive. |
| Container is killed or runs out of memory | Increase Docker Desktop memory allocation and close other large applications. |
| Illegal instruction or slow processing on Apple Silicon | Check x86 emulation/Rosetta settings. Try a shorter clip. This release has no native ARM extractor build. |
| Feature code or manifest mismatch | Use the code shipped with the release, or rebuild a compatible bundle. Do not disable the compatibility checks. |

Paths containing spaces are supported. Avoid commas in the project path because Docker's bind-mount syntax uses commas as separators.

## Working on the source

The launcher mounts this checkout and imports `src/mosei` from it, so UI changes are picked up on restart. The image uses the locked Python dependencies, with PyTorch 2.14.1's CPU wheel replacing its CUDA build and CUDA-only packages omitted. Feature-extractor package versions remain unchanged. If you change dependencies, rebuild and export the runtime. Changes to the fingerprinted feature-extraction or preprocessing code require a compatible checkpoint and manifest; the demo deliberately rejects mismatches.

For native Python development and research training, follow the separate setup and training sections in the main README. The demo bundle does not contain the training dataset or optimizer state.

## Maintainer export

From the configured author machine, verify the existing model and build the image, then export:

```bash
docker build --platform linux/amd64 -t mosei-demo:20261013 -f tools/demo/Dockerfile .
.venv/bin/python scripts/build_demo_bundle.py --output dist/demo-v1
./demo.sh setup "$PWD/dist/demo-v1"
```

The exporter includes only model files named and hashed in the feature manifest, `best.pt`, and `run.json`. It rewrites only the model locations, leaving the feature contract unchanged. It exports the exact runtime and extractor image IDs and splits the compressed image archive into files below GitHub's 2 GiB per-asset limit. Publish `SHA256SUMS`, `demo-release.json`, `demo-assets.tar.gz`, and all image parts; the export's staging directory and unsplit image archive are local intermediates.

The supplied assets are for this academic project. BERT, faster-whisper/CTranslate2, COVAREP and OpenFace retain their upstream licenses; consult those projects before redistributing or using the bundle outside the course.

References: [Docker Desktop installation](https://docs.docker.com/desktop/), [Docker Desktop settings and Rosetta](https://docs.docker.com/desktop/settings-and-maintenance/settings/), [GitHub release asset limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
