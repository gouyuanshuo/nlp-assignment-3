#!/usr/bin/env bash
# Linux, macOS, and Windows inside WSL2. All ML dependencies run in Docker.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEMO_HOME="${MOSEI_DEMO_HOME:-$ROOT/.demo}"
RELEASE="demo-v1"
REPO="gouyuanshuo/nlp-assignment-3"
ACTION="${1:-serve}"
if [ "$#" -gt 0 ]; then shift; fi
fail() { printf '%s\n' "$*" >&2; exit 1; }

case "$ACTION" in
  setup|doctor|serve) ;;
  *) fail 'Usage: ./demo.sh [setup [download-directory]|doctor|serve]' ;;
esac
if [ "$ACTION" != setup ] && [ ! -f "$DEMO_HOME/demo-release.json" ]; then
  fail 'Demo assets are missing. Run ./demo.sh setup first.'
fi
command -v docker >/dev/null || fail 'Install and start Docker Desktop (or Docker Engine on Linux) first.'
docker info >/dev/null 2>&1 || fail 'Docker is not ready. Start Docker and check that docker info works.'
docker info --format '{{json .DriverStatus}}' | grep -q 'io.containerd.snapshotter' \
  || fail 'Enable the containerd image store in Docker Desktop settings (or Docker Engine) before setup. See docs/classmate-setup.md.'
if [ "$ACTION" = setup ]; then
  DOWNLOAD="${1:-$ROOT/dist/$RELEASE}"
  mkdir -p "$DOWNLOAD" "$DEMO_HOME"
  DOWNLOAD="$(cd "$DOWNLOAD" && pwd -P)"
  if [ "$#" -eq 0 ]; then
    command -v gh >/dev/null || fail 'Install GitHub CLI and run gh auth login, or pass a downloaded bundle directory to ./demo.sh setup.'
    gh release download "$RELEASE" --repo "$REPO" --dir "$DOWNLOAD" --skip-existing
  fi
  [ -f "$DOWNLOAD/SHA256SUMS" ] || fail 'Bundle is incomplete: SHA256SUMS is missing.'
  EXPECTED_FILES="$(cd "$DOWNLOAD" && printf '%s\n' demo-assets.tar.gz demo-release.json demo-images.tar.gz.part-* | LC_ALL=C sort)"
  CHECKED_FILES="$(awk '{print $2}' "$DOWNLOAD/SHA256SUMS" | LC_ALL=C sort)"
  [ "$EXPECTED_FILES" = "$CHECKED_FILES" ] || fail 'Bundle checksum file list does not match the assets. Remove stale parts or download the complete bundle.'
  PART_NUMBER=0
  for PART in "$DOWNLOAD"/demo-images.tar.gz.part-*; do
    [ "$(basename "$PART")" = "$(printf 'demo-images.tar.gz.part-%03d' "$PART_NUMBER")" ] \
      || fail 'Bundle image parts are missing or out of order.'
    PART_NUMBER=$((PART_NUMBER + 1))
  done
  printf 'Verifying downloaded bundle…\n'
  if command -v sha256sum >/dev/null; then
    (cd "$DOWNLOAD" && sha256sum -c SHA256SUMS)
  else
    (cd "$DOWNLOAD" && shasum -a 256 -c SHA256SUMS)
  fi
  printf 'Loading Docker images (this can take several minutes)…\n'
  cat "$DOWNLOAD"/demo-images.tar.gz.part-* | gzip -dc | docker load
  tar -xzf "$DOWNLOAD/demo-assets.tar.gz" -C "$DEMO_HOME"
  cp "$DOWNLOAD/demo-release.json" "$DEMO_HOME/demo-release.json"
  ACTION=doctor
  set --
fi
DEMO_HOME="$(cd "$DEMO_HOME" && pwd -P)"
mkdir -p "$DEMO_HOME/work/home" "$DEMO_HOME/work/tmp"
IMAGE="$(sed -n 's/.*"image": "\([^"]*\)".*/\1/p' "$DEMO_HOME/demo-release.json")"
[[ "$IMAGE" =~ ^sha256:[a-f0-9]{64}$ ]] || fail 'Invalid demo image ID. Run ./demo.sh setup again.'
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail 'The demo runtime image is missing. Run ./demo.sh setup again.'
ENDPOINT="${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}"
case "$ENDPOINT" in unix://*) SOCKET="${ENDPOINT#unix://}" ;; *) fail 'Use a local Docker context; remote Docker daemons cannot read the local model files.' ;; esac
if [ "$(docker info --format '{{.OperatingSystem}}')" = 'Docker Desktop' ]; then
  # Mount the VM socket, not macOS's host-side proxy socket.
  SOCKET=/var/run/docker.sock.raw
else
  [ -S "$SOCKET" ] || fail "Docker socket not found: $SOCKET"
fi
SOCKET_GROUP="$(docker run --rm --platform linux/amd64 --user 0:0 --entrypoint sh \
  --mount "type=bind,source=$SOCKET,target=/var/run/docker.sock" "$IMAGE" \
  -c 'test -S /var/run/docker.sock && stat -c %g /var/run/docker.sock')" \
  || fail 'Cannot access the Docker engine socket from a container. Check Docker Desktop permissions.'
RUN_USER="$(id -u):$(id -g)"
if docker info --format '{{json .SecurityOptions}}' | grep -q rootless; then
  # Container root maps to the invoking host user in a rootless daemon.
  RUN_USER=0:0
fi
PORT="${MOSEI_DEMO_PORT:-7860}"
[[ "$PORT" =~ ^[0-9]+$ ]] && [ "$PORT" -gt 0 ] && [ "$PORT" -lt 65536 ] || fail 'MOSEI_DEMO_PORT must be between 1 and 65535.'
RUN_FLAGS=(--rm --init --platform linux/amd64)
if [ "$ACTION" = serve ]; then
  printf 'Starting demo at http://localhost:%s (Ctrl+C stops it)\n' "$PORT"
  RUN_FLAGS+=(--publish "127.0.0.1:$PORT:7860")
fi
exec docker run "${RUN_FLAGS[@]}" \
  --user "$RUN_USER" --group-add "$SOCKET_GROUP" \
  --mount "type=bind,source=$ROOT,target=$ROOT,readonly" \
  --mount "type=bind,source=$DEMO_HOME,target=$DEMO_HOME,readonly" \
  --mount "type=bind,source=$DEMO_HOME/work,target=$DEMO_HOME/work" \
  --mount "type=bind,source=$SOCKET,target=/var/run/docker.sock" \
  --workdir "$ROOT" \
  --env "PYTHONPATH=$ROOT/src" --env "HOME=$DEMO_HOME/work/home" \
  --env "TMPDIR=$DEMO_HOME/work/tmp" --env "GRADIO_TEMP_DIR=$DEMO_HOME/work/tmp/gradio" \
  "$IMAGE" python "$ROOT/scripts/demo_runtime.py" "$ACTION" --home "$DEMO_HOME" "$@"
