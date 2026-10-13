FROM ghcr.io/astral-sh/uv:0.12.16 AS uv
FROM ubuntu:26.04@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7 AS base
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg=7:8.0.1-3ubuntu2 docker.io libgomp1 \
    && rm -rf /var/lib/apt/lists/*
FROM base AS dependencies
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /opt/dependencies
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --extra video --no-dev --no-install-project --python 3.12.14 \
    && uv cache clean && chmod -R a+rX /opt/python /opt/venv
# Same PyTorch release, CPU wheel: the demo never uses CUDA. Keep all feature
# extraction versions from uv.lock; discard CUDA-only dependencies in this stage.
RUN uv pip install --python /opt/venv/bin/python --no-deps \
    'torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=5a6363570c753812540a05eb82380e329469cbe668643e88111414c12627711f' \
    && uv pip list --python /opt/venv/bin/python --format freeze \
       | sed -n -E 's/^(nvidia-[^=]+|cuda-[^=]+|triton)==.*/\1/p' \
       | xargs -r uv pip uninstall --python /opt/venv/bin/python \
    && uv cache clean
FROM base
COPY --from=dependencies /opt/python /opt/python
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 GRADIO_ANALYTICS_ENABLED=False \
    DOCKER_DEFAULT_PLATFORM=linux/amd64 PYTHONDONTWRITEBYTECODE=1
LABEL org.opencontainers.image.title="CMU MOSEI classmate demo runtime"
