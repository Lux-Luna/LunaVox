# syntax=docker/dockerfile:1.7
#
# LunaVox CPU runtime image.
#
# Multi-stage build:
#
#   Stage 1 (builder)  — full toolchain: cmake, ninja, g++, Python build
#                        tools, HuggingFace Hub client. Downloads the
#                        Linux CPU ONNX Runtime + llama.cpp binaries via
#                        `lunavox build libs --platform linux_cpu` and
#                        compiles the C++ engine via `lunavox build`.
#                        Everything needed for a runtime image lands
#                        under /src/build and /src/lib.
#
#   Stage 2 (runtime)  — slim Python base with only the runtime OS
#                        libraries (libgomp1 / libstdc++6) that ONNX
#                        Runtime needs. Pip-installs lunavox from PyPI
#                        (so we get the exact 2.2.0 release), copies
#                        the prebuilt C++ artifacts from stage 1, and
#                        marks /app as a deployment-layout project
#                        root via the .lunavox-root sentinel file.
#
# Models are NOT baked in — they're large (hundreds of MB per variant)
# and come from a separate download step. Mount them at /app/models
# via a volume:
#
#     docker run --rm -p 8000:8000 \
#         -v $(pwd)/models:/app/models \
#         lunavox:2.2.0
#
# Or use the companion compose.yml which does the mount for you.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# ────────────────────────────────────────────────────────────────────────
# Stage 1: builder
# ────────────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS builder

ARG LUNAVOX_PLATFORM=linux_cpu

# Build toolchain. ninja is the recommended cmake generator per LunaVox
# docs; we pin to apt's shipped version to keep the image reproducible.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        curl \
        cmake \
        ninja-build \
        g++ \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy just enough to run `pip install -e .` first so the dependency
# layer caches well. We bring in the full source afterwards for the
# C++ compile step.
COPY pyproject.toml README.md ./
COPY src/lunavox/__init__.py src/lunavox/__init__.py
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir ".[serve]" huggingface_hub

COPY . .

# Editable install is required so `lunavox build` can find the source
# tree as a dev checkout. The --no-deps flag avoids re-fetching the
# PyPI deps we already installed above.
RUN python -m pip install --no-cache-dir --no-deps -e .

# Download Linux CPU runtime libraries (ONNX Runtime + llama.cpp).
# The --yes flag auto-confirms all prompts so the build is
# non-interactive.
RUN lunavox --yes build libs --platform ${LUNAVOX_PLATFORM}

# Compile the C++ engine. The build emits liblunavox.so and
# lunavox-cli into /src/build/.
RUN lunavox --yes build --clean --j 4


# ────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
# ────────────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime

# Runtime-only dependencies. libgomp1 is OpenMP (ONNX Runtime CPU EP),
# libstdc++6 is the C++ standard library. dumb-init gives us a tiny
# PID-1 that forwards SIGTERM so `docker stop` shuts uvicorn cleanly.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        dumb-init \
 && rm -rf /var/lib/apt/lists/*

# Non-root user for safer containers. Uid 10001 stays out of the
# common Linux reserved range so volume-mount permission collisions
# are rare.
RUN groupadd --gid 10001 lunavox \
 && useradd --uid 10001 --gid 10001 --no-create-home --shell /sbin/nologin lunavox

WORKDIR /app

# Install LunaVox from PyPI. Pinning to the image tag keeps the
# runtime dependencies in lockstep with whatever version was tested.
ARG LUNAVOX_VERSION=2.2.0
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir "lunavox==${LUNAVOX_VERSION}"

# Copy the built C++ artifacts from the builder stage. The layout
# mirrors what `lunavox build` produces in a dev checkout so
# lunavox.runtime._capi finds liblunavox.so at /app/build/ via the
# LUNAVOX_PROJECT_ROOT env var below.
COPY --from=builder /src/build /app/build
COPY --from=builder /src/lib /app/lib

# Create the deployment-layout marker so lunavox.core.project.resolve_project_root
# accepts /app as a valid root even though there's no CMakeLists.txt
# or src/ here.
RUN mkdir -p /app/models /app/output /app/ref \
 && printf "lunavox %s deployment image\n" "${LUNAVOX_VERSION}" > /app/.lunavox-root \
 && chown -R lunavox:lunavox /app

USER lunavox

# LUNAVOX_PROJECT_ROOT tells resolve_project_root to trust /app even
# when the current working directory doesn't look like a checkout.
ENV LUNAVOX_PROJECT_ROOT=/app \
    LUNAVOX_LIB_PATH=/app/build/liblunavox.so \
    LUNAVOX_LIB_DIR=/app/build \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Default command: start the server listening on 0.0.0.0 so the
# container's port is reachable from the host. --batch-size auto
# picks a pool size based on container memory limits. Override any
# of these via `docker run ... lunavox:2.2.0 serve --batch-size 1`.
ENTRYPOINT ["dumb-init", "--"]
CMD ["lunavox", "serve", "--host", "0.0.0.0", "--port", "8000", "--batch-size", "auto"]
