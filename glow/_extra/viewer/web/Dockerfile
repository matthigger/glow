# syntax=docker/dockerfile:1.7-labs
# Multi-demo viewer image, for Google Cloud Run or any Docker host.
#
# The syntax line above must stay the first line of the file -- a parser
# directive is only read before any other content -- and pins a BuildKit
# frontend new enough for COPY --exclude, which docker pulls on its own.
# BuildKit is the default builder; a build forced onto the legacy one
# (DOCKER_BUILDKIT=0) ignores the directive and fails on --exclude.
#
# Build context must be the project root (src/), not glow/_extra/viewer/web/,
# so the whole glow/ package is reachable.  Build with:
#
#     # from src/
#     docker build -t glow-viewer-web -f glow/_extra/viewer/web/Dockerfile .
#     docker run --rm -p 7860:7860 glow-viewer-web
#
# Bundles must already exist in glow/_extra/viewer/web/pickles/ at build
# time (run `python -m glow._extra.viewer.web.bake_demos` locally first).

FROM python:3.12-slim

WORKDIR /app

# --- python deps ---------------------------------------------------------
COPY glow/_extra/viewer/web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- baked demo bundles --------------------------------------------------
# Deliberately its own layer, and deliberately above the package: the
# bundles are the bulk of the image and are rebaked once in a blue moon,
# while the code beside them changes every commit.  A layer is invalidated
# by every layer before it, so the rarely-changing half has to come first
# for `docker push` to skip re-uploading it after a code edit.  Do not fold
# these two COPYs back into one.
COPY glow/_extra/viewer/web/pickles ./glow/_extra/viewer/web/pickles

# --- glow package --------------------------------------------------------
# --exclude keeps the bundles out of this layer; its pattern is relative to
# the COPY source (glow/), not to the build context.
COPY --exclude=_extra/viewer/web/pickles glow ./glow

ENV PYTHONPATH=/app
# The default for a local run; Cloud Run overrides PORT at start-up and
# the container has to bind whatever it is handed, hence the shell-form
# CMD below (exec form would pass the string "$PORT" to gunicorn).
ENV PORT=7860
EXPOSE 7860

# Set as an env var to retune without a rebuild:
#   GLOW_VIEWER_MAX_MOUNTS  live viewers before LRU eviction

# ONE worker, several threads: the mount table lives in-process, so a second
# worker process would not see a viewer mounted by the first.  Threads share
# that memory and keep slow Dash callbacks (3D slicer, regression) from
# blocking each other.  --max-requests recycles the worker periodically to
# reclaim memory from evicted mounts.
#
# BuildKit warns JSONArgsRecommended here, about a shell-form CMD leaving
# /bin/sh as PID 1 and swallowing SIGTERM.  `exec` is what answers that:
# it replaces the shell, so gunicorn IS PID 1 and shuts down cleanly when
# the host scales the instance away.  Do not switch this to JSON form --
# that would pass gunicorn the literal string "$PORT".
CMD exec gunicorn glow._extra.viewer.web.server:application \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 120 \
    --max-requests 400 \
    --max-requests-jitter 50 \
    --access-logfile -
