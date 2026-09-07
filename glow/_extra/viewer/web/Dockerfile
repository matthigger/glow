# Multi-demo viewer image, for Google Cloud Run or any Docker host.
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

# --- glow package + baked bundles ----------------------------------------
COPY glow ./glow

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
CMD exec gunicorn glow._extra.viewer.web.server:application \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 120 \
    --max-requests 400 \
    --max-requests-jitter 50 \
    --access-logfile -
