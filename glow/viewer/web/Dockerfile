# Multi-demo viewer image for HuggingFace Spaces (Docker SDK).
#
# Build context must be the project root (src/), not glow/viewer/web/, so the
# whole glow/ package is reachable.  Build with:
#
#     # from src/
#     docker build -t glow-viewer-web -f glow/viewer/web/Dockerfile .
#     docker run --rm -p 7860:7860 glow-viewer-web
#
# Pickles must already exist in glow/viewer/web/pickles/ at build time
# (run `python -m glow.viewer.web.bake_demos` locally first).

FROM python:3.12-slim

WORKDIR /app

# --- python deps ---------------------------------------------------------
COPY glow/viewer/web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- glow package + baked demo pickles -----------------------------------
COPY glow ./glow

ENV PYTHONPATH=/app
ENV PORT=7860
EXPOSE 7860

# Zenodo browser config (set these as HF Space "Variables", not Secrets --
# a public record id is not sensitive).  Unset GLOW_ZENODO_RECORD_ID just
# disables the browser; the curated demos still serve.
#   GLOW_ZENODO_RECORD_ID   record to browse (pin a version-specific id)
#   GLOW_ZENODO_API_BASE    https://sandbox.zenodo.org/api to test on Sandbox
#   GLOW_ZENODO_MAX_MB      per-file download cap (default 64)
#   GLOW_ZENODO_MAX_MOUNTS  live viewers before LRU eviction (default 6)

# ONE worker, several threads: the Zenodo mount table lives in-process, so a
# second worker process would not see a viewer mounted by the first.  Threads
# share that memory and keep slow Dash callbacks (3D slicer, regression) from
# blocking each other.  --max-requests recycles the worker periodically to
# reclaim memory from evicted/abandoned mounts.
CMD ["gunicorn", "glow.viewer.web.server:application", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--max-requests", "400", \
     "--max-requests-jitter", "50", \
     "--access-logfile", "-"]
