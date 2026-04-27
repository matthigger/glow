# Multi-demo viewer image for HuggingFace Spaces (Docker SDK).
#
# Build context must be the project root (src/), not glow/viewer_web/, so the
# whole glow/ package is reachable.  Build with:
#
#     # from src/
#     docker build -t glow-viewer-web -f glow/viewer_web/Dockerfile .
#     docker run --rm -p 7860:7860 glow-viewer-web
#
# Pickles must already exist in glow/viewer_web/pickles/ at build time
# (run `python -m glow.viewer_web.bake_demos` locally first).

FROM python:3.12-slim

WORKDIR /app

# --- python deps ---------------------------------------------------------
COPY glow/viewer_web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- glow package + baked demo pickles -----------------------------------
COPY glow ./glow

ENV PYTHONPATH=/app
ENV PORT=7860
EXPOSE 7860

# 2 workers is plenty for free-tier HF Spaces (2 vCPU); each worker holds
# all 5 Dash apps in memory.  --timeout is bumped from gunicorn's 30s
# default because some Dash callbacks (3D slicer, regression) can be slow.
CMD ["gunicorn", "glow.viewer_web.server:application", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-"]
