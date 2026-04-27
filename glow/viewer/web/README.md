---
title: GLOW Viewer
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# GLOW Viewer (web demo)

Interactive demos of the GLOW hierarchical-segmentation viewer.

## Architecture

- `bake_demos.py` builds a curated set of `AnalysisGLOW` pickles from the
  builders in `glow.viewer.__main__`, written to `pickles/`.
- `server.py` boots a single Flask server, mounts one `glow.viewer` Dash app
  per pickle at `/{key}/`, and serves a small landing page at `/`.
- `play.py` loads a single pickle through the unmodified single-analysis
  `launch()` for local round-trip checks.

## Local development

```bash
# bake the curated demo set (~1 min)
python -m glow.viewer.web.bake_demos

# spot-check one pickle through the existing single-analysis viewer
python -m glow.viewer.web.play wgn2d_b1_medium_s0

# run the multi-demo server on port 7860
python -m glow.viewer.web.server
```

## Docker / HF Spaces

```bash
# from src/ (project root)
python -m glow.viewer.web.bake_demos          # pickles are baked into the image
docker build -t glow-viewer-web -f glow/viewer/web/Dockerfile .
docker run --rm -p 7860:7860 glow-viewer-web
```

## Curated demos

The combo list lives in `bake_demos.py::COMBOS`. To grow or shrink the
demo library, edit that list and re-bake.
