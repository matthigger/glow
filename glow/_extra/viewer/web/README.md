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

- `bake_demos.py` builds a curated set of `AnalysisGLOWSplit` pickles from the
  builders in `glow._extra.viewer.__main__`, written to `pickles/`.
- `server.py` boots a single Flask server (wrapped in a
  `DispatcherMiddleware`), mounts one `glow._extra.viewer` Dash app per baked pickle
  at `/{key}/`, and serves a small landing page at `/`.
- `zenodo.py` fetches individual pickles from one published Zenodo record.
- `play.py` loads a single pickle through the unmodified single-analysis
  `launch()` for local round-trip checks.

## Zenodo browser (load any published experiment)

The full benchmark is too large to host or load on a Space (tens-to-hundreds
of GB; a Space has ≤32 GB RAM). Instead, the heavy precomputed experiments
live on **Zenodo** (durable, DOI'd) and the Space loads **one at a time** on
demand: `/zenodo/` lists the configured record's files; picking one downloads
just that pickle (size-capped, MD5-verified, cached), mounts a fresh viewer
under `/zenodo/view/<slug>/`, and redirects there. An LRU keeps at most
`GLOW_ZENODO_MAX_MOUNTS` viewers live.

**Security:** only files of the configured record id are ever fetched and
unpickled — the record id is an allowlist. The server never unpickles
user-uploaded bytes (`pickle.load` on untrusted input is arbitrary code
execution), which is why this fetches by record id rather than accepting an
upload.

Configure via env (HF Space **Variables**, not Secrets):

| var | meaning |
|-----|---------|
| `GLOW_ZENODO_RECORD_ID` | record to browse; unset = browser disabled |
| `GLOW_ZENODO_API_BASE`  | `https://sandbox.zenodo.org/api` to test on Sandbox |
| `GLOW_ZENODO_MAX_MB`    | per-file download cap (default 64) |
| `GLOW_ZENODO_MAX_MOUNTS`| live viewers before LRU eviction (default 6) |

```bash
# test locally against a Zenodo Sandbox deposit before a real DOI exists
GLOW_ZENODO_API_BASE=https://sandbox.zenodo.org/api \
GLOW_ZENODO_RECORD_ID=123456 \
python -m glow._extra.viewer.web.server
```

## Local development

```bash
# bake the curated demo set (~1 min)
python -m glow._extra.viewer.web.bake_demos

# spot-check one pickle through the existing single-analysis viewer
python -m glow._extra.viewer.web.play wgn2d_b1_medium_s0

# run the multi-demo server on port 7860
python -m glow._extra.viewer.web.server
```

## Docker / HF Spaces

```bash
# from src/ (project root)
python -m glow._extra.viewer.web.bake_demos          # pickles are baked into the image
docker build -t glow-viewer-web -f glow/_extra/viewer/web/Dockerfile .
docker run --rm -p 7860:7860 glow-viewer-web
```

## Curated demos

The combo list lives in `bake_demos.py::COMBOS`. To grow or shrink the
demo library, edit that list and re-bake.
