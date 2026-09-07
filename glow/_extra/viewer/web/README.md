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

Interactive viewer for a representative set of the paper's benchmark cells:
one fitted GLOW analysis per link, each a point on an axis some figure
sweeps.

## Architecture

- `bake_demos.py` builds each entry in its `DEMOS` list — a cell of
  `glow._extra.benchmark.config.CONFIG` — fits the reported GLOW recipe on
  it, and writes `{ana, exp, mask_target, demo}` to `pickles/<key>.p.gz`
  plus a `manifest.json` describing the set.
- `server.py` boots one Flask server (wrapped in a `DispatcherMiddleware`),
  serves a landing page read from `manifest.json` alone, and mounts a
  `glow._extra.viewer` Dash app per bundle **on first request**, bounded by
  an LRU.
- `play.py` loads a single bundle through the unmodified single-analysis
  `launch()` for local round-trip checks.
- `check_space.py` boots the server with only `requirements.txt` available
  and `benchmark/` excluded, which is what the image actually has. Run it
  before a deploy — a dev checkout satisfies imports the Space cannot.

Loading lazily is what keeps boot cheap: one Dash app per bundle built at
import would pay the whole set's unpickle and per-region DataFrame before
the port opens.

| var | meaning |
|-----|---------|
| `GLOW_VIEWER_MAX_MOUNTS` | live viewers before LRU eviction |

## What is baked

`DEMOS` reads every axis value off `benchmark.config`, so an entry is the
same cell the corresponding figure reports and moves with the paper's grid.
The analysis is the shipped recipe (`config.REPORTED_GLOW_LABEL` — the
per-perm arm on the Focus projection, greedy selection) with only the knob
a given entry varies overridden.

The set covers effect strength, the null, feature count, effect extent,
Ward projection, selection rule, and analysis volume. The moderate-effect
Focus fit is the hub the other entries sit one step away from.

**HCP entries are gated behind `--hcp` and must not be published.** The
maps are DUA-restricted, so an HCP-derived bundle is fine to view locally
and not fine to host. WGN entries carry no such restriction.

## Hosting

A Dash app needs compute, so this is a Docker Space. Docker Spaces require
a paid plan (PRO for personal accounts); only Static Spaces are free. On
the free **CPU Basic** hardware a Space gets 16 GB RAM, 2 vCPU and 50 GB of
ephemeral disk, which the set above does not come close to filling — the
binding constraint is editorial, not technical.

## Local development

```bash
# see the plan without building anything
python -m glow._extra.viewer.web.bake_demos --list

# bake the set (WGN only; add --hcp for the gated entries)
python -m glow._extra.viewer.web.bake_demos

# bake or refit one entry
python -m glow._extra.viewer.web.bake_demos --only llr_moderate --force

# spot-check one bundle through the single-analysis viewer
python -m glow._extra.viewer.web.play llr_moderate

# run the multi-demo server on port 7860
python -m glow._extra.viewer.web.server

# confirm the image would boot (blocks torch, numba-if-unlisted, benchmark)
python -m glow._extra.viewer.web.check_space
```

## Docker / HF Spaces

```bash
# from src/ (project root)
python -m glow._extra.viewer.web.bake_demos   # bundles go into the image
docker build -t glow-viewer-web -f glow/_extra/viewer/web/Dockerfile .
docker run --rm -p 7860:7860 glow-viewer-web

# or push the deployable subset to a Space
python -m glow._extra.viewer.web.check_space   # do this first
glow/_extra/viewer/web/deploy_hf.sh --user <hf-username>
```
