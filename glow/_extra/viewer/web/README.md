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
  before a deploy — a dev checkout satisfies imports the image cannot.

Loading lazily is what keeps boot cheap: one Dash app per bundle built at
import would pay the whole set's unpickle and per-region DataFrame before
the port opens. A `/view/<key>/` URL that arrives with nothing mounted —
a bookmark, after the host has scaled to zero — mounts itself.

| var | meaning |
|-----|---------|
| `GLOW_VIEWER_MAX_MOUNTS` | live viewers before LRU eviction |
| `PORT` | port to bind; Cloud Run sets this, default 7860 |

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

# confirm the image would boot (blocks torch, benchmark, anything
# missing from requirements.txt)
python -m glow._extra.viewer.web.check_space
```

## Deploy: Google Cloud Run

A Dash app needs a live Python process, so this deploys as a container.
Cloud Run's free tier (2M requests, 180k vCPU-s, 360k GiB-s per month)
covers a demo comfortably, and the service scales to zero between
visitors — an idle month bills only for the image in Artifact Registry.
A GCP project with billing enabled is still required to exist.

```bash
# from src/ (project root)
python -m glow._extra.viewer.web.bake_demos     # bundles go in the image
glow/_extra/viewer/web/deploy_cloud_run.sh --project <gcp-project-id>

# build the image without pushing, to run it locally first
glow/_extra/viewer/web/deploy_cloud_run.sh --project <id> --build-only
docker run --rm -p 7860:7860 <image>
```

The script bakes nothing and deploys nothing until `check_space` passes.
Its Cloud Run flags, and why each one is set, are commented at the call.

Two context filters matter and are easy to get wrong:

- `.dockerignore` (repo root — Docker reads the **context** root, not the
  directory holding the Dockerfile) keeps the ~1 GB `.git` out of the
  build.
- `.gcloudignore` (repo root) must exist. Without it `gcloud` generates a
  default that imports `.gitignore`, and `pickles/` is gitignored — the
  bundles would be dropped and the deployed viewer would list nothing.

## The HuggingFace Space

The Space is a **static** landing page that links to the Cloud Run demo,
not the viewer itself: a Dash app needs a live Python process, which only
a Docker Space provides, and those require a paid HF plan. Static Spaces
are free, so the Space keeps the discoverability and Cloud Run does the
serving.

`deploy_hf.sh` writes exactly two files — `hf_space_card.md` (the Space's
YAML front matter) as `README.md`, and `hf_index.html` with the demo URL
substituted in. It clears the checkout first, so a Space left over from
the old Docker deploy sheds the package and bundles it used to carry.

```bash
glow/_extra/viewer/web/deploy_hf.sh --user <hf-username> \
    --url "$(gcloud run services describe glow-viewer \
             --region us-central1 --format 'value(status.url)')"
```
