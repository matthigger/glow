"""Multi-demo web server for glow._extra.viewer.

Serves the baked bundles in pickles/ (see bake_demos) off one Flask
server wrapped in a WSGI DispatcherMiddleware. The root / lists the set,
read from manifest.json alone; a bundle is unpickled and given its own
Dash app only when someone asks for it, and at most MAX_MOUNTS stay live.

Loading lazily is what keeps boot cheap: the alternative, one Dash app
per bundle built at import, pays the whole set's unpickle and per-region
DataFrame before the port opens.

Flask forbids registering routes after the first request, so a viewer
cannot be added to the shared server at request time. Hence the
dispatcher: each viewer is its own Dash app added to the mount table at
runtime, with a bound LRU evicting the oldest past MAX_MOUNTS. That table
lives in the worker process, so the deployment runs a single gunicorn
worker (threads, not processes) -- see the Dockerfile.

Bundles are trusted by construction: they are the ones baked into this
image, never user-supplied bytes, so pickle.load's arbitrary-code-
execution risk does not apply here.

A /view/<key>/ request that finds nothing mounted mounts it and
redirects, so a bookmarked link survives the host scaling to zero.

Run locally:
    python -m glow._extra.viewer.web.server            # dev, port 7860
    PORT=8080 python -m glow._extra.viewer.web.server  # custom port

Run under gunicorn (Docker / Cloud Run):
    gunicorn glow._extra.viewer.web.server:application \\
        --bind 0.0.0.0:7860 --workers 1
"""

import argparse
import gzip
import html
import json
import os
import pathlib
import pickle
import sys
import threading
from collections import OrderedDict
from string import Template
from typing import Dict, List, Optional

from flask import Flask, abort, redirect
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from glow._extra.viewer.app import _create_app, _resolve_min_vox
from glow._extra.viewer.web import MANIFEST_NAME


_PICKLE_DIR = pathlib.Path(__file__).parent / 'pickles'

# Live viewers before the least-recently-used one is dropped. Overridable
# by env so the deployment is tuned without a rebuild.
DEFAULT_MAX_MOUNTS = 8

# Sources whose data use terms let the derived maps be shared only with
# recipients bound by those same terms. An anonymous visitor is not, so
# their viewers serve the group mean and withhold the per-subject images
# (see _create_app's per_image). The statistics are unaffected.
GATED_PER_IMAGE_SOURCES = frozenset({'hcp'})

# Region ceiling per viewer, the same default launch() applies. A
# full-brain tree is hundreds of thousands of regions, which is a scatter
# no browser draws smoothly and a layout payload to match; over the
# ceiling the tree is cut to its largest regions. launch() resolves this
# for a local caller, and nothing did it here.
DEFAULT_MAX_REGIONS = 10_000

# cache -> where the paper reports it. Counted off the order of the
# figure environments in publications/submissions/2026_glow.tex, so it
# needs rechecking whenever a figure is added or moved. sweep_b and
# sweep_extent are deliberately absent: the catalogue builds them and no
# current figure reads them.
_CACHE_FIGURES = {
    'segment': 7,
    'prune': 8,
    'null': 9,
    'sweep_llr': 10,
    'runtime_num_vox': 12,
}

# cache -> the question that cache's axis answers.
_CACHE_HEADINGS = {
    'sweep_llr': 'Effect strength',
    'null': 'No effect',
    'sweep_b': 'Imaging features',
    'sweep_extent': 'Effect extent',
    'segment': 'Ward projection',
    'prune': 'Selection rule',
    'runtime_num_vox': 'Volume',
}

# cache -> what a reader is looking at once they pick it.
_CACHE_BLURBS = {
    'sweep_llr': 'Detection against planted effect strength: the power '
                 'curve, from the weakest effect on the grid to the '
                 'strongest.',
    'null': 'Nothing planted. What the test reports when there is no '
            'effect to find, which is what calibrates it.',
    'sweep_b': 'Detection against the number of imaging features the '
               'response carries.',
    'sweep_extent': 'Detection against how much of the analysis volume '
                    'the planted effect covers.',
    'segment': 'What the Ward tree is built on, with the permutation '
               'test held fixed.',
    'prune': 'How the significant set is read out of one shared fit, so '
             'the comparison isolates the rule.',
    'runtime_num_vox': 'Volume, from a small crop up to the full brain.',
}

# The parameters the picker offers, in the order it shows them. The kind
# decides only how a value is rendered: 'enum' is looked up in
# _VALUE_LABELS, 'int' takes thousands separators, 'num' prints as it
# arrives.
_PARAM_SPEC = [
    ('source', 'Image source', 'enum'),
    ('effect_llr', 'Effect strength (LLR)', 'num'),
    ('n_vox_frac', 'Effect extent', 'num'),
    ('b', 'Imaging features (b)', 'int'),
    ('num_img', 'Subjects', 'int'),
    ('num_vox', 'Voxels', 'int'),
    ('cluster_mode', 'Ward projection', 'enum'),
    ('prune_rule', 'Selection rule', 'enum'),
    ('keep_stat', 'Permutation histogram', 'enum'),
    ('seed', 'Random seed', 'int'),
]

# The few things a reader almost always wants to see first, one click
# each, in the order they answer "does this method work?". A shortcut
# naming a bundle this build did not bake is dropped rather than shown
# broken.
_SHORTCUTS = [
    ('llr_strong', 'A strong effect, found'),
    ('llr_weak', 'A weak effect, missed'),
    ('null', 'No effect planted at all'),
    ('hcp_llr_moderate', 'Real HCP diffusion maps'),
    ('vox_1k', 'With the permutation histogram'),
    ('vox_full_brain', 'A whole brain'),
]

# Values a reader should not have to decode. Anything absent renders by
# its kind, so only the codes need an entry here. The keys are the
# javascript String() of the value, hence 'true' and 'null'.
_VALUE_LABELS = {
    'source': {'wgn': 'White Gaussian noise',
               'hcp': 'HCP diffusion maps'},
    'cluster_mode': {'NAIVE': 'Naive (raw y)',
                     'GLM_ERROR': 'GLM Error',
                     'FOCUS': 'Focus'},
    'prune_rule': {'greedy': 'Greedy max-LLR set',
                   'single_max': 'Single max-LLR region',
                   'dp': 'Dynamic-programming cut'},
    'keep_stat': {'true': 'included', 'false': 'not included'},
    'effect_llr': {'null': 'none (null case)'},
    'n_vox_frac': {'null': 'n/a'},
}

# Where a code's alphabetical order is not the order to read it in. The
# first value of each dropdown is also its default, so these put the arm
# the paper reports in front: sorting prune_rule as text would open on
# dp, and source on hcp. Anything unlisted sorts numerically or by text.
_VALUE_ORDER = {
    'source': ['wgn', 'hcp'],
    'cluster_mode': ['FOCUS', 'GLM_ERROR', 'NAIVE'],
    'prune_rule': ['greedy', 'single_max', 'dp'],
}


def read_manifest(pickle_dir: pathlib.Path) -> List[dict]:
    """Read the baked set's manifest, keeping only entries present on disk.

    Falls back to a bare directory listing when no manifest was written,
    so a hand-dropped bundle still shows up (unlabelled).

    Args:
        pickle_dir (pathlib.Path): the bundle directory.

    Returns:
        list[dict]: one entry per bundle, each with at least key and blurb.
    """
    if not pickle_dir.exists():
        return []

    path = pickle_dir / MANIFEST_NAME
    if path.exists():
        entries = json.loads(path.read_text()).get('demos', [])
        return [e for e in entries
                if (pickle_dir / f"{e['key']}.p.gz").exists()]

    return [{'key': p.name.removesuffix('.p.gz'), 'blurb': '', 'cache': '',
             'source': '', 'size_bytes': p.stat().st_size}
            for p in sorted(pickle_dir.glob('*.p.gz'))]


def _extract(payload):
    """Pull (ana, exp, mask_target) out of a baked bundle.

    Args:
        payload: the unpickled object; the bake_demos dict, or a bare
            fitted Analysis (which carries no experiment, so the viewer
            cannot render it).

    Returns:
        ana: the fitted analysis to view.
        exp: the experiment it was fit on, or None for a bare payload.
        mask_target: the planted support, or None.
    """
    if isinstance(payload, dict) and 'ana' in payload:
        return payload['ana'], payload.get('exp'), payload.get('mask_target')
    return payload, None, None


def _no_store_on_error(response):
    """Stop the CDN from reusing anything that is not a success.

    Firebase Hosting gives a rewritten response that carries no
    Cache-Control of its own a ten-minute edge lifetime. Two of this
    server's replies are transient by construction and wrong to reuse:
    the 404 for a bundle whose viewer is not mounted yet, and the
    /view/<key>/ self-redirect, which points at itself and turns into a
    redirect loop as soon as the edge answers it instead of the process
    that would have mounted it. Both clear on the next request that
    reaches the origin, so only a 2xx may be cached.

    Args:
        response: the outgoing Flask response.

    Returns:
        response: the same response, marked no-store unless it succeeded.
    """
    if not 200 <= response.status_code < 300:
        response.headers.setdefault('Cache-Control', 'no-store')
    return response


class LocalMounter:
    """Mount per-bundle viewers on a dispatcher, bounded by an LRU.

    On demand, unpickles one baked bundle, builds a viewer Dash app for
    it, and adds it to the dispatcher's mount table. At most max_mounts
    viewers are live; the least-recently-loaded is evicted (its mount
    dropped, freeing the analysis it held) when the cap is exceeded.

    Operation parameters (set at __init__):
        pickle_dir (pathlib.Path): directory of baked bundles.
        max_mounts (int): live viewer cap before LRU eviction.
        max_regions (int): region ceiling handed to each viewer.

    Runtime state:
        application (DispatcherMiddleware | None): set by
            build_application; its .mounts table is what this object adds
            to and evicts from.
        manifest (list[dict]): the baked set, read at construction.
        mounted_ (OrderedDict): key -> mount prefix, in LRU order.
    """

    def __init__(self, pickle_dir, *, max_mounts: int = DEFAULT_MAX_MOUNTS,
                 max_regions: int = DEFAULT_MAX_REGIONS):
        self.pickle_dir = pathlib.Path(pickle_dir)
        self.max_mounts = max_mounts
        self.max_regions = max_regions
        self.application: Optional[DispatcherMiddleware] = None
        self.manifest = read_manifest(self.pickle_dir)

        self._lock = threading.Lock()
        self._by_key: Dict[str, dict] = {e['key']: e for e in self.manifest}
        self.mounted_: 'OrderedDict[str, str]' = OrderedDict()

    def ensure_mounted(self, key: str) -> str:
        """Load, build, and mount the viewer for one bundle; return its path.

        Idempotent: a key already mounted is moved to most-recent and its
        existing prefix returned without reloading.

        Args:
            key (str): a bundle key from the manifest.

        Returns:
            prefix (str): the mount path to redirect the browser to.

        Raises:
            KeyError: key names no bundle in this build.
        """
        if key not in self._by_key:
            raise KeyError(key)

        with self._lock:
            if key in self.mounted_:
                self.mounted_.move_to_end(key)
                return self.mounted_[key]

            path = self.pickle_dir / f'{key}.p.gz'
            with gzip.open(path, 'rb') as f:
                ana, exp, mask_target = _extract(pickle.load(f))

            mount_key = f'/view/{key}'
            min_vox = _resolve_min_vox(ana, None, self.max_regions)
            source = self._by_key[key].get('source')
            app = _create_app(ana, exp, mask_target=mask_target,
                              min_vox=min_vox,
                              per_image=(source not in
                                         GATED_PER_IMAGE_SOURCES),
                              routes_pathname_prefix='/',
                              requests_pathname_prefix=f'{mount_key}/')
            app.server.after_request(_no_store_on_error)
            self.application.mounts[mount_key] = app.server
            self.mounted_[key] = f'{mount_key}/'

            while len(self.mounted_) > self.max_mounts:
                _, old_prefix = self.mounted_.popitem(last=False)
                # drop the mount; with no other reference the Dash app and
                # the analysis it closed over become collectable
                self.application.mounts.pop(old_prefix.rstrip('/'), None)

            return self.mounted_[key]


def _json_for_script(obj) -> str:
    """Serialise obj for embedding in an inline script element.

    Args:
        obj: any json-serialisable value.

    Returns:
        str: its json, with the one sequence that could close the script
            element early neutralised.
    """
    return json.dumps(obj, separators=(',', ':')).replace('</', r'<\/')


def _figure_list(manifest: List[dict]) -> List[dict]:
    """Order the figures the baked set covers, for the picker.

    Args:
        manifest (list[dict]): the baked set.

    Ordered by the paper's own figure numbering, so a reader who came
    looking for a figure finds it where they expect. The caches no
    figure reports come last, said plainly rather than left to look like
    an omission.

    Args:
        manifest (list[dict]): the baked set.

    Returns:
        list[dict]: {cache, heading, blurb, label} per figure, label
            being what the dropdown shows.
    """
    present = set()
    for entry in manifest:
        present.update(entry.get('caches') or [entry.get('cache', '')])
    present.discard('')

    def rank(cache):
        return (_CACHE_FIGURES.get(cache, 10 ** 6), cache)

    figures = []
    for cache in sorted(present, key=rank):
        heading = _CACHE_HEADINGS.get(cache, cache)
        num = _CACHE_FIGURES.get(cache)
        named = f'{heading} ({cache})'
        figures.append({
            'cache': cache, 'heading': heading,
            'blurb': _CACHE_BLURBS.get(cache, ''),
            'label': (f'Figure {num} -- {named}' if num
                      else f'{named} -- not in the paper')})
    return figures


def _picker_demos(manifest: List[dict]) -> List[dict]:
    """Reduce the manifest to the fields the picker's script reads.

    Args:
        manifest (list[dict]): the baked set, every entry carrying params.

    Returns:
        list[dict]: {key, blurb, mb, caches, p} per bundle, where p is
            the parameter dict the dropdowns are built from.
    """
    return [{'key': e['key'],
             'blurb': e.get('blurb', ''),
             'mb': round(e.get('size_bytes', 0) / (1024 ** 2), 1),
             'caches': e.get('caches') or [e.get('cache', '')],
             'p': e['params']}
            for e in manifest]


def _shortcuts_html(manifest: List[dict]) -> str:
    """Render the one-click entry points, skipping any not baked.

    Args:
        manifest (list[dict]): the baked set.

    Returns:
        str: the html, or '' when this build baked none of them.
    """
    have = {e['key'] for e in manifest}
    links = [f'<a href="/load/{html.escape(key)}">{html.escape(label)}</a>'
             for key, label in _SHORTCUTS if key in have]
    if not links:
        return ''
    return ('<h2 class="sc">Start here</h2>\n'
            f'<div class="shortcuts">{"".join(links)}</div>')


def _bundle_list_html(manifest: List[dict]) -> str:
    """Render every bundle as a plain load link, grouped by its cache.

    The picker replaces this for anyone running javascript; it stays as
    the noscript body, and as the whole page for a set whose manifest
    carries no parameters to pick over.

    Args:
        manifest (list[dict]): the baked set.

    Returns:
        str: the html, or a stand-in when nothing is baked.
    """
    by_cache: 'OrderedDict[str, list]' = OrderedDict()
    for entry in manifest:
        by_cache.setdefault(entry.get('cache', ''), []).append(entry)

    blocks = []
    for cache, entries in by_cache.items():
        heading = _CACHE_HEADINGS.get(cache, cache or 'Other')
        rows = []
        for e in entries:
            size_mb = e.get('size_bytes', 0) / (1024 ** 2)
            label = e.get('blurb') or e['key']
            rows.append(
                f'<li><a href="/load/{html.escape(e["key"])}">'
                f'{html.escape(label)}</a> <span class="key">'
                f'({html.escape(e["key"])}, {size_mb:.1f} MB)</span></li>')
        blocks.append(f'<h2>{html.escape(heading)}</h2>\n'
                      f'<ul>\n{chr(10).join(rows)}\n</ul>')

    return ('\n'.join(blocks)
            or '<p><em>no bundles baked into this build</em></p>')


def _picker_html(figures: List[dict]) -> str:
    """Render the picker's static shell; its script fills the rest.

    Args:
        figures (list[dict]): the figure list from _figure_list.

    Returns:
        str: the html for the figure select and the empty containers the
            script populates.
    """
    options = '\n'.join(
        f'    <option value="{html.escape(f["cache"])}">'
        f'{html.escape(f["label"])}</option>'
        for f in figures)
    return ('<div class="picker">\n'
            '  <div class="row">\n'
            '    <span class="tag">Paper figure</span>\n'
            '    <select id="figure">\n'
            f'{options}\n'
            '    </select>\n'
            '  </div>\n'
            '  <p id="figure-blurb" class="blurb"></p>\n'
            '  <div id="params"></div>\n'
            '  <p id="chosen" class="chosen"></p>\n'
            '  <a id="open" class="open">Open viewer</a>\n'
            '</div>')


# The picker runs client-side over the whole manifest rather than asking
# the server per choice: the set is a few kB of json, so a round trip per
# dropdown would buy nothing, and a page that needs no origin behind it
# survives the CDN caching it.
#
# Each dropdown offers only the values reachable given the OTHER choices
# already made, so no combination the picker can reach is one the baked
# set is missing. A parameter the chosen figure holds fixed renders as
# text rather than as a dropdown with one option.
_PICKER_JS = Template("""
const DEMOS = $demos;
const FIGURES = $figures;
const SPEC = $spec;
const LABELS = $labels;
const ORDER = $order;

const figSel = document.getElementById('figure');
const figBlurb = document.getElementById('figure-blurb');
const paramBox = document.getElementById('params');
const openBtn = document.getElementById('open');
const chosen = document.getElementById('chosen');
let sel = {};

function show(name, raw) {
  const map = LABELS[name] || {};
  const key = String(raw);
  if (Object.prototype.hasOwnProperty.call(map, key)) return map[key];
  if (raw === null) return 'n/a';
  const spec = SPEC.filter(function (s) { return s[0] === name; })[0];
  const kind = spec ? spec[2] : '';
  if (kind === 'int') return Number(raw).toLocaleString();
  // a grid built by logspace lands on 0.030000000000000013, and three
  // significant figures is the precision the axis was specified to
  if (kind === 'num') return String(parseFloat(Number(raw).toPrecision(3)));
  return String(raw);
}

function pool() {
  const fig = figSel.value;
  return DEMOS.filter(function (d) { return d.caches.indexOf(fig) >= 0; });
}

function fits(d, skip) {
  for (const k in sel) {
    if (k === skip) continue;
    if (String(d.p[k]) !== sel[k]) return false;
  }
  return true;
}

function render() {
  const here = pool();
  for (const s of SPEC) {
    const n = s[0];
    if (sel[n] === undefined) continue;
    const live = here.some(function (d) { return String(d.p[n]) === sel[n]; });
    if (!live) delete sel[n];
  }

  paramBox.innerHTML = '';
  for (const s of SPEC) {
    const name = s[0];
    const vals = [];
    const seen = Object.create(null);
    for (const d of here) {
      if (!fits(d, name)) continue;
      const k = String(d.p[name]);
      if (seen[k] === undefined) { seen[k] = 1; vals.push(d.p[name]); }
    }
    if (!vals.length) continue;
    const rank = ORDER[name];
    function at(v) {
      const i = rank.indexOf(String(v));
      // a value the order forgot sorts last, never silently first
      return i < 0 ? 1e9 : i;
    }
    vals.sort(function (a, b) {
      if (rank) return at(a) - at(b);
      if (typeof a === 'number' && typeof b === 'number') return a - b;
      return String(a).localeCompare(String(b));
    });

    const row = document.createElement('div');
    row.className = 'row';
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = s[1];
    row.appendChild(tag);

    if (vals.length === 1) {
      const fixed = document.createElement('span');
      fixed.className = 'fixed';
      fixed.textContent = show(name, vals[0]);
      row.appendChild(fixed);
    } else {
      const box = document.createElement('select');
      for (const raw of vals) {
        const opt = document.createElement('option');
        opt.value = String(raw);
        opt.textContent = show(name, raw);
        if (sel[name] === String(raw)) opt.selected = true;
        box.appendChild(opt);
      }
      if (sel[name] === undefined) sel[name] = box.value;
      box.addEventListener('change', function () {
        sel[name] = box.value;
        render();
      });
      row.appendChild(box);
    }
    paramBox.appendChild(row);
  }

  const hit = here.filter(function (d) { return fits(d, null); });
  if (hit.length) {
    openBtn.href = '/load/' + encodeURIComponent(hit[0].key);
    openBtn.classList.remove('off');
    chosen.textContent = hit[0].blurb
      + ' (' + hit[0].key + ', ' + hit[0].mb + ' MB)';
  } else {
    openBtn.removeAttribute('href');
    openBtn.classList.add('off');
    chosen.textContent = 'nothing baked for that combination';
  }
}

function pickFigure() {
  sel = {};
  const f = FIGURES.filter(function (x) {
    return x.cache === figSel.value;
  })[0];
  figBlurb.textContent = f ? f.blurb : '';
  render();
}

figSel.addEventListener('change', pickFigure);
pickFigure();
""")


_LANDING_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLOW Viewer</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px;
         margin: 3rem auto; padding: 0 1rem; line-height: 1.5; color: #222; }
  h1 { margin-bottom: 0.25rem; }
  h2 { font-size: 1.05rem; margin: 1.6rem 0 0.3rem; color: #444; }
  p.lede { color: #555; margin-top: 0; }
  ul { padding-left: 1.25rem; margin: 0.2rem 0; }
  li { margin: 0.3rem 0; }
  a { color: #0066cc; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .key { color: #999; font-family: monospace; font-size: 0.85em; }
  h2.sc { font-size: 0.78rem; text-transform: uppercase; color: #999;
          letter-spacing: 0.05em; margin: 1.9rem 0 0.6rem; }
  .shortcuts { display: flex; flex-wrap: wrap; gap: 0.5rem; }
  .shortcuts a { border: 1px solid #cfe0f5; background: #f2f7fd;
                 color: #0a58a6; padding: 0.32rem 0.8rem;
                 border-radius: 999px; font-size: 0.88rem; }
  .shortcuts a:hover { background: #e3eefb; text-decoration: none; }
  .picker { border: 1px solid #e2e2e2; border-radius: 8px;
            padding: 1.1rem 1.3rem 1.3rem; background: #fbfbfb;
            margin-top: 1.5rem; }
  .row { display: flex; align-items: baseline; gap: 0.75rem;
         margin: 0.45rem 0; }
  .tag { flex: 0 0 12.5rem; color: #444; font-size: 0.92rem; }
  .picker select { flex: 1 1 auto; max-width: 22rem; padding: 0.3rem 0.4rem;
                   font: inherit; font-size: 0.92rem; border: 1px solid #ccc;
                   border-radius: 4px; background: #fff; }
  .fixed { color: #666; font-size: 0.92rem; }
  .blurb { color: #555; font-size: 0.92rem; margin: 0.7rem 0 1.1rem; }
  .chosen { color: #777; font-family: monospace; font-size: 0.82rem;
            margin: 1.1rem 0 0.9rem; }
  a.open { display: inline-block; background: #0066cc; color: #fff;
           padding: 0.5rem 1.1rem; border-radius: 5px; font-size: 0.95rem; }
  a.open:hover { background: #0055aa; text-decoration: none; }
  a.open.off { background: #bbb; pointer-events: none; }
  footer { color: #888; font-size: 0.85em; margin-top: 3rem; }
</style>
</head>
<body>
<h1>GLOW Viewer</h1>
<p class="lede">Every entry is one cell of the benchmark the paper
reports, fitted with the arm it reports. Take a shortcut, or pick the
figure you want and then the point on its axis. Either way the viewer
opens on the region scatter, the volume overlay, the per-region
regression and, where the fit kept its draws, the permutation histogram.
The first load of a bundle takes a moment.</p>

$body

<footer>
GLOW: General Linear models Optimized with Ward's method.<br>
Questions or feedback:
<a href="mailto:mhigger@ccs.neu.edu">mhigger@ccs.neu.edu</a>
</footer>
$script
</body>
</html>""")


def _landing_html(mounter: LocalMounter) -> str:
    """Render the landing page: a figure picker over the baked set.

    Degrades to the plain grouped list of load links whenever the picker
    cannot be built or run -- a manifest with no parameters (a bundle
    dropped in by hand has none), and any browser without javascript.
    Neither may leave a baked bundle unreachable.

    Args:
        mounter (LocalMounter): holds the manifest to render.

    Returns:
        str: the whole page.
    """
    manifest = mounter.manifest
    listing = _bundle_list_html(manifest)
    if not manifest or not all('params' in e for e in manifest):
        return _LANDING_TEMPLATE.substitute(body=listing, script='')

    figures = _figure_list(manifest)
    script = _PICKER_JS.substitute(
        demos=_json_for_script(_picker_demos(manifest)),
        figures=_json_for_script(figures),
        spec=_json_for_script(_PARAM_SPEC),
        labels=_json_for_script(_VALUE_LABELS),
        order=_json_for_script(_VALUE_ORDER))

    body = (f'{_shortcuts_html(manifest)}\n'
            f'<h2 class="sc">Or pick a cell</h2>\n'
            f'{_picker_html(figures)}\n'
            f'<noscript>\n{listing}\n</noscript>')
    return _LANDING_TEMPLATE.substitute(
        body=body, script=f'<script>{script}</script>')


def build_server(pickle_dir=_PICKLE_DIR) -> Flask:
    """Build the Flask server: the landing page and the bundle loader.

    Tolerates a missing or empty bundle dir (the Space still boots and
    says so). The returned server carries its LocalMounter as
    server.mounter; build_application wires that mounter to the
    dispatcher.

    Args:
        pickle_dir: directory of baked .p.gz bundles.

    Returns:
        server (Flask): the configured Flask application, not yet wrapped.
    """
    mounter = LocalMounter(
        pickle_dir,
        max_mounts=int(os.environ.get('GLOW_VIEWER_MAX_MOUNTS',
                                      DEFAULT_MAX_MOUNTS)),
        max_regions=int(os.environ.get('GLOW_VIEWER_MAX_REGIONS',
                                       DEFAULT_MAX_REGIONS)))
    print(f'{len(mounter.manifest)} bundle(s) available from {pickle_dir}')

    server = Flask('glow_viewer_web')
    server.mounter = mounter
    server.after_request(_no_store_on_error)

    @server.route('/')
    def index():
        """Serve the landing page."""
        return _landing_html(mounter)

    @server.route('/health')
    def health():
        """Serve the liveness probe.

        Not /healthz: Cloud Run's frontend answers that path itself with
        its own 404 and never forwards it, so the conventional name is
        the one name that cannot work behind it.
        """
        return 'ok', 200

    @server.route('/load/<key>')
    def load(key):
        """Mount the viewer for one bundle and redirect to it."""
        try:
            prefix = mounter.ensure_mounted(key)
        except KeyError:
            abort(404)
        return redirect(prefix, code=302)

    @server.route('/view/<key>/')
    def view(key):
        """Mount a viewer asked for directly, then hand it the request.

        Only reached when the key is NOT mounted -- the dispatcher serves
        a mounted one before Flask sees it. That happens to a bookmarked
        or shared /view/<key>/ URL whenever the process holding the mount
        is gone: the host scales to zero between visits, the worker
        recycled, or the LRU evicted this key. Without this the link
        would 404 on a cold instance.
        """
        try:
            mounter.ensure_mounted(key)
        except KeyError:
            abort(404)
        return redirect(f'/view/{key}/', code=302)

    return server


def build_application(pickle_dir=_PICKLE_DIR) -> DispatcherMiddleware:
    """Build the WSGI app: the Flask server wrapped in a dispatcher.

    The dispatcher is what lets a viewer be mounted after start-up (Flask
    forbids adding routes to the server after its first request). Links
    the server's LocalMounter to this dispatcher so its mount table is the
    one the mounter adds to and evicts from.

    Args:
        pickle_dir: directory of baked .p.gz bundles.

    Returns:
        application (DispatcherMiddleware): the WSGI entry point.
    """
    server = build_server(pickle_dir)
    application = DispatcherMiddleware(server)
    server.mounter.application = application
    return application


# WSGI entry point: gunicorn loads
# glow._extra.viewer.web.server:application. Built eagerly at import so
# workers don't race on first request; it unpickles nothing.
application = build_application()


def main():
    """Run the dev server (werkzeug) from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('PORT', 7860)),
                        help='port to bind (default 7860 / $PORT)')
    parser.add_argument('--host', default='127.0.0.1',
                        help="bind host (use '0.0.0.0' for Docker)")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    from werkzeug.serving import run_simple
    print(f'\n  glow:viewer:web running at http://{args.host}:{args.port}')
    print('  press Ctrl+C to stop\n')
    run_simple(args.host, args.port, application,
               use_reloader=args.debug, use_debugger=args.debug)


if __name__ == '__main__':
    sys.exit(main())
