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
from typing import Dict, List, Optional

from flask import Flask, abort, redirect
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from glow._extra.viewer.app import _create_app, _resolve_min_vox
from glow._extra.viewer.web import MANIFEST_NAME


_PICKLE_DIR = pathlib.Path(__file__).parent / 'pickles'

# Live viewers before the least-recently-used one is dropped. Overridable
# by env so the deployment is tuned without a rebuild.
DEFAULT_MAX_MOUNTS = 8

# Region ceiling per viewer, the same default launch() applies. A
# full-brain tree is hundreds of thousands of regions, which is a scatter
# no browser draws smoothly and a layout payload to match; over the
# ceiling the tree is cut to its largest regions. launch() resolves this
# for a local caller, and nothing did it here.
DEFAULT_MAX_REGIONS = 10_000

# cache -> the question that cache's axis answers, for the landing page
_CACHE_HEADINGS = {
    'sweep_llr': 'Effect strength',
    'null': 'No effect',
    'sweep_b': 'Imaging features',
    'sweep_extent': 'Effect extent',
    'segment': 'Ward projection',
    'prune': 'Selection rule',
    'runtime_num_vox': 'Volume',
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
            app = _create_app(ana, exp, mask_target=mask_target,
                              min_vox=min_vox,
                              routes_pathname_prefix='/',
                              requests_pathname_prefix=f'{mount_key}/')
            self.application.mounts[mount_key] = app.server
            self.mounted_[key] = f'{mount_key}/'

            while len(self.mounted_) > self.max_mounts:
                _, old_prefix = self.mounted_.popitem(last=False)
                # drop the mount; with no other reference the Dash app and
                # the analysis it closed over become collectable
                self.application.mounts.pop(old_prefix.rstrip('/'), None)

            return self.mounted_[key]


def _landing_html(mounter: LocalMounter) -> str:
    """Render the landing page: every baked bundle, grouped by its axis."""
    by_cache: 'OrderedDict[str, list]' = OrderedDict()
    for entry in mounter.manifest:
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

    body = ('\n'.join(blocks)
            or '<p><em>no bundles baked into this build</em></p>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GLOW Viewer</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px;
         margin: 3rem auto; padding: 0 1rem; line-height: 1.5; color: #222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.05rem; margin: 1.6rem 0 0.3rem; color: #444; }}
  p.lede {{ color: #555; margin-top: 0; }}
  ul {{ padding-left: 1.25rem; margin: 0.2rem 0; }}
  li {{ margin: 0.3rem 0; }}
  a {{ color: #0066cc; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .key {{ color: #999; font-family: monospace; font-size: 0.85em; }}
  footer {{ color: #888; font-size: 0.85em; margin-top: 3rem; }}
</style>
</head>
<body>
<h1>GLOW Viewer</h1>
<p class="lede">One fitted analysis per link, each a cell of the benchmark
the paper reports, grouped by the axis it sits on. Pick one to explore the
region scatter, the volume overlay, the per-region regression and the
permutation histogram. The first click on a bundle loads it, which takes a
moment.</p>

{body}

<footer>
GLOW: General Linear models Optimized with Ward's method.<br>
Questions or feedback:
<a href="mailto:mhigger@ccs.neu.edu">mhigger@ccs.neu.edu</a>
</footer>
</body>
</html>"""


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

    @server.route('/')
    def index():
        """Serve the landing page."""
        return _landing_html(mounter)

    @server.route('/healthz')
    def healthz():
        """Serve the liveness probe."""
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
