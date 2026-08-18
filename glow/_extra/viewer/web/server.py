"""Multi-demo web server for glow._extra.viewer.

Serves two things off one Flask server, wrapped in a WSGI
DispatcherMiddleware so apps can be mounted after start-up:

Curated demos -- every baked pickle in pickles/ is loaded at boot into its
own Dash app, mounted under /{key}/ (see
glow._extra.viewer.app._create_app). The root / serves a landing page.

Zenodo browser -- /zenodo/ lists the files of one pre-configured Zenodo
record (GLOW_ZENODO_RECORD_ID); picking one downloads that single pickle,
mounts a fresh viewer for it under /zenodo/view/<slug>/, and redirects
there. Only files of the configured record are ever fetched and unpickled
-- the record id is an allowlist, so the server never deserialises an
uploaded (untrusted) pickle (that would be arbitrary-code-execution; see
README.md).

Flask forbids registering routes after the first request, so a per-file
viewer cannot be mounted onto the shared server at request time. Hence the
dispatcher: each viewer is its own Dash app, added to the mount table at
runtime, with a bound LRU evicting the oldest past MAX_MOUNTS. That table
lives in the worker process, so the deployment runs a single gunicorn
worker (threads, not processes) -- see the Dockerfile.

Run locally:
    python -m glow._extra.viewer.web.server            # dev, port 7860
    PORT=8080 python -m glow._extra.viewer.web.server  # custom port

Run under gunicorn (Docker / HF Spaces):
    gunicorn glow._extra.viewer.web.server:application \\
        --bind 0.0.0.0:7860 --workers 1
"""

import argparse
import gzip
import html
import os
import pathlib
import pickle
import re
import sys
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from flask import Flask, abort, redirect
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from glow._extra.viewer.app import _create_app
from glow._extra.viewer.web import zenodo


_PICKLE_DIR = pathlib.Path(__file__).parent / 'pickles'

# Defaults overridable by env so the Space is configured without a rebuild.
DEFAULT_MAX_MOUNTS = 6
_FILE_LIST_TTL = 300

# human-readable labels for the landing page
_IMAGE_SET_LABELS = {
    'wgn2d':    '2D White Gaussian Noise',
    'wgn3d':    '3D White Gaussian Noise',
    'mandrill': '2D Mandrill (RGB)',
    'dti2d':    '2D Axial-slice DTI (HCP)',
    'dti3d':    '3D DTI (HCP)',
}
_FEATURE_LABELS = {
    'fa':    'Fractional Anisotropy',
    'md':    'Mean Diffusivity',
    'all':   'all features',
    'red':   'red channel',
    'green': 'green channel',
    'blue':  'blue channel',
}


def _describe_combo(combo):
    """Build a human-readable label for a combo dict."""
    parts = [_IMAGE_SET_LABELS.get(combo['image_set'], combo['image_set'])]
    if 'b' in combo:
        parts.append(f"b={combo['b']}")
    if 'features' in combo:
        parts.append(_FEATURE_LABELS.get(combo['features'], combo['features']))
    parts.append(f"{combo['severity']} effect")
    return ', '.join(parts)


def _load_registry(pickle_dir):
    """Load every *.p.gz in pickle_dir, keyed by file stem."""
    registry = {}
    for path in sorted(pickle_dir.glob('*.p.gz')):
        key = path.name.removesuffix('.p.gz')
        with gzip.open(path, 'rb') as f:
            registry[key] = pickle.load(f)
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f'  loaded {key} ({size_mb:.1f} MB)')
    return registry


def _extract_ana(payload) -> Tuple[object, object, object]:
    """Pull (ana, exp, mask_target) from a loaded pickle payload.

    Accepts the baked-demo dict ({'ana', 'exp', 'mask_target', 'combo'}) or
    a bare fitted Analysis object (which carries no experiment, so the
    viewer cannot render it).

    Args:
        payload: the unpickled object.

    Returns:
        ana: the fitted analysis to view.
        exp: the experiment it was fit on (None for a bare-object payload).
        mask_target: the target mask, or None.
    """
    if isinstance(payload, dict) and 'ana' in payload:
        return payload['ana'], payload.get('exp'), payload.get('mask_target')
    return payload, None, None


def _open_pickle(path: pathlib.Path):
    """Unpickle a (possibly gzipped) file, sniffing the gzip magic bytes.

    SAFE BY CONSTRUCTION: the only callers pass either a baked demo from
    pickles/ or a file just downloaded from the allowlisted Zenodo record
    (zenodo.fetch_file). No user-supplied bytes reach this function, so the
    pickle.load arbitrary-code-execution risk does not apply.
    """
    with open(path, 'rb') as f:
        head = f.read(2)
    opener = gzip.open if head == b'\x1f\x8b' else open
    with opener(path, 'rb') as f:
        return pickle.load(f)


def _slugify(key: str) -> str:
    """Turn a Zenodo filename into a URL-path-safe slug."""
    stem = re.sub(r'\.p\.gz$|\.pkl$|\.pickle$', '', key)
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '-', stem).strip('-')
    return slug or 'file'


class ZenodoMounter:
    """Mount per-file Zenodo viewers on a dispatcher, bounded by an LRU.

    Lists one configured record's files, and on demand downloads a single
    pickle, builds a viewer Dash app for it, and adds it to the
    dispatcher's mount table. At most max_mounts viewers are live at once;
    the least-recently-loaded is evicted (its mount dropped, freeing the
    analysis it held) when the cap is exceeded.

    Operation parameters (set at __init__):
        record_id (str | None): Zenodo record id to browse; None disables
            the browser.
        max_bytes (int): per-file download cap.
        max_mounts (int): live viewer cap before LRU eviction.

    Runtime state:
        application (DispatcherMiddleware | None): set by build_application;
            its .mounts table is what this object adds to and evicts from.
        mounted_ (OrderedDict): slug -> mount prefix, LRU order.
    """

    def __init__(self, record_id: Optional[str], *, max_bytes: int,
                 max_mounts: int = DEFAULT_MAX_MOUNTS):
        self.record_id = record_id or None
        self.max_bytes = max_bytes
        self.max_mounts = max_mounts
        self.application: Optional[DispatcherMiddleware] = None

        self._lock = threading.Lock()
        self._files_by_slug: Dict[str, zenodo.ZenodoFile] = {}
        self._files_at = 0.0
        self.mounted_: 'OrderedDict[str, str]' = OrderedDict()

    @property
    def enabled(self) -> bool:
        """True when a record id is configured and a dispatcher is attached."""
        return self.record_id is not None and self.application is not None

    def list_files(self, *, force: bool = False) -> List[zenodo.ZenodoFile]:
        """List the record's files, cached for _FILE_LIST_TTL seconds.

        Args:
            force (bool): refetch even if the cached listing is fresh.

        Returns:
            files (list): ZenodoFile, empty if the browser is disabled or
                the record cannot be read.
        """
        if not self.record_id:
            return []
        now = time.time()
        if (force or now - self._files_at > _FILE_LIST_TTL
                or not self._files_by_slug):
            files = zenodo.list_record_files(self.record_id)
            by_slug: Dict[str, zenodo.ZenodoFile] = {}
            for f in files:
                slug = _slugify(f.key)
                while slug in by_slug:
                    slug += '-x'
                by_slug[slug] = f
            self._files_by_slug = by_slug
            self._files_at = now
        return list(self._files_by_slug.values())

    def slug_items(self) -> List[Tuple[str, zenodo.ZenodoFile]]:
        """Return (slug, ZenodoFile) pairs in record order."""
        self.list_files()
        return list(self._files_by_slug.items())

    def ensure_mounted(self, slug: str) -> str:
        """Download, build, and mount the viewer for one slug; return its path.

        Idempotent: a slug already mounted is moved to most-recent and its
        existing prefix returned without refetching.

        Args:
            slug (str): the per-file slug from slug_items.

        Returns:
            prefix (str): the mount path (e.g. "/zenodo/view/<slug>/") to
                redirect the browser to.

        Raises:
            KeyError: slug is not a file of the configured record.
        """
        self.list_files()
        if slug not in self._files_by_slug:
            raise KeyError(slug)
        zfile = self._files_by_slug[slug]

        with self._lock:
            if slug in self.mounted_:
                self.mounted_.move_to_end(slug)
                return self.mounted_[slug]

            path = zenodo.fetch_file(zfile, max_bytes=self.max_bytes)
            ana, exp, mask_target = _extract_ana(_open_pickle(path))

            mount_key = f'/zenodo/view/{slug}'
            app = _create_app(
                ana, exp, mask_target=mask_target,
                routes_pathname_prefix='/',
                requests_pathname_prefix=f'{mount_key}/')
            self.application.mounts[mount_key] = app.server
            self.mounted_[slug] = f'{mount_key}/'

            while len(self.mounted_) > self.max_mounts:
                old_slug, old_prefix = self.mounted_.popitem(last=False)
                # Drop the mount; with no other reference the Dash app and the
                # analysis it closed over become collectable.
                self.application.mounts.pop(old_prefix.rstrip('/'), None)

            return self.mounted_[slug]


def _landing_html(registry, mounter: ZenodoMounter):
    """Render the landing page: curated demos plus the Zenodo browser link."""
    items_html = []
    for key, payload in registry.items():
        combo = payload.get('combo', {})
        label = _describe_combo(combo) if combo else key
        items_html.append(
            f'<li><a href="/{html.escape(key)}/">{html.escape(label)}</a>'
            f' <span class="key">({html.escape(key)})</span></li>'
        )
    items = ('\n'.join(items_html)
             or '<li><em>none baked into this build</em></li>')

    zenodo_html = ''
    if mounter.enabled:
        zenodo_html = (
            '<h2>Browse the Zenodo archive</h2>\n'
            '<p class="lede">Load any single precomputed experiment from the '
            'published Zenodo record into the interactive viewer.</p>\n'
            f'<ul><li><a href="/zenodo/">Browse record '
            f'{html.escape(str(mounter.record_id))} &rarr;</a></li></ul>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GLOW Viewer Demos</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px;
         margin: 3rem auto; padding: 0 1rem; line-height: 1.5; color: #222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  p.lede {{ color: #555; margin-top: 0; }}
  ul {{ padding-left: 1.25rem; }}
  li {{ margin: 0.4rem 0; }}
  a {{ color: #0066cc; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .key {{ color: #999; font-family: monospace; font-size: 0.85em; }}
  footer {{ color: #888; font-size: 0.85em; margin-top: 3rem; }}
</style>
</head>
<body>
<h1>GLOW Viewer</h1>
<p class="lede">Interactive demos of the GLOW hierarchical-segmentation
viewer. Each link below loads a pre-baked analysis with synthetic effects
imposed on the source images. Pick one to explore the scatter,
region overlay, and per-region regression panels.</p>

<h2>Curated demos</h2>
<ul>
{items}
</ul>
{zenodo_html}

<footer>
GLOW: General Linear models Optimized with Ward's method.<br>
Questions or feedback:
<a href="mailto:mhigger@ccs.neu.edu">mhigger@ccs.neu.edu</a>
</footer>
</body>
</html>"""


def _zenodo_index_html(mounter: ZenodoMounter):
    """Render the Zenodo file picker: one load link per record file."""
    try:
        items = mounter.slug_items()
    except Exception as e:
        return (f'<!doctype html><meta charset="utf-8">'
                f'<title>Zenodo</title>'
                f'<p>Could not read record '
                f'{html.escape(str(mounter.record_id))}: '
                f'{html.escape(str(e))}</p>'
                f'<p><a href="/">&larr; back</a></p>'), 502

    rows = []
    for slug, f in items:
        size_mb = f.size / (1024 ** 2)
        over = f.size > mounter.max_bytes
        link = (f'<span class="key" title="over the size cap">'
                f'{html.escape(f.key)} (too large)</span>' if over else
                f'<a href="/zenodo/load/{html.escape(slug)}">'
                f'{html.escape(f.key)}</a>')
        rows.append(
            f'<li>{link} <span class="key">'
            f'({size_mb:.1f} MB)</span></li>')
    body = '\n'.join(rows) or '<li><em>record has no files</em></li>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GLOW: Zenodo archive</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px;
         margin: 3rem auto; padding: 0 1rem; line-height: 1.5; color: #222; }}
  a {{ color: #0066cc; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
  .key {{ color: #999; font-family: monospace; font-size: 0.85em; }}
</style></head>
<body>
<h1>Zenodo record {html.escape(str(mounter.record_id))}</h1>
<p>Pick a precomputed experiment to load into the viewer. The first load of
a file downloads it (then it is cached); large files open more slowly.</p>
<ul>
{body}
</ul>
<p><a href="/">&larr; back to demos</a></p>
</body></html>"""


def build_server(pickle_dir=_PICKLE_DIR) -> Flask:
    """Build the Flask server: landing page, curated demos, Zenodo routes.

    Tolerates a missing or empty pickle dir (the Space still boots with the
    Zenodo browser and an empty curated list). The returned server carries
    its ZenodoMounter as server.zenodo_mounter; build_application wires that
    mounter to the dispatcher.

    Args:
        pickle_dir: directory of baked *.p.gz demos.

    Returns:
        server (Flask): the configured Flask application (not yet wrapped).
    """
    pickle_dir = pathlib.Path(pickle_dir)
    if pickle_dir.exists():
        print(f'loading registry from {pickle_dir}')
        registry = _load_registry(pickle_dir)
    else:
        print(f'pickle dir not found ({pickle_dir}); serving without demos')
        registry = {}

    mounter = ZenodoMounter(
        os.environ.get('GLOW_ZENODO_RECORD_ID'),
        max_bytes=int(os.environ.get('GLOW_ZENODO_MAX_MB',
                                     zenodo.DEFAULT_MAX_BYTES // (1024 ** 2)))
        * 1024 ** 2,
        max_mounts=int(os.environ.get('GLOW_ZENODO_MAX_MOUNTS',
                                      DEFAULT_MAX_MOUNTS)))

    server = Flask('glow_viewer_web_demo')
    server.zenodo_mounter = mounter

    @server.route('/')
    def index():
        """Serve the landing page."""
        return _landing_html(registry, mounter)

    @server.route('/healthz')
    def healthz():
        """Serve the liveness probe."""
        return 'ok', 200

    @server.route('/zenodo/')
    def zenodo_index():
        """Serve the Zenodo file picker, or 404 if disabled."""
        if not mounter.enabled:
            abort(404)
        return _zenodo_index_html(mounter)

    @server.route('/zenodo/load/<slug>')
    def zenodo_load(slug):
        """Mount the viewer for one slug and redirect to it."""
        if not mounter.enabled:
            abort(404)
        try:
            prefix = mounter.ensure_mounted(slug)
        except KeyError:
            abort(404)
        except ValueError as e:
            return (f'<!doctype html><meta charset="utf-8">'
                    f'<p>Could not load: {html.escape(str(e))}</p>'
                    f'<p><a href="/zenodo/">&larr; back</a></p>'), 413
        return redirect(prefix, code=302)

    print(f'mounting {len(registry)} curated Dash app(s)')
    for key, payload in registry.items():
        ana, exp, mask = _extract_ana(payload)
        prefix = f'/{key}/'
        _create_app(ana, exp, mask_target=mask,
                    url_base_pathname=prefix, server=server)
        print(f'  mounted {prefix}')

    return server


def build_application(pickle_dir=_PICKLE_DIR) -> DispatcherMiddleware:
    """Build the WSGI app: the Flask server wrapped in a dispatcher.

    The dispatcher lets per-file Zenodo viewers be mounted after start-up
    (Flask forbids adding routes to the server after its first request).
    Links the server's ZenodoMounter to this dispatcher so its mount table
    is the one the mounter adds to and evicts from.

    Args:
        pickle_dir: directory of baked *.p.gz demos.

    Returns:
        application (DispatcherMiddleware): the WSGI entry point.
    """
    server = build_server(pickle_dir)
    application = DispatcherMiddleware(server)
    server.zenodo_mounter.application = application
    return application


# WSGI entry point: gunicorn loads
# glow._extra.viewer.web.server:application. Built eagerly at import so
# workers don't race on first request.
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
