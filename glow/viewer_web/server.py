"""Multi-demo web server for glow.viewer.

Loads every baked pickle in ``pickles/`` into its own Dash app, all
sharing a single Flask server.  Each Dash app is mounted under
``/{key}/`` via ``url_base_pathname``; the root ``/`` serves a small
landing page listing the demos.

The Dash apps are reused as-is via ``glow.viewer.app._create_app``; no
session-key plumbing or hot-swap is required.

Run locally::

    python -m glow.viewer_web.server                # dev server, port 7860
    PORT=8080 python -m glow.viewer_web.server      # custom port

Run under gunicorn (Docker / HF Spaces)::

    gunicorn glow.viewer_web.server:application --bind 0.0.0.0:7860
"""

import argparse
import gzip
import html
import os
import pathlib
import pickle
import sys

from flask import Flask

from glow.viewer.app import _create_app


_PICKLE_DIR = pathlib.Path(__file__).parent / 'pickles'

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
    """Human-readable label for a combo dict."""
    parts = [_IMAGE_SET_LABELS.get(combo['image_set'], combo['image_set'])]
    if 'b' in combo:
        parts.append(f"b={combo['b']}")
    if 'features' in combo:
        parts.append(_FEATURE_LABELS.get(combo['features'], combo['features']))
    parts.append(f"{combo['severity']} effect")
    return ', '.join(parts)


def _load_registry(pickle_dir):
    """Load every ``*.p.gz`` in ``pickle_dir``.  Keyed by file stem."""
    registry = {}
    for path in sorted(pickle_dir.glob('*.p.gz')):
        key = path.name.removesuffix('.p.gz')
        with gzip.open(path, 'rb') as f:
            registry[key] = pickle.load(f)
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f'  loaded {key} ({size_mb:.1f} MB)')
    return registry


def _landing_html(registry):
    """Render the landing page that links to each demo."""
    items_html = []
    for key, payload in registry.items():
        combo = payload.get('combo', {})
        label = _describe_combo(combo) if combo else key
        items_html.append(
            f'<li><a href="/{html.escape(key)}/">{html.escape(label)}</a>'
            f' <span class="key">({html.escape(key)})</span></li>'
        )
    items = '\n'.join(items_html)
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

<h2>Available demos</h2>
<ul>
{items}
</ul>

<footer>
GLOW: General Linear models Optimized with Ward's method.
</footer>
</body>
</html>"""


def build_application(pickle_dir=_PICKLE_DIR):
    """Build the WSGI application: one shared Flask server, one Dash app
    per pickle mounted at ``/{key}/``, landing page at ``/``.
    """
    pickle_dir = pathlib.Path(pickle_dir)
    if not pickle_dir.exists():
        raise SystemExit(
            f'pickle dir not found: {pickle_dir}\n'
            f'run: python -m glow.viewer_web.bake_demos')

    print(f'loading registry from {pickle_dir}')
    registry = _load_registry(pickle_dir)
    if not registry:
        raise SystemExit(f'no pickles found in {pickle_dir}')

    server = Flask('glow_viewer_web')
    page = _landing_html(registry)

    @server.route('/')
    def index():
        return page

    @server.route('/healthz')
    def healthz():
        return 'ok', 200

    print(f'mounting {len(registry)} Dash apps')
    for key, payload in registry.items():
        ana = payload['ana']
        mask = payload.get('mask_target')
        prefix = f'/{key}/'
        _create_app(ana, mask_target=mask,
                    url_base_pathname=prefix, server=server)
        print(f'  mounted {prefix}')

    return server


# WSGI entry point: gunicorn loads ``glow.viewer_web.server:application``.
# Built eagerly at import so workers don't race on first request.
application = build_application()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('PORT', 7860)),
                        help='port to bind (default 7860 / $PORT)')
    parser.add_argument('--host', default='127.0.0.1',
                        help="bind host (use '0.0.0.0' for Docker)")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    from werkzeug.serving import run_simple
    print(f'\n  glow:viewer_web running at http://{args.host}:{args.port}')
    print('  press Ctrl+C to stop\n')
    run_simple(args.host, args.port, application,
               use_reloader=args.debug, use_debugger=args.debug)


if __name__ == '__main__':
    sys.exit(main())
