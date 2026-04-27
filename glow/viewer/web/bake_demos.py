"""Pre-bake a curated set of glow.viewer demo analyses for the web demo.

Each entry in ``COMBOS`` calls one of the existing ``_demo_*`` builders in
``glow.viewer.__main__`` and pickles the resulting ``(ana, mask_target)`` pair
to ``PICKLE_DIR``.

Usage::

    python -m glow.viewer.web.bake_demos                    # bake all
    python -m glow.viewer.web.bake_demos --force            # rebuild even
                                                            # if pickle exists
    python -m glow.viewer.web.bake_demos --out path/to/dir  # custom output

The pickles are loaded by ``glow.viewer.web.server`` at startup and served
via the multi-user Dash app.

Each combo produces a single ``.p.gz`` file named by ``canonical_key()`` so
filenames are stable across rebuilds.
"""

import argparse
import gzip
import pathlib
import pickle
import sys
import time

# import the demo builders from the existing CLI module
from glow.viewer.__main__ import (
    _demo_wgn_2d,
    _demo_wgn_3d,
    _demo_mandrill,
    _demo_dti_2d,
    _demo_dti_3d,
    _EFFECT_MAP,
)


# ---------------------------------------------------------------------------
# Curated demo set
# ---------------------------------------------------------------------------
# Each entry is a dict consumed by ``_build_combo``.  Edit this list to grow
# or shrink the demo library.  Keep it small -- every entry adds RAM at
# server startup and bytes to the Docker image.

COMBOS = [
    # --- 2D White Gaussian Noise --------------------------------------
    # univariate, the canonical "what is GLOW doing?" demo
    {'image_set': 'wgn2d', 'b': 1, 'severity': 'medium'},
    # multivariate, motivates the b>1 story
    {'image_set': 'wgn2d', 'b': 3, 'severity': 'medium'},

    # --- 2D Mandrill --------------------------------------------------
    # the visual / fun demo; helps build intuition about Ward clustering
    {'image_set': 'mandrill', 'features': 'all', 'severity': 'medium'},

    # --- 2D Axial-slice DTI (HCP) -------------------------------------
    # real neuro data, fast enough for a 2D viewer
    {'image_set': 'dti2d', 'features': 'all', 'severity': 'medium'},

    # --- 3D DTI (HCP) -------------------------------------------------
    # the headline 3D experience
    {'image_set': 'dti3d', 'features': 'all', 'severity': 'medium'},
]

# all combos use these defaults unless overridden in the dict above
_DEFAULTS = {'seed': 0, 'roughness': None}


# ---------------------------------------------------------------------------
# Combo -> pickle path
# ---------------------------------------------------------------------------

def canonical_key(combo):
    """Stable filename-safe key for a combo dict."""
    parts = [combo['image_set']]
    if 'b' in combo:
        parts.append(f"b{combo['b']}")
    if 'features' in combo:
        parts.append(combo['features'])
    parts.append(combo['severity'])
    rough = combo.get('roughness', _DEFAULTS['roughness'])
    if rough is not None:
        parts.append(f'r{rough}')
    parts.append(f"s{combo.get('seed', _DEFAULTS['seed'])}")
    return '_'.join(parts)


# ---------------------------------------------------------------------------
# Build a single combo
# ---------------------------------------------------------------------------

def _build_combo(combo):
    """Resolve a combo dict to (ana, mask_target) by calling the right builder."""
    image_set = combo['image_set']
    seed = combo.get('seed', _DEFAULTS['seed'])
    roughness = combo.get('roughness', _DEFAULTS['roughness'])
    effect_llr = _EFFECT_MAP[combo['severity']]
    kw = dict(effect_llr=effect_llr, seed=seed, roughness=roughness)

    if image_set == 'wgn2d':
        return _demo_wgn_2d(combo['b'], **kw)
    if image_set == 'wgn3d':
        return _demo_wgn_3d(combo['b'], **kw)
    if image_set == 'mandrill':
        return _demo_mandrill(combo['features'], **kw)
    if image_set == 'dti2d':
        return _demo_dti_2d(combo['features'], **kw)
    if image_set == 'dti3d':
        return _demo_dti_3d(combo['features'], **kw)
    raise ValueError(f'unknown image_set: {image_set!r}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--out', type=pathlib.Path,
        default=pathlib.Path(__file__).parent / 'pickles',
        help='output directory for baked pickles')
    parser.add_argument(
        '--force', action='store_true',
        help='rebuild combos even if a pickle already exists')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f'baking {len(COMBOS)} combos into {args.out}')

    total_t0 = time.time()
    for i, combo in enumerate(COMBOS, 1):
        key = canonical_key(combo)
        out = args.out / f'{key}.p.gz'

        if out.exists() and not args.force:
            size_mb = out.stat().st_size / (1024 ** 2)
            print(f'[{i}/{len(COMBOS)}] {key}: skip (exists, {size_mb:.1f} MB)')
            continue

        print(f'[{i}/{len(COMBOS)}] {key}: building ...')
        t0 = time.time()
        ana, mask_target = _build_combo(combo)
        build_s = time.time() - t0

        with gzip.open(out, 'wb') as f:
            pickle.dump({'ana': ana, 'mask_target': mask_target,
                         'combo': combo}, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = out.stat().st_size / (1024 ** 2)
        print(f'    -> {out.name} ({size_mb:.1f} MB, build {build_s:.1f}s)')

    total_s = time.time() - total_t0
    print(f'done in {total_s:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
