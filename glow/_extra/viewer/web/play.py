"""Load a baked demo pickle and launch the single-analysis viewer.

Local validator: confirms a pickle written by bake_demos.py round-trips
through the full Dash app, as the interactive python -m glow._extra.viewer
--demo flow does. Spot-check each baked combo before deploying.

Usage:
    python -m glow._extra.viewer.web.play wgn2d_b1_medium_s0
    python -m glow._extra.viewer.web.play --list
    python -m glow._extra.viewer.web.play path/to/some.p.gz
"""

import argparse
import gzip
import pathlib
import pickle
import sys

from glow._extra.viewer import launch

_DEFAULT_DIR = pathlib.Path(__file__).parent / 'pickles'


def _resolve(arg: str, pickle_dir: pathlib.Path) -> pathlib.Path:
    """Resolve a combo key or path to a baked pickle path.

    Args:
        arg (str): a combo key (e.g. 'wgn2d_b1_medium_s0') or a file path.
        pickle_dir (pathlib.Path): directory keys are looked up in.

    Returns:
        path (pathlib.Path): the resolved pickle file.

    Raises:
        SystemExit: neither the path nor the keyed pickle exists.
    """
    p = pathlib.Path(arg)
    if p.exists():
        return p
    candidate = pickle_dir / f'{arg}.p.gz'
    if candidate.exists():
        return candidate
    raise SystemExit(f'no pickle at {p} or {candidate}')


def _load(path: pathlib.Path):
    """Unpickle a gzipped baked-demo file."""
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def main():
    """Parse args and launch the viewer (or list combos) for one pickle."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('key', nargs='?',
                        help='combo key or path to a baked pickle')
    parser.add_argument('--dir', type=pathlib.Path, default=_DEFAULT_DIR,
                        help='directory holding baked pickles')
    parser.add_argument('--list', action='store_true',
                        help='list available baked combos and exit')
    parser.add_argument('--port', type=int, default=8050)
    args = parser.parse_args()

    if args.list:
        if not args.dir.exists():
            print(f'(no pickle dir at {args.dir})')
            return 0
        keys = sorted(
            p.stem.removesuffix('.p') for p in args.dir.glob('*.p.gz'))
        if not keys:
            print(f'(no pickles in {args.dir})')
        else:
            for k in keys:
                print(f'  {k}')
        return 0

    if not args.key:
        parser.error('provide a combo key or path (or use --list)')

    path = _resolve(args.key, args.dir)
    print(f'loading {path} ...')
    payload = _load(path)
    ana = payload['ana']
    exp = payload['exp']
    mask_target = payload.get('mask_target')
    combo = payload.get('combo', {})
    if combo:
        print(f'combo: {combo}')

    launch(ana, exp, mask_target=mask_target, port=args.port)


if __name__ == '__main__':
    sys.exit(main())
