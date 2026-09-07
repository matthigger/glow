"""Check the Space would boot: import the server with only its own deps.

The deployed image installs requirements.txt and nothing else, and
deploy_hf.sh rsyncs glow/ without the benchmark subpackage. Neither
constraint is visible from a dev checkout, where every package is
present, so an import the image cannot satisfy passes locally and fails
on the Space. This runs the boot path with those two constraints imposed.

Usage:
    python -m glow._extra.viewer.web.check_space

Blocks glow._extra.benchmark plus the heavy packages a dev environment
has and the image does not, then boots the server, renders the landing
page, and mounts one baked bundle. Exits non-zero on the first import the
image could not satisfy.

The guarded run happens in a subprocess: this module lives inside the
glow package, so importing it has already pulled glow.analysis (and the
third-party modules underneath it) into sys.modules, and a guard armed
after that would have nothing left to refuse.

Run it before a deploy; it is not a unit test because what it asserts is
a property of the image, not of the code.
"""

import argparse
import pathlib
import re
import subprocess
import sys

# Heavy packages a dev checkout has and requirements.txt does not install.
# Anything requirements.txt names is dropped from this set at run time, so
# adding a dependency there is the one edit needed.
_MAYBE_ABSENT = {'torch', 'numba', 'matplotlib', 'seaborn', 'boto3',
                 'botocore', 'statsmodels', 'dipy', 'h5py'}

_REQUIREMENTS = pathlib.Path(__file__).parent / 'requirements.txt'
_PICKLE_DIR = pathlib.Path(__file__).parent / 'pickles'

# The guarded boot, run in a fresh interpreter. Arms the import hook
# first, so every glow and third-party import below it is checked.
_BOOTSTRAP = '''
import builtins, pathlib, sys

blocked = set(%(blocked)r)
_real = builtins.__import__

def _guard(name, *args, **kwargs):
    if name.startswith('glow._extra.benchmark'):
        raise ModuleNotFoundError(
            name + ' is excluded from the image (deploy_hf.sh)')
    if name.split('.')[0] in blocked:
        raise ModuleNotFoundError(name + ' is not in requirements.txt')
    return _real(name, *args, **kwargs)

builtins.__import__ = _guard

from werkzeug.test import Client
from glow._extra.viewer.web import server

pickle_dir = pathlib.Path(%(pickle_dir)r)
client = Client(server.build_application(pickle_dir=pickle_dir))
if client.get('/').status_code != 200:
    print('FAIL: landing page did not render')
    sys.exit(1)
print('landing page OK')

manifest = server.read_manifest(pickle_dir)
if not manifest:
    print('no bundles baked; boot path checked, render skipped')
    sys.exit(0)

key = manifest[0]['key']
client.get('/load/' + key)
layout = client.get('/view/' + key + '/_dash-layout')
if layout.status_code != 200:
    print('FAIL: ' + key + ' did not render')
    sys.exit(1)
print('mounted and rendered ' + key)
'''


def read_requirements(path=_REQUIREMENTS) -> set:
    """Return the distribution names requirements.txt installs.

    Args:
        path (pathlib.Path): the requirements file.

    Returns:
        set[str]: lower-cased names, dashes normalised to underscores.
    """
    names = set()
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        name = re.split(r'[<>=!\[]', line)[0].strip().lower()
        names.add(name.replace('-', '_'))
    return names


def check(pickle_dir=_PICKLE_DIR) -> int:
    """Boot the server under the image's constraints and render a bundle.

    Args:
        pickle_dir: directory of baked bundles; a bundle is mounted only
            if one is present.

    Returns:
        int: the subprocess return code, 0 when the boot path is
            satisfiable.
    """
    blocked = sorted(_MAYBE_ABSENT - read_requirements())
    print(f'blocking: {blocked or "nothing"} + benchmark')

    code = _BOOTSTRAP % {'blocked': blocked,
                         'pickle_dir': str(pickle_dir)}
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True)
    for line in (out.stdout + out.stderr).splitlines():
        if 'ModuleNotFoundError' in line or not line.startswith('  '):
            print(f'  {line}')
    return out.returncode


def main():
    """Parse args and run the check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dir', default=_PICKLE_DIR,
                        help='directory of baked bundles')
    args = parser.parse_args()
    rc = check(args.dir)
    print('SPACE ENV OK' if rc == 0 else 'SPACE ENV BROKEN')
    return rc


if __name__ == '__main__':
    sys.exit(main())
