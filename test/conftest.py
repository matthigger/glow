"""Root test configuration — runs before test collection."""
import os
from pathlib import Path

import pytest

# Add FSL to PATH if installed but not already available
_FSL_DIR = Path('/usr/local/fsl')
if _FSL_DIR.is_dir():
    os.environ.setdefault('FSLDIR', str(_FSL_DIR))
    os.environ.setdefault('FSLOUTPUTTYPE', 'NIFTI_GZ')
    fsl_bin = str(_FSL_DIR / 'bin')
    if fsl_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = fsl_bin + os.pathsep + os.environ.get('PATH', '')


@pytest.fixture(autouse=True)
def _cache_to_tmp(monkeypatch, tmp_path):
    """Redirect the benchmark's shared joblib cache to a tmp dir, every test.

    glow._extra.benchmark.data.MEMORY points at the user's real cache and the
    tests call the memoised builders and leaves directly, so without this the
    suite reads and writes tens of GB of production artifacts. Worse, joblib
    stores each memoised function's source and clears that function's whole
    cache directory when it changes -- and what it stores is the recorder
    wrapper every one of them is nested in, so a single edit there wipes the real
    cache the next time the suite runs.

    Rebinding MEMORY would not do it: the decorators captured the store at
    import time, so the location moves on the shared backend the already-built
    MemorizedFuncs hold. Imported inside the fixture to keep collection of the
    unrelated suites free of the benchmark import chain.
    """
    from glow._extra.benchmark import data

    monkeypatch.setattr(data.MEMORY.store_backend, 'location', str(tmp_path))


def pytest_addoption(parser):
    """Register the --runslow CLI flag."""
    parser.addoption(
        '--runslow', action='store_true', default=False,
        help='Run heavyweight tests (calibration, long-running simulations)',
    )


def pytest_configure(config):
    """Register custom markers so --strict-markers accepts them."""
    config.addinivalue_line(
        'markers',
        'slow: heavyweight test, skipped by default (use --runslow)',
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --runslow is passed."""
    run_slow = config.getoption('--runslow')

    skip_slow = pytest.mark.skip(reason='needs --runslow to run')

    for item in items:
        if 'slow' in item.keywords and not run_slow:
            item.add_marker(skip_slow)
