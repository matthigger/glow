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


def pytest_addoption(parser):
    """Register --runslow and --runaws CLI flags."""
    parser.addoption(
        '--runslow', action='store_true', default=False,
        help='Run heavyweight tests (calibration, long-running simulations)',
    )
    parser.addoption(
        '--runaws', action='store_true', default=False,
        help='Run tests that hit real AWS (S3, Batch); requires credentials',
    )


def pytest_configure(config):
    """Register custom markers so --strict-markers accepts them."""
    config.addinivalue_line(
        'markers',
        'slow: heavyweight test, skipped by default (use --runslow)',
    )
    config.addinivalue_line(
        'markers',
        'runaws: hits real AWS, skipped by default (use --runaws)',
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow / runaws tests unless their flags are passed."""
    run_slow = config.getoption('--runslow')
    run_aws = config.getoption('--runaws')

    skip_slow = pytest.mark.skip(reason='needs --runslow to run')
    skip_aws = pytest.mark.skip(reason='needs --runaws to run')

    for item in items:
        if 'slow' in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if 'runaws' in item.keywords and not run_aws:
            item.add_marker(skip_aws)
