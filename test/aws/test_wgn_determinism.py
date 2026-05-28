"""WGN data sources rebuild bit-for-bit from `(seed, shape, b, num_img)`.

driver_aws relies on this so workers don't need a copy of the
DataSourceWGN experiment uploaded to S3 — they rebuild locally from
the trial's `ds`.  If this test ever fails, every WGN trial would
diverge between local and AWS runs and `driver_local` would re-run
them (or worse, silently disagree at the float-tolerance level).
"""

import numpy as np

from glow.benchmark.data import DataSource, DataSourceWGN


def _fresh_wgn(**kwargs):
    """Clear the class-level _exp_cache so each build is from scratch."""
    DataSource._exp_cache.clear()
    return DataSourceWGN(**kwargs).exp


def test_y_bytes_identical_across_builds():
    a = _fresh_wgn(seed=0, shape=(4, 4, 4), b=2, num_img=20)
    b = _fresh_wgn(seed=0, shape=(4, 4, 4), b=2, num_img=20)
    assert np.array_equal(a.y, b.y)
    assert a.y.dtype == b.y.dtype
    assert a.y.shape == b.y.shape


def test_x_bytes_identical_across_builds():
    a = _fresh_wgn(seed=0, shape=(4, 4, 4), b=2, num_img=20)
    b = _fresh_wgn(seed=0, shape=(4, 4, 4), b=2, num_img=20)
    assert np.array_equal(a.x, b.x)


def test_different_seed_differs():
    a = _fresh_wgn(seed=0, shape=(4, 4, 4), b=2, num_img=20)
    b = _fresh_wgn(seed=1, shape=(4, 4, 4), b=2, num_img=20)
    assert not np.array_equal(a.y, b.y)


def test_memoised_returns_same_object():
    """Two calls with equal source kwargs share the cached exp."""
    DataSource._exp_cache.clear()
    a = DataSourceWGN(seed=0, shape=(4, 4, 4), b=2, num_img=20).exp
    b = DataSourceWGN(seed=0, shape=(4, 4, 4), b=2, num_img=20).exp
    assert a is b
