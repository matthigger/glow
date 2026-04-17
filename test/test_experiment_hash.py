"""Tests for Experiment._hash() and Config._get_exp_orig_signature()."""

import numpy as np
import pytest

from glow.experiment.exper import ExperimentImageOnly, Experiment
from glow.mask import get_mask_idx


# ---------------------------------------------------------------------------
# ExperimentImageOnly._hash
# ---------------------------------------------------------------------------

class TestExperimentImageOnlyHash:
    def test_deterministic(self):
        """Same data produces the same hash."""
        y = np.random.randn(2, 5, 10)
        mask_idx = get_mask_idx(np.ones((2, 5)))
        e1 = ExperimentImageOnly(y=y, mask_idx=mask_idx)
        e2 = ExperimentImageOnly(y=y.copy(), mask_idx=mask_idx.copy())
        assert e1._hash() == e2._hash()

    def test_different_y(self):
        """Different y produces a different hash."""
        mask_idx = get_mask_idx(np.ones((2, 5)))
        e1 = ExperimentImageOnly(y=np.zeros((2, 5, 10)), mask_idx=mask_idx)
        e2 = ExperimentImageOnly(y=np.ones((2, 5, 10)), mask_idx=mask_idx)
        assert e1._hash() != e2._hash()

    def test_different_mask(self):
        """Different mask_idx produces a different hash."""
        y = np.random.randn(2, 5, 10)
        m1 = get_mask_idx(np.ones((2, 5)))
        m2 = get_mask_idx(np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]]))
        e1 = ExperimentImageOnly(y=y, mask_idx=m1)
        e2 = ExperimentImageOnly(y=y[:, :, :m2.max() + 1], mask_idx=m2)
        assert e1._hash() != e2._hash()

    def test_hex_format(self):
        """Hash is a 16-char hex string."""
        e = ExperimentImageOnly(y=np.zeros((1, 2, 3)),
                                mask_idx=get_mask_idx(np.ones(3)))
        h = e._hash()
        assert len(h) == 16
        assert all(c in '0123456789abcdef' for c in h)


# ---------------------------------------------------------------------------
# Experiment._hash (includes x and contrast)
# ---------------------------------------------------------------------------

class TestExperimentHash:
    def _make_exp(self, seed=0, shape=(3, 3)):
        return Experiment.from_gauss(a=2, seed=seed, shape=shape, num_img=5)

    def test_deterministic(self):
        """Same seed/params gives same hash."""
        assert self._make_exp(seed=0)._hash() == self._make_exp(seed=0)._hash()

    def test_different_seed(self):
        """Different seed gives different hash."""
        assert self._make_exp(seed=0)._hash() != self._make_exp(seed=1)._hash()

    def test_different_shape(self):
        """Different spatial shape gives different hash."""
        assert (self._make_exp(shape=(3, 3))._hash() !=
                self._make_exp(shape=(4, 4))._hash())

    def test_includes_x_and_contrast(self):
        """Experiment hash differs from ExperimentImageOnly hash on same y."""
        exp = self._make_exp()
        eio = ExperimentImageOnly(y=exp.y, mask_idx=exp.mask_idx)
        # Experiment hashes x + contrast too, so the hashes must differ
        assert exp._hash() != eio._hash()

    def test_hex_format(self):
        h = self._make_exp()._hash()
        assert len(h) == 16
        assert all(c in '0123456789abcdef' for c in h)


# ---------------------------------------------------------------------------
# Config.prep_exp_orig + exp_orig._hash (end-to-end)
# ---------------------------------------------------------------------------

class TestConfigExpOrigHash:
    def test_wgn_deterministic(self):
        """Same WGN config gives the same hash."""
        from glow.benchmark.config import Config
        c1 = Config(label='t1', source='wgn', wgn_shape=(3, 3), exp_seed=0)
        c2 = Config(label='t2', source='wgn', wgn_shape=(3, 3), exp_seed=0)
        c1.prep_exp_orig()
        c2.prep_exp_orig()
        assert c1.exp_orig._hash() == c2.exp_orig._hash()

    def test_wgn_different_seed(self):
        """Different exp_seed gives a different hash."""
        from glow.benchmark.config import Config
        c1 = Config(label='t1', source='wgn', wgn_shape=(3, 3), exp_seed=0)
        c2 = Config(label='t2', source='wgn', wgn_shape=(3, 3), exp_seed=1)
        c1.prep_exp_orig()
        c2.prep_exp_orig()
        assert c1.exp_orig._hash() != c2.exp_orig._hash()

    def test_wgn_different_shape(self):
        """Different wgn_shape gives a different hash."""
        from glow.benchmark.config import Config
        c1 = Config(label='t1', source='wgn', wgn_shape=(3, 3), exp_seed=0)
        c2 = Config(label='t2', source='wgn', wgn_shape=(4, 4), exp_seed=0)
        c1.prep_exp_orig()
        c2.prep_exp_orig()
        assert c1.exp_orig._hash() != c2.exp_orig._hash()


# ---------------------------------------------------------------------------
# Config.prep_exp_orig per-trial resampling (seed kwarg)
# ---------------------------------------------------------------------------

class TestPrepExpOrigPerTrial:
    """Per-trial seed must redraw WGN noise and HCP design X.

    Regression guard for commit 1f51e28: before that fix, noise (WGN) and
    X (HCP) were drawn once at exp_seed and reused across all null trials,
    which collapsed null_wgn to 4 unique p-values over 500 seeds.
    """

    def test_wgn_seed_varies_noise(self):
        """Different trial seed → different WGN noise realization."""
        from glow.benchmark.config import Config
        c = Config(label='t', source='wgn', wgn_shape=(4, 4),
                   wgn_num_img=10, exp_seed=0)
        c.prep_exp_orig(seed=1)
        y1 = c.exp_orig.y.copy()
        c.prep_exp_orig(seed=2)
        y2 = c.exp_orig.y.copy()
        assert not np.array_equal(y1, y2), \
            'WGN y unchanged across trial seeds — noise is being cached'

    def test_wgn_seed_deterministic(self):
        """Same trial seed → same WGN noise realization."""
        from glow.benchmark.config import Config
        c = Config(label='t', source='wgn', wgn_shape=(4, 4),
                   wgn_num_img=10, exp_seed=0)
        c.prep_exp_orig(seed=7)
        y_a = c.exp_orig.y.copy()
        c.prep_exp_orig(seed=7)
        y_b = c.exp_orig.y.copy()
        np.testing.assert_array_equal(y_a, y_b)

    def test_hcp_seed_varies_x(self):
        """Different trial seed → different HCP design matrix.

        Uses a synthetic ExperimentImageOnly injected as ``_exp_img_only``
        to avoid requiring real HCP files.
        """
        from glow.benchmark.config import Config
        c = Config(label='t', source='hcp', exp_seed=0)
        c._exp_img_only = ExperimentImageOnly.from_gauss(
            b=1, num_img=20, shape=(4, 4), seed=0)
        c.prep_exp_orig(seed=1)
        x1 = c.exp_orig.x.copy()
        c.prep_exp_orig(seed=2)
        x2 = c.exp_orig.x.copy()
        assert not np.array_equal(x1, x2), \
            'HCP x unchanged across trial seeds — X is being cached'

    def test_hcp_seed_preserves_y(self):
        """HCP y (image data) must NOT change across trial seeds.

        Only X is redrawn per trial; the underlying imaging data is the
        expensive part and is reused from ``_exp_img_only``.
        """
        from glow.benchmark.config import Config
        c = Config(label='t', source='hcp', exp_seed=0)
        c._exp_img_only = ExperimentImageOnly.from_gauss(
            b=1, num_img=20, shape=(4, 4), seed=0)
        c.prep_exp_orig(seed=1)
        y1 = c.exp_orig.y.copy()
        c.prep_exp_orig(seed=2)
        y2 = c.exp_orig.y.copy()
        np.testing.assert_array_equal(y1, y2)

    def test_hcp_exp_img_only_cached(self):
        """Re-preparing HCP must not rebuild ``_exp_img_only``."""
        from glow.benchmark.config import Config
        c = Config(label='t', source='hcp', exp_seed=0)
        c._exp_img_only = ExperimentImageOnly.from_gauss(
            b=1, num_img=20, shape=(4, 4), seed=0)
        img_only_id = id(c._exp_img_only)
        c.prep_exp_orig(seed=1)
        c.prep_exp_orig(seed=2)
        assert id(c._exp_img_only) == img_only_id, \
            '_exp_img_only was rebuilt — trial loop will reload HCP data'
