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
