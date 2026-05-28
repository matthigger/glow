"""Tests for Experiment._hash()."""

import numpy as np

from glow.experiment.exper import ExperimentImageOnly, Experiment
from glow.mask import get_mask_idx


# ---------------------------------------------------------------------------
# ExperimentImageOnly._hash
# ---------------------------------------------------------------------------

class TestExperimentImageOnlyHash:
    def test_wires_in_y_and_mask_idx(self):
        """_hash delegates to hash_array (tested exhaustively in
        test_util.py); here we only confirm both y and mask_idx are fed
        in, so changing either changes the hash."""
        y = np.random.randn(2, 5, 10)
        m1 = get_mask_idx(np.ones((2, 5)))
        ref = ExperimentImageOnly(y=y, mask_idx=m1)

        # changing y changes the hash
        e_other_y = ExperimentImageOnly(y=y + 1, mask_idx=m1)
        assert ref._hash() != e_other_y._hash()

        # changing mask_idx changes the hash
        m2 = get_mask_idx(np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]]))
        e_other_mask = ExperimentImageOnly(
            y=y[:, :, :m2.max() + 1], mask_idx=m2)
        assert ref._hash() != e_other_mask._hash()


# ---------------------------------------------------------------------------
# Experiment._hash (includes x and contrast)
# ---------------------------------------------------------------------------

class TestExperimentHash:
    def _make_exp(self, seed=0, shape=(3, 3)):
        return Experiment.from_gauss(a=2, seed=seed, shape=shape, num_img=5)

    def test_includes_x_and_contrast(self):
        """Experiment._hash feeds the same y/mask_idx as ExperimentImageOnly
        but additionally hashes x + contrast, so the two hashes differ on
        identical image data.  Also confirms determinism (same seed/params
        → same hash) and seed sensitivity, which are properties of the
        underlying hash_array (proven in test_util.py)."""
        exp = self._make_exp(seed=0)

        # deterministic for fixed seed/params
        assert exp._hash() == self._make_exp(seed=0)._hash()

        # a different seed yields different data, hence a different hash
        assert exp._hash() != self._make_exp(seed=1)._hash()

        # x + contrast participate: same y/mask_idx but no x/contrast differs
        eio = ExperimentImageOnly(y=exp.y, mask_idx=exp.mask_idx)
        assert exp._hash() != eio._hash()


