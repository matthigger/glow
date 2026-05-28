"""TFCE consistency tests across 2D/3D and varying H, E parameters.

Verifies:
- 2D and 3D images produce valid TFCE output
- H (height) exponent controls sensitivity to peak intensity
- E (extent) exponent controls sensitivity to cluster size
- Analytical single-pixel/voxel case matches expected sum
- apply_tfce_x works end-to-end with 2D and 3D masks
"""

import numpy as np
import pytest

from glow.analysis.vba._tfce import apply_tfce_img, apply_tfce_x


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _single_element_expected(value, H, n_steps):
    """Exact TFCE for a single pixel/voxel with extent=1.

    TFCE = sum_{i=1}^{n_steps} 1^E * h_i^H
         = sum of h_i^H  (extent=1 so E term is always 1)
    where h_i = value * i / n_steps.
    """
    dh = value / n_steps
    hs = np.arange(1, n_steps + 1) * dh
    return np.sum(hs ** H)


def _make_two_clusters_2d(val_small, val_large, size_small=1, size_large=9):
    """2D image with a small bright cluster and a large dim cluster."""
    img = np.zeros((15, 15))
    # small cluster at top-left
    img[1:1 + size_small, 1:1 + size_small] = val_large
    # large cluster at bottom-right (3x3)
    s = int(np.sqrt(size_large))
    img[10:10 + s, 10:10 + s] = val_small
    return img


def _make_two_clusters_3d(val_small, val_large, size_small=1, size_large=27):
    """3D image with a small bright cluster and a large dim cluster."""
    img = np.zeros((15, 15, 15))
    # small cluster
    img[2:2 + size_small, 2:2 + size_small, 2:2 + size_small] = val_large
    # large cluster (3x3x3)
    s = int(round(size_large ** (1/3)))
    img[10:10 + s, 10:10 + s, 10:10 + s] = val_small
    return img


# ---------------------------------------------------------------------------
# 2D support
# ---------------------------------------------------------------------------

class TestTFCE2D:

    @pytest.mark.parametrize('img', [np.zeros((5, 5)), -np.ones((5, 5))])
    def test_nonpositive_gives_zeros(self, img):
        """img_max <= 0 produces all-zero TFCE."""
        assert np.allclose(apply_tfce_img(img), 0)

    def test_cluster_enhancement(self):
        """Larger cluster should get higher TFCE than single pixel."""
        x1 = np.zeros((9, 9))
        x1[4, 4] = 1.0
        x2 = np.zeros((9, 9))
        x2[3:6, 3:6] = 1.0
        assert apply_tfce_img(x2)[4, 4] > apply_tfce_img(x1)[4, 4]

    def test_connectivity_4_vs_8(self):
        """8-connected should merge diagonal neighbors, giving larger extent."""
        x = np.zeros((9, 9))
        x[3, 3] = 1.0
        x[4, 4] = 1.0  # diagonal neighbor
        t4 = apply_tfce_img(x, connectivity=4)
        t8 = apply_tfce_img(x, connectivity=8)
        # 8-connected merges them into one cluster (extent=2 > extent=1)
        assert t8[3, 3] > t4[3, 3]


# ---------------------------------------------------------------------------
# 3D support (extends existing TestTFCEPurePython)
# ---------------------------------------------------------------------------

class TestTFCE3D:

    def test_connectivity_6_vs_26(self):
        """26-connected merges face-diagonal neighbors into larger clusters."""
        x = np.zeros((9, 9, 9))
        x[3, 3, 3] = 1.0
        x[4, 4, 4] = 1.0  # space diagonal
        t6 = apply_tfce_img(x, connectivity=6)
        t26 = apply_tfce_img(x, connectivity=26)
        assert t26[3, 3, 3] > t6[3, 3, 3]


# ---------------------------------------------------------------------------
# H and E parameter sensitivity
# ---------------------------------------------------------------------------

class TestHeightExponent:
    """Higher H should favor tall peaks over broad low clusters."""

    def _imgs(self, ndim):
        if ndim == 2:
            return _make_two_clusters_2d(val_small=0.5, val_large=2.0)
        return _make_two_clusters_3d(val_small=0.5, val_large=2.0)

    def _peak_ratio(self, img, H, E=0.5):
        t = apply_tfce_img(img, H=H, E=E)
        peak_bright = t[1, 1] if img.ndim == 2 else t[2, 2, 2]
        peak_dim = t[11, 11] if img.ndim == 2 else t[11, 11, 11]
        return peak_bright / max(peak_dim, 1e-30)

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_higher_H_favors_peaks(self, ndim):
        """Increasing H should increase the ratio of bright/dim TFCE."""
        img = self._imgs(ndim)
        ratio_low = self._peak_ratio(img, H=1.0)
        ratio_high = self._peak_ratio(img, H=3.0)
        assert ratio_high > ratio_low


class TestExtentExponent:
    """Higher E should favor large clusters over isolated peaks."""

    def _imgs(self, ndim):
        if ndim == 2:
            return _make_two_clusters_2d(val_small=1.0, val_large=1.0,
                                         size_small=1, size_large=9)
        return _make_two_clusters_3d(val_small=1.0, val_large=1.0,
                                     size_small=1, size_large=27)

    def _large_ratio(self, img, E, H=2.0):
        t = apply_tfce_img(img, H=H, E=E)
        peak_single = t[1, 1] if img.ndim == 2 else t[2, 2, 2]
        peak_cluster = t[11, 11] if img.ndim == 2 else t[11, 11, 11]
        return peak_cluster / max(peak_single, 1e-30)

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_higher_E_favors_extent(self, ndim):
        """Increasing E should increase the ratio of large/small cluster TFCE."""
        img = self._imgs(ndim)
        ratio_low = self._large_ratio(img, E=0.0)
        ratio_high = self._large_ratio(img, E=1.5)
        assert ratio_high > ratio_low

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_E_zero_ignores_extent(self, ndim):
        """E=0 means extent is ignored: single element == cluster center
        when both have the same height."""
        center = (4,) * ndim
        x1 = np.zeros((9,) * ndim)
        x1[center] = 1.0
        x2 = np.zeros((9,) * ndim)
        x2[(slice(3, 6),) * ndim] = 1.0
        t1 = apply_tfce_img(x1, E=0.0)[center]
        t2 = apply_tfce_img(x2, E=0.0)[center]
        np.testing.assert_allclose(t1, t2, rtol=1e-10)


# ---------------------------------------------------------------------------
# analytical single-element match (2D + 3D)
# ---------------------------------------------------------------------------

class TestAnalytical:

    @pytest.mark.parametrize('ndim', [2, 3])
    @pytest.mark.parametrize('H', [1.0, 2.0, 3.0])
    def test_single_element_matches_sum(self, ndim, H):
        """A single element's TFCE matches the discrete sum, in 2D and 3D."""
        val = 2.5
        n_steps = 100
        center = (3,) * ndim
        x = np.zeros((7,) * ndim)
        x[center] = val
        r = apply_tfce_img(x, H=H, n_steps=n_steps)
        expected = _single_element_expected(val, H, n_steps)
        np.testing.assert_allclose(r[center], expected, rtol=1e-10)

    def test_uniform_cluster_all_equal(self):
        """All pixels in a uniform cluster get the same TFCE value."""
        x = np.zeros((9, 9))
        x[3:6, 3:6] = 1.0
        r = apply_tfce_img(x)
        cluster_vals = r[3:6, 3:6]
        assert np.allclose(cluster_vals, cluster_vals[0, 0])


# ---------------------------------------------------------------------------
# apply_tfce_x integration
# ---------------------------------------------------------------------------

class TestApplyTfceX:

    def test_2d_mask(self):
        """apply_tfce_x should work with a 2D mask_idx."""
        mask_idx = np.full((5, 5), -1)
        mask_idx[1:4, 1:4] = np.arange(9).reshape(3, 3)
        x = np.ones(9)
        r = apply_tfce_x(x, mask_idx)
        assert r.shape == (9,)
        assert (r > 0).all()

    def test_3d_mask(self):
        """apply_tfce_x should work with a 3D mask_idx."""
        mask_idx = np.full((5, 5, 5), -1)
        mask_idx[1:4, 1:4, 1:4] = np.arange(27).reshape(3, 3, 3)
        x = np.ones(27)
        r = apply_tfce_x(x, mask_idx)
        assert r.shape == (27,)
        assert (r > 0).all()

    def test_2d_preserves_relative_order(self):
        """Brighter voxels should get higher TFCE in 2D pipeline."""
        mask_idx = np.full((7, 7), -1)
        mask_idx[1:6, 1:6] = np.arange(25).reshape(5, 5)
        x = np.zeros(25)
        x[12] = 2.0  # center pixel, bright
        x[0] = 0.5   # corner pixel, dim
        r = apply_tfce_x(x, mask_idx)
        assert r[12] > r[0]
