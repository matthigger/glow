"""Backend-dispatch tests for TFCE (python vs fslmaths).

Verifies:
- resolve_backend honours the request and falls back when fslmaths
  cannot serve it (2d images, a non-default step count, no install)
- get_chunk_slices tiles a permutation stack exactly once
- the batched entry points agree with the per-image reference
- fslmaths on a 4d stack matches one call per volume, bit for bit
- volumes with no positive value come back zero instead of aborting

The fsl tests skip when fslmaths is not installed.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from glow.analysis.vba import AnalysisVBA
from glow.analysis.vba import _tfce
from glow.analysis.vba._tfce import (apply_tfce_at, apply_tfce_img,
                                     apply_tfce_stack_fsl, apply_tfce_stat,
                                     apply_tfce_x, get_chunk_slices,
                                     get_mask_bb, get_thresholds, has_fsl,
                                     resolve_backend)

needs_fsl = pytest.mark.skipif(not has_fsl(), reason='fslmaths not installed')

N_STEPS = 100

# Fraction of a peak voxel's TFCE carried by the top height step. This is
# the whole of the backend disagreement (see TestTopStepEndpoint): 2.96%
# at H=2, and the most the production 100-step grid can be off by.
TOP_STEP_FRAC = (N_STEPS ** 2
                 / sum(k ** 2 for k in range(1, N_STEPS + 1)))

# fslmaths steps one of exactly two grids -- the one apply_tfce_img
# builds, or that grid without its top threshold, which it drops on about
# half of all images. Against whichever it used, agreement is near
# machine precision: the matching grid measures below 2e-6 across image
# sizes, so this bound keeps ~6x headroom while the non-matching grid
# sits 400x above it. Tight enough that a real regression cannot hide.
FSL_GRID_RTOL = 1e-5


def get_grid_residual(stat, mask_idx):
    """Residual against fslmaths for the better of the two height grids.

    fslmaths picks its grid per image, so the choice is made per row
    rather than once for the stack.

    Args:
        stat (np.array): (n_img, num_vox) statistical values
        mask_idx (np.array): 3d index array, -1 for inactive voxels

    Returns:
        residual (np.array): (n_img,) max abs difference from fslmaths
            over voxels, relative to that image's fsl maximum, taking
            the smaller of the full and top-dropped grids
    """
    fsl = apply_tfce_stat(stat, mask_idx, backend='fsl')
    mask_bb = get_mask_bb(mask_idx)
    img = np.zeros((len(stat), *mask_bb.shape))
    img[:, mask_bb] = stat

    residual = np.zeros(len(stat))
    for idx, one in enumerate(img):
        scale = np.abs(fsl[idx]).max()
        if one.max() <= 0 or scale == 0:
            # TFCE is identically zero, so both grids agree exactly
            residual[idx] = np.abs(fsl[idx]).max()
            continue
        thr = get_thresholds(one.max(), N_STEPS)
        residual[idx] = min(
            np.abs(apply_tfce_at(one, thr)[mask_bb] - fsl[idx]).max(),
            np.abs(apply_tfce_at(one, thr[:-1])[mask_bb] - fsl[idx]).max(),
        ) / scale
    return residual


@pytest.fixture
def stat_mask():
    """Build a smooth (n_img, num_vox) stat matrix and its 3d mask index.

    Returns:
        stat (np.array): (n_img, num_vox) smoothed noise, signed
        mask_idx (np.array): (X, Y, Z) int, -1 outside the sphere
    """
    n_side, n_img = 16, 6
    grid = np.mgrid[:n_side, :n_side, :n_side] - (n_side - 1) / 2
    mask = np.sqrt((grid ** 2).sum(axis=0)) <= n_side / 2 - 1
    num_vox = int(mask.sum())

    mask_idx = np.full(mask.shape, -1)
    mask_idx[mask] = np.arange(num_vox)

    rng = np.random.default_rng(0)
    stat = np.zeros((n_img, num_vox))
    for idx in range(n_img):
        img = np.zeros(mask.shape)
        img[mask] = rng.standard_normal(num_vox)
        stat[idx] = gaussian_filter(img, sigma=1.5)[mask]
    return stat, mask_idx


# ---------------------------------------------------------------------------
# resolve_backend
# ---------------------------------------------------------------------------

class TestResolveBackend:

    def test_python_always_available(self):
        """A python request is served whether or not fslmaths exists."""
        assert resolve_backend('python') == 'python'
        assert resolve_backend('python', ndim=2, n_steps=7) == 'python'

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match='auto, fsl or python'):
            resolve_backend('fslmaths')

    def test_auto_falls_back_off_contract(self):
        """2d images and non-default step counts are python-only."""
        assert resolve_backend('auto', ndim=2) == 'python'
        assert resolve_backend('auto', n_steps=50) == 'python'

    def test_demanding_fsl_off_contract_raises(self):
        """An explicit fsl request must fail loudly, not fall back."""
        with pytest.raises(RuntimeError, match='3d or 4d'):
            resolve_backend('fsl', ndim=2)
        with pytest.raises(RuntimeError, match='height steps'):
            resolve_backend('fsl', n_steps=50)

    def test_auto_without_fsl_is_python(self, monkeypatch):
        monkeypatch.setattr(_tfce, 'has_fsl', lambda: False)
        assert resolve_backend('auto') == 'python'
        with pytest.raises(RuntimeError, match='not found'):
            resolve_backend('fsl')

    def test_none_takes_module_default(self, monkeypatch):
        monkeypatch.setattr(_tfce, 'BACKEND', 'python')
        assert resolve_backend(None) == 'python'

    @needs_fsl
    def test_auto_prefers_fsl_when_installed(self):
        assert resolve_backend('auto') == 'fsl'


# ---------------------------------------------------------------------------
# get_chunk_slices
# ---------------------------------------------------------------------------

class TestGetChunkSlices:

    @pytest.mark.parametrize('n_img,n_chunk', [(12, 4), (3, 8), (1, 4),
                                               (250, 32), (7, 1)])
    def test_tiles_every_row_once(self, n_img, n_chunk):
        sl_list = get_chunk_slices(n_img, n_vox_img=1000, n_chunk=n_chunk)
        covered = np.concatenate([np.arange(n_img)[sl] for sl in sl_list])
        assert np.array_equal(covered, np.arange(n_img))

    def test_never_more_chunks_than_images(self):
        assert len(get_chunk_slices(3, n_vox_img=10, n_chunk=99)) == 3

    def test_big_images_split_beyond_the_worker_count(self):
        """A stack over MAX_CHUNK_BYTES is split even on a single worker."""
        n_vox_img = _tfce.MAX_CHUNK_BYTES // 8
        sl_list = get_chunk_slices(8, n_vox_img=n_vox_img, n_chunk=1)
        assert len(sl_list) == 8


# ---------------------------------------------------------------------------
# batched entry points against the per-image reference
# ---------------------------------------------------------------------------

class TestBatchedPython:

    def test_stat_matches_per_image(self, stat_mask):
        """apply_tfce_stat must equal apply_tfce_img on each embedded row."""
        stat, mask_idx = stat_mask
        mask_bb = get_mask_bb(mask_idx)

        out = apply_tfce_stat(stat, mask_idx, backend='python')

        for idx, row in enumerate(stat):
            img = np.zeros(mask_bb.shape)
            img[mask_bb] = row
            assert np.array_equal(out[idx], apply_tfce_img(img)[mask_bb])

    def test_x_matches_a_one_row_stat(self, stat_mask):
        stat, mask_idx = stat_mask
        out = apply_tfce_stat(stat[:1], mask_idx, backend='python')
        assert np.array_equal(apply_tfce_x(stat[0], mask_idx,
                                           backend='python'), out[0])

    def test_analysis_apply_tfce_matches(self, stat_mask):
        """AnalysisVBA.apply_tfce must agree with the module entry point."""
        stat, mask_idx = stat_mask
        out = AnalysisVBA.apply_tfce(stat, mask_idx, backend='python')
        assert np.allclose(out, apply_tfce_stat(stat, mask_idx,
                                                backend='python'))


# ---------------------------------------------------------------------------
# fsl backend
# ---------------------------------------------------------------------------

@needs_fsl
class TestFslBackend:

    def test_4d_call_matches_one_call_per_volume(self, stat_mask):
        """Batching is exact: fslmaths treats 4d volumes independently."""
        stat, mask_idx = stat_mask
        mask_bb = get_mask_bb(mask_idx)
        img = np.zeros((len(stat), *mask_bb.shape))
        img[:, mask_bb] = stat

        batched = apply_tfce_stack_fsl(img)
        for idx, one in enumerate(img):
            solo = apply_tfce_stack_fsl(one[None])[0]
            assert np.array_equal(batched[idx], solo)

    def test_non_positive_volumes_return_zero(self, stat_mask):
        """A flat or all-negative volume is zero, and does not abort."""
        stat, mask_idx = stat_mask
        stat = stat.copy()
        stat[1] = -np.abs(stat[1])
        stat[3] = 0.0

        out = apply_tfce_stat(stat, mask_idx, backend='fsl')

        assert not out[[1, 3]].any()
        assert out[0].any()
        assert np.allclose(out[[1, 3]],
                           apply_tfce_stat(stat, mask_idx,
                                           backend='python')[[1, 3]])

    def test_all_volumes_non_positive(self, stat_mask):
        """Every volume non-positive means no fslmaths call at all."""
        stat, mask_idx = stat_mask
        out = apply_tfce_stat(-np.abs(stat), mask_idx, backend='fsl')
        assert not out.any()

    def test_matches_one_of_the_two_grids(self, stat_mask):
        """Against the grid fslmaths used, agreement is near machine eps.

        This is the real regression guard: it holds the backend to 1e-5
        rather than to the 1-2% the fixed 100-step grid alone allows.
        """
        stat, mask_idx = stat_mask
        assert get_grid_residual(stat, mask_idx).max() < FSL_GRID_RTOL

    def test_both_grids_are_needed(self, stat_mask):
        """Guard the test above against passing for the wrong reason.

        If fslmaths always used one grid, taking the better of two would
        be vacuous. Over enough images it uses each, so neither grid
        alone clears the tight bound.
        """
        stat, mask_idx = stat_mask
        fsl = apply_tfce_stat(stat, mask_idx, backend='fsl')
        mask_bb = get_mask_bb(mask_idx)
        img = np.zeros((len(stat), *mask_bb.shape))
        img[:, mask_bb] = stat

        wins = set()
        for idx, one in enumerate(img):
            thr = get_thresholds(one.max(), N_STEPS)
            full = np.abs(apply_tfce_at(one, thr)[mask_bb] - fsl[idx]).max()
            drop = np.abs(
                apply_tfce_at(one, thr[:-1])[mask_bb] - fsl[idx]).max()
            wins.add('full' if full < drop else 'dropped')
        assert wins == {'full', 'dropped'}, wins

    def test_agrees_with_python_within_the_top_step(self, stat_mask):
        """The production grid is off by at most that one step."""
        stat, mask_idx = stat_mask
        out_fsl = apply_tfce_stat(stat, mask_idx, backend='fsl')
        out_py = apply_tfce_stat(stat, mask_idx, backend='python')
        assert out_fsl.shape == out_py.shape
        atol = 1.1 * TOP_STEP_FRAC * np.abs(out_py).max()
        assert np.abs(out_fsl - out_py).max() < atol

    def test_agrees_with_python_on_the_max_stat_null(self, stat_mask):
        """The FWER null is what a backend swap must not move much."""
        stat, mask_idx = stat_mask
        max_fsl = apply_tfce_stat(stat, mask_idx, backend='fsl').max(axis=1)
        max_py = apply_tfce_stat(stat, mask_idx, backend='python').max(axis=1)
        assert np.allclose(max_fsl, max_py, rtol=1.1 * TOP_STEP_FRAC)

    @pytest.mark.parametrize('n_jobs', [1, 2, 4])
    def test_chunking_is_invariant_to_n_jobs(self, stat_mask, n_jobs):
        """Chunk boundaries must not change a single value."""
        stat, mask_idx = stat_mask
        out = AnalysisVBA.apply_tfce(stat, mask_idx, backend='fsl',
                                     n_jobs=n_jobs)
        ref = apply_tfce_stat(stat, mask_idx, backend='fsl')
        assert np.array_equal(out, ref)

    def test_analysis_backends_agree(self, stat_mask):
        stat, mask_idx = stat_mask
        out_fsl = AnalysisVBA.apply_tfce(stat, mask_idx, backend='fsl')
        out_py = AnalysisVBA.apply_tfce(stat, mask_idx, backend='python')
        atol = 1.1 * TOP_STEP_FRAC * np.abs(out_py).max()
        assert np.abs(out_fsl - out_py).max() < atol


@needs_fsl
class TestTopStepEndpoint:
    """Pin the one systematic difference between the backends.

    fslmaths accumulates its height in float32 and admits voxels
    strictly above it, so its last step lands either just below the
    image maximum (the peak voxel counts) or just above it (the peak
    drops out), depending on the low bits of the maximum.
    apply_tfce_img steps an exact linspace ending on the maximum and
    admits h and above, so it always counts that step. On a lone voxel
    the extent is 1 at every height, so the whole gap is that one term.
    """

    H = 2.0

    @staticmethod
    def _single_voxel_ratio(value, H):
        """Return python / fsl TFCE at a lone voxel of the given value."""
        mask_idx = np.full((5, 5, 5), -1)
        mask_idx[2, 2, 2] = 0
        x = np.array([value])
        out_py = apply_tfce_x(x, mask_idx, backend='python')[0]
        out_fsl = apply_tfce_x(x, mask_idx, backend='fsl')[0]
        return out_py / out_fsl

    @pytest.mark.parametrize('value', [1.0, 2.0, 3.0, 0.2094, 13.562,
                                       40.682, 0.92473, 27.227])
    def test_gap_is_the_top_step_or_nothing(self, value):
        """Either fsl counted the top step, or it is short by exactly it."""
        n_steps = 100
        step = n_steps ** self.H / sum(k ** self.H
                                       for k in range(1, n_steps + 1))
        ratio = self._single_voxel_ratio(value, self.H)
        assert np.isclose(ratio, 1.0, atol=1e-4) or \
            np.isclose(ratio, 1.0 + step, rtol=1e-3), ratio

    def test_python_is_never_below_fsl(self):
        """glow always counts the top step, so it cannot come out lower."""
        for value in (0.2094, 1.0, 13.562, 40.682):
            assert self._single_voxel_ratio(value, self.H) >= 1.0 - 1e-6


class TestNanVoxelParity:
    """A voxel with no statistic costs a voxel, not the image.

    Handed NaN the two backends used to disagree completely: fslmaths
    ignored it, while apply_tfce_img's x.max() went NaN, took every
    height threshold with it and returned an all-zero image. So the same
    cell scored normally on a machine with FSL and zero on one without.
    """

    @staticmethod
    def build():
        """(21, 1000) stats over a 10^3 block, with a blob to detect."""
        rng = np.random.default_rng(0)
        mask_idx = np.arange(1000).reshape((10, 10, 10))
        stat = np.abs(rng.standard_normal((21, mask_idx.size))) * 3
        stat[0, 400:460] += 8
        return stat, mask_idx

    @staticmethod
    def gap(a, b, drop=3):
        """Largest relative difference between two enhanced stacks."""
        a, b = np.delete(a, drop, axis=1), np.delete(b, drop, axis=1)
        return np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))

    @pytest.mark.parametrize('backend', ['python', 'fsl'])
    def test_image_not_collapsed(self, backend):
        """The voxel goes; the image keeps its peak.

        Not bit-identical to the clean run, and it should not be: at the
        lowest heights nearly every voxel joins one cluster, so removing
        a member shifts that cluster's extent by one and every member's
        value with it.
        """
        if backend == 'fsl' and not _tfce.has_fsl():
            pytest.skip('fslmaths not installed')
        stat, mask_idx = self.build()
        clean = AnalysisVBA.apply_tfce(stat=stat.copy(), mask_idx=mask_idx,
                                       backend=backend)
        stat[:, 3] = np.nan
        dirty = AnalysisVBA.apply_tfce(stat=stat, mask_idx=mask_idx,
                                       backend=backend)
        assert np.isnan(dirty[:, 3]).all()
        assert np.nanmax(dirty) == pytest.approx(np.nanmax(clean), rel=1e-6)
        assert np.count_nonzero(np.nan_to_num(dirty)) > 0

    @pytest.mark.parametrize('backend', ['python', 'fsl'])
    def test_detection_survives(self, backend):
        """The planted blob is still found with a voxel dropped."""
        if backend == 'fsl' and not _tfce.has_fsl():
            pytest.skip('fslmaths not installed')
        stat, mask_idx = self.build()
        stat[:, 3] = np.nan
        tfce = AnalysisVBA.apply_tfce(stat=stat, mask_idx=mask_idx,
                                      backend=backend)
        assert np.nanmin(AnalysisVBA.get_fwer(tfce, alpha=.05).pval) < 0.05

    def test_backends_agree_on_a_dropped_voxel(self):
        """A NaN voxel adds nothing to the standing backend difference.

        The two disagree by a fixed amount whatever the input, summing
        over height grids that differ at one endpoint (see the _tfce
        module docstring), so the test is that the gap does not move.
        """
        if not _tfce.has_fsl():
            pytest.skip('fslmaths not installed')
        stat, mask_idx = self.build()

        def both(s):
            """Enhance one stack on each backend."""
            return (AnalysisVBA.apply_tfce(stat=s.copy(), mask_idx=mask_idx,
                                           backend='python'),
                    AnalysisVBA.apply_tfce(stat=s.copy(), mask_idx=mask_idx,
                                           backend='fsl'))

        gap_clean = self.gap(*both(stat))
        stat[:, 3] = np.nan
        gap_nan = self.gap(*both(stat))
        # 1e-3 leaves room for the extent shift the drop itself causes;
        # the bug this guards moved the gap to 1.0 (python returning zeros)
        assert gap_nan == pytest.approx(gap_clean, rel=1e-3)

    def test_apply_tfce_img_survives_nan(self):
        """The public single-image entry point bails safely, not silently."""
        rng = np.random.default_rng(0)
        img = np.abs(rng.standard_normal((8, 8, 8)))
        clean = _tfce.apply_tfce_img(img)
        img[0, 0, 0] = np.nan
        assert _tfce.apply_tfce_img(img).max() == pytest.approx(clean.max())
