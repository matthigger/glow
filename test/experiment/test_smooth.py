"""Tests for glow.experiment.smooth.

The claim under test is that borrowing context is exact: smoothing a crop
with a halo as deep as the kernel's own truncation radius gives what
smoothing the whole volume and then cropping gives, value for value.

Experiment.smooth is the entry point, so the reference here is the same
method on an experiment holding every voxel its source has, where the
inflate adds nothing and the kernel runs over the whole volume.
"""

import numpy as np
import pytest

from glow.experiment import ExperimentImageOnly
from glow.experiment.exper import ExperimentScaled
from glow.experiment.smooth import (TRUNCATE, get_sigma_vox,
                                    halo_vox_needed, smooth_volume,
                                    smooth_y)

SHAPE = (16, 17, 18)
AFFINE_2MM = np.diag([2.0, 2.0, 2.0, 1.0])


def _core(shape=SHAPE):
    """Return an interior block, far enough in to have room for a halo."""
    mask = np.zeros(shape, dtype=bool)
    mask[7:10, 7:10, 7:10] = True
    return mask


def _exp(b=1, num_img=5, affine=None, shape=SHAPE):
    """Return a Gaussian experiment over the whole box, with a design."""
    meta = {} if affine is None else {'affine': affine}
    exp = ExperimentImageOnly.from_gauss(b=b, num_img=num_img, shape=shape,
                                         seed=0, meta=meta)
    return exp.sample_x(a=1, seed=0, add_bias=True)


class TestHaloDepth:
    """The halo depth is the kernel's own reach, in voxels."""

    @pytest.mark.parametrize('fwhm, affine, want', [
        (4.0, None, int(TRUNCATE * 4 / np.sqrt(8 * np.log(2)) + 0.5)),
        (4.0, AFFINE_2MM, int(TRUNCATE * 2 / np.sqrt(8 * np.log(2)) + 0.5)),
    ])
    def test_matches_the_truncation_radius(self, fwhm, affine, want):
        """The depth is int(TRUNCATE * sigma + 0.5), sigma in voxels."""
        assert halo_vox_needed(fwhm, affine=affine) == want

    def test_scales_with_the_voxel_size(self):
        """A coarser grid needs fewer voxels for the same physical kernel."""
        assert (halo_vox_needed(8.0, affine=AFFINE_2MM)
                < halo_vox_needed(8.0))

    @pytest.mark.parametrize('fwhm', [2.0, 4.0, 8.0])
    def test_is_where_the_kernel_stops_reading(self, fwhm):
        """A delta smooths to exactly that Chebyshev radius, no further.

        The measurement behind the whole design: what a separable kernel
        reads is the cube of its truncation radius, so a halo of this depth
        holds every voxel that can reach a voxel under study.
        """
        # wide enough that the widest kernel's reach fits inside it, or
        # the volume's own edge would be what the measurement found
        shape = (3 * halo_vox_needed(fwhm) + 1,) * 3
        vol = np.zeros(shape)
        centre = tuple(n // 2 for n in shape)
        vol[centre] = 1.0

        out = smooth_volume(vol, fwhm)
        reach = np.array(np.where(out != 0)) - np.array(centre)[:, np.newaxis]
        assert np.abs(reach).max() == halo_vox_needed(fwhm)


class TestSmoothWithContext:
    """Borrowing context, and giving it back."""

    @pytest.mark.parametrize('fwhm', [2.0, 4.0, 8.0])
    def test_reproduces_a_whole_volume_smooth(self, fwhm):
        """A crop smoothed with context equals the volume smoothed whole.

        The reference is the same images smoothed over every voxel they
        have and only then cropped, which is what a preprocessing pipeline
        would have produced before anyone cut a crop out of it.
        """
        exp = _exp()
        core = _core()
        want = exp.smooth(fwhm).apply_mask(core)
        got = exp.apply_mask(core).smooth(fwhm)

        assert np.array_equal(got.mask_idx, want.mask_idx)
        assert np.allclose(got.y, want.y, rtol=0, atol=1e-7)

    def test_a_halo_one_voxel_short_is_visibly_worse(self):
        """The depth matters: one voxel less and the edge moves.

        Guards the exactness above against passing for the wrong reason
        (a kernel too narrow to reach the crop edge at all).
        """
        exp, core, fwhm = _exp(), _core(), 4.0
        want = exp.smooth(fwhm).apply_mask(core)

        # smooth_y, not the method: the point is a halo the method would
        # never cut, one voxel short of the kernel's reach
        thin = exp.apply_mask(core).inflate(
            halo_vox=halo_vox_needed(fwhm) - 1)
        y = smooth_y(thin.y, mask_idx=thin.mask_idx, fwhm=fwhm)
        got = y[:, :, thin.mask_idx[core]]
        assert not np.allclose(got, want.y, rtol=0, atol=1e-7)

    def test_returns_the_voxels_it_was_given(self):
        """The context is dropped again, numbering included."""
        crop = _exp().apply_mask(_core())
        out = crop.smooth(4.0)
        assert np.array_equal(out.mask_idx, crop.mask_idx)
        assert out.y.shape == crop.y.shape

    @pytest.mark.parametrize('fwhm', [0, 0.0, None])
    def test_no_kernel_is_the_identity(self, fwhm):
        """fwhm of 0 or None hands back the experiment untouched."""
        crop = _exp().apply_mask(_core())
        assert crop.smooth(fwhm) is crop

    def test_no_source_raises(self):
        """Without a source there is no context to read, so it refuses."""
        crop = _exp().apply_mask(_core()).permute(1)
        with pytest.raises(ValueError, match='no source'):
            crop.smooth(4.0)

    def test_the_affine_sets_the_kernel(self):
        """fwhm is in mm, so the same fwhm on a 2 mm grid smooths less."""
        core = _core()
        fine = _exp().apply_mask(core).smooth(4.0)
        coarse = _exp(affine=AFFINE_2MM).apply_mask(core).smooth(4.0)
        # same draw, so a narrower kernel leaves more of the variance
        assert coarse.y.std() > fine.y.std()


class TestNormalization:
    """Smoothing within a mask keeps a constant constant."""

    def test_a_constant_field_is_unchanged(self):
        """S(const) == const: what makes a pre-scaling commute."""
        mask = _core()
        exp = _exp().apply_mask(mask)
        y = np.full_like(exp.y, 7.5)
        out = smooth_y(y, mask_idx=exp.mask_idx, fwhm=4.0)
        assert np.allclose(out, 7.5, rtol=0, atol=1e-6)

    def test_pre_scaling_commutes_with_the_kernel(self):
        """prep(smooth(y)) == smooth(prep(y)), to float32 epsilon.

        pre_scale acts on the feature axis and mean_orig is constant across
        voxels, so neither can be moved by a kernel acting within the
        voxel axis. This is what licenses scaling before smoothing, and
        applying a frozen transform to voxels it was not fit on.
        """
        exp = _exp(b=2, num_img=8)
        core = _core()
        crop = exp.apply_mask(core)
        scaled = ExperimentScaled.from_exp(crop)

        # scale first, then smooth
        got = scaled.smooth(4.0)
        # smooth first, then scale with the same frozen transform
        want = scaled.prep(crop.smooth(4.0).y)

        assert np.allclose(got.y, want, rtol=0, atol=1e-6)
        assert np.array_equal(got.pre_scale, scaled.pre_scale)


class TestBrainEdge:
    """A crop against the images' own edge gets what context there is."""

    def test_exact_where_there_is_no_context_to_have(self, tmp_path):
        """A halo stops at the support, and so does the reference.

        15% of an HCP crop's voxels sit against the brain's own edge, so
        the guarantee has to hold where a full halo is impossible: what a
        whole-volume smooth sees there is the same nothing.
        """
        import nibabel as nib
        import pandas as pd

        rng = np.random.default_rng(0)
        shape = (12, 12, 12)
        paths = {}
        for sbj in ('sbj0', 'sbj1', 'sbj2'):
            path = tmp_path / f'{sbj}.nii.gz'
            nib.Nifti1Image(rng.standard_normal(shape).astype(np.float32),
                            AFFINE_2MM).to_filename(path)
            paths[sbj] = {'feat': path}

        support = np.zeros(shape, dtype=bool)
        support[:6, :, :] = True
        mask_path = tmp_path / 'mask.nii.gz'
        nib.Nifti1Image(support.astype(np.float32),
                        AFFINE_2MM).to_filename(mask_path)

        exp = ExperimentImageOnly.from_paths(
            pd.DataFrame.from_dict(paths, orient='index'), mask=mask_path)
        exp = exp.sample_x(a=1, seed=0, add_bias=True)

        # a crop reaching the support's flat edge
        core = np.zeros(shape, dtype=bool)
        core[3:6, 4:8, 4:8] = True

        want = exp.smooth(4.0).apply_mask(core)
        got = exp.apply_mask(core).smooth(4.0)
        assert np.allclose(got.y, want.y, rtol=0, atol=1e-7)


class TestSigma:
    """The fwhm-to-sigma conversion, where a kernel width is set."""

    def test_fwhm_is_the_full_width_at_half_maximum(self):
        """A kernel of this fwhm is at half its peak fwhm/2 mm out."""
        sigma = get_sigma_vox(4.0, affine=AFFINE_2MM)
        # fwhm 4 mm on a 2 mm grid is 2 voxels wide at half maximum
        peak = np.exp(0.0)
        half_out = np.exp(-(1.0 ** 2) / (2 * sigma[0] ** 2))
        assert np.isclose(half_out / peak, 0.5)
