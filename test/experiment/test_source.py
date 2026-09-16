"""Tests for glow.experiment.source: reload a voxel subset, exactly.

The contract every source shares is that load(mask) returns what the
loader would have put in those columns, bit for bit and in get_mask_idx
order, so an experiment can grow back voxels it cropped away.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from glow.experiment import ExperimentImageOnly
from glow.experiment.source import SourceGauss, SourceNifti

SHAPE = (4, 5, 6)
CONST_VOX = (3, 4, 5)


def _checkerboard(shape=SHAPE):
    """Return a boolean mask of every other voxel, in C order.

    Interleaved rather than a contiguous block, so a source that gathered
    in the wrong order (or off a transposed mask) could not accidentally
    agree with the columns it is checked against.
    """
    mask = np.zeros(np.prod(shape), dtype=bool)
    mask[::2] = True
    return mask.reshape(shape)


def _write_nii(path, arr, affine):
    """Write arr to path as a float32 NIfTI on the given affine."""
    nib.Nifti1Image(np.asarray(arr, dtype=np.float32), affine).to_filename(
        path)


@pytest.fixture
def nii_paths(tmp_path):
    """Write two subjects x two features of random volumes, plus a mask.

    Returns:
        df (pd.DataFrame): index=subject, columns=feature, values=paths
        mask_path (pathlib.Path): an explicit support, two voxels off
        affine (np.array): (4, 4) the shared affine
    """
    rng = np.random.default_rng(0)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    paths = {}
    for sbj in ('sbj0', 'sbj1'):
        for feat in ('featA', 'featB'):
            arr = rng.standard_normal(SHAPE)
            # one voxel constant across images, for drop_constant_vox
            arr[CONST_VOX] = 7.0
            path = tmp_path / f'{sbj}_{feat}.nii.gz'
            _write_nii(path, arr, affine)
            paths.setdefault(sbj, {})[feat] = path

    mask = np.ones(SHAPE, dtype=bool)
    mask[0, 0, 0] = False
    mask[1, 2, 3] = False
    mask_path = tmp_path / 'mask.nii.gz'
    _write_nii(mask_path, mask, affine)

    return pd.DataFrame.from_dict(paths, orient='index'), mask_path, affine


class TestSourceGauss:
    """A Gaussian source redraws its box and gathers from it."""

    def test_from_gauss_carries_its_source(self):
        """from_gauss attaches the SourceGauss it drew through."""
        exp = ExperimentImageOnly.from_gauss(b=2, num_img=3, shape=SHAPE,
                                             seed=0)
        assert isinstance(exp.source, SourceGauss)
        assert exp.source.shape == SHAPE
        assert exp.source.b == 2
        assert exp.source.num_img == 3
        assert exp.source.features == ('feat_0', 'feat_1')

    @pytest.mark.parametrize('mu, cov', [
        (None, None),
        (np.ones(2), None),
        (None, np.array([[2.0, 3.0], [3.0, 10.0]])),
    ])
    def test_reload_is_bit_identical(self, mu, cov):
        """load() over the whole support reproduces the drawn y exactly.

        Covers the covariance projection and the mean shift, which are
        applied over every voxel at once and so are the parts a subset
        draw could not reproduce.
        """
        exp = ExperimentImageOnly.from_gauss(b=2, num_img=3, shape=SHAPE,
                                             seed=0, mu=mu, cov=cov)
        assert np.array_equal(exp.source.load(), exp.y)
        assert exp.source.load().dtype == exp.y.dtype

    def test_subset_matches_the_columns_it_stands_for(self):
        """load(mask) equals the y columns mask_idx numbers for that mask."""
        exp = ExperimentImageOnly.from_gauss(b=2, num_img=3, shape=SHAPE,
                                             seed=0)
        mask = _checkerboard()
        want = exp.y[:, :, exp.mask_idx[mask]]
        assert np.array_equal(exp.source.load(mask), want)

    def test_wrong_shape_raises(self):
        """A mask off the source's grid is refused, not silently gathered."""
        exp = ExperimentImageOnly.from_gauss(shape=SHAPE, seed=0)
        with pytest.raises(ValueError, match='does not match'):
            exp.source.load(np.ones((2, 2, 2), dtype=bool))


class TestSourceNifti:
    """A NIfTI source reads each volume again and gathers from it."""

    def test_reload_is_bit_identical(self, nii_paths):
        """load() reproduces what load_image_nii put in y, bit for bit."""
        df, mask_path, _ = nii_paths
        exp = ExperimentImageOnly.from_paths(df, mask=mask_path)
        assert isinstance(exp.source, SourceNifti)
        assert np.array_equal(exp.source.load(), exp.y)
        assert exp.source.features == ('featA', 'featB')

    def test_subset_matches_the_columns_it_stands_for(self, nii_paths):
        """load(mask) equals the y columns mask_idx numbers for that mask."""
        df, mask_path, _ = nii_paths
        exp = ExperimentImageOnly.from_paths(df, mask=mask_path)
        mask = _checkerboard() & (exp.mask_idx > -1)
        want = exp.y[:, :, exp.mask_idx[mask]]
        assert np.array_equal(exp.source.load(mask), want)

    def test_support_is_the_realized_mask(self, nii_paths):
        """The source's support is what the loader kept, not what it read."""
        df, mask_path, _ = nii_paths
        exp = ExperimentImageOnly.from_paths(df, mask=mask_path)
        assert np.array_equal(exp.source.mask, exp.mask_idx > -1)
        assert not exp.source.mask[0, 0, 0]

    def test_reaching_past_the_support_raises(self, nii_paths):
        """Asking for a voxel the source does not hold is refused."""
        df, mask_path, _ = nii_paths
        exp = ExperimentImageOnly.from_paths(df, mask=mask_path)
        with pytest.raises(ValueError, match='outside the source support'):
            exp.source.load(np.ones(SHAPE, dtype=bool))

    def test_inferred_support_reloads(self, tmp_path):
        """The every-image-nonzero support is carried and reloads exactly."""
        rng = np.random.default_rng(1)
        affine = np.eye(4)
        paths = {}
        for sbj_idx, sbj in enumerate(('sbj0', 'sbj1')):
            arr = rng.standard_normal(SHAPE)
            # a voxel zero in one image drops out of the inferred support
            arr[0, 0, sbj_idx] = 0
            path = tmp_path / f'{sbj}_feat.nii.gz'
            _write_nii(path, arr, affine)
            paths[sbj] = {'feat': path}

        df = pd.DataFrame.from_dict(paths, orient='index')
        exp = ExperimentImageOnly.from_paths(df)
        assert exp.source.mask.sum() == np.prod(SHAPE) - 2
        assert np.array_equal(exp.source.load(), exp.y)


class TestLayout:
    """Every source loads in the layout the statistics read.

    get_mancova reshapes y to (b, num_img * num_vox) in Fortran order --
    a view of an F-contiguous array, a copy of anything else -- and
    Experiment.inflate and smooth_y allocate the same way, so the layout
    is part of the contract rather than whatever the last gather produced.
    """

    def test_gauss_loads_fortran_order(self):
        """Whole support or a subset, the gather comes back F-contiguous."""
        exp = ExperimentImageOnly.from_gauss(b=2, num_img=3, shape=SHAPE,
                                             seed=0)
        assert exp.source.load().flags['F_CONTIGUOUS']
        assert exp.source.load(_checkerboard()).flags['F_CONTIGUOUS']

    def test_nifti_loads_fortran_order(self, nii_paths):
        """The NIfTI reader allocates in that layout too."""
        df, mask_path, _ = nii_paths
        exp = ExperimentImageOnly.from_paths(df, mask=mask_path)
        assert exp.source.load().flags['F_CONTIGUOUS']
        subset = _checkerboard() & exp.source.mask
        assert exp.source.load(subset).flags['F_CONTIGUOUS']

    def test_the_reshape_the_statistics_make_is_a_view(self):
        """What the layout is for: no copy of y on the way into a fit."""
        exp = ExperimentImageOnly.from_gauss(b=2, num_img=3, shape=SHAPE,
                                             seed=0)
        y = exp.source.load()
        flat = y.reshape((y.shape[0], -1), order='F')
        assert np.shares_memory(flat, y)


class TestSourceIsCarried:
    """Which operations keep a source, and which cannot honour one."""

    def _exp(self):
        """Return a small Gaussian experiment with a design matrix."""
        exp = ExperimentImageOnly.from_gauss(b=1, num_img=6, shape=SHAPE,
                                             seed=0)
        return exp.sample_x(a=1, seed=0, add_bias=True)

    def test_kept_by_voxel_selection(self):
        """apply_mask and drop_constant_vox only drop voxels, so it stays."""
        exp = self._exp()
        assert exp.source is not None
        assert exp.apply_mask(_checkerboard()).source is exp.source
        assert exp.drop_constant_vox().source is exp.source

    def test_kept_by_an_offset_which_is_recorded(self):
        """An offset is replayable, so it is recorded rather than refused."""
        exp = self._exp()
        offset = np.ones((1, 6), dtype=exp.y.dtype)
        out = exp.add_offset(offset, mask=_checkerboard())

        assert out.source is exp.source
        assert len(out.patch_list) == 1
        assert np.array_equal(out.patch_list[0]['offset'], offset)
        assert not exp.patch_list

    @pytest.mark.parametrize('op', ['take_img', 'bootstrap', 'sigma',
                                    'permute'])
    def test_dropped_where_it_cannot_be_replayed(self, op):
        """Image selection, a sigma stretch and a permutation each drop it.

        A source reloads the images as they were: in the original order,
        unpermuted. Neither that nor a stretch of the deviations from a
        support's own voxel mean can be brought into line with the columns
        already held, so the source goes rather than lie about what a
        reload would return.
        """
        exp = self._exp()
        if op == 'take_img':
            out = exp.split_img(seed=0)[0]
        elif op == 'bootstrap':
            out = exp.bootstrap_img(n=4, seed=0)
        elif op == 'sigma':
            out = exp.add_offset(np.zeros((1, 6)), mask=_checkerboard(),
                                 sigma_scale=2.0)
        else:
            out = exp.permute(1)
        assert out.source is None


class TestInflate:
    """Growing an experiment back, and getting the same one when cropped."""

    @staticmethod
    def _core(shape=SHAPE):
        """Return a small interior block, well inside any halo."""
        mask = np.zeros(shape, dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        return mask

    def _gauss(self):
        """Return a Gaussian experiment cropped to the interior block."""
        exp = ExperimentImageOnly.from_gauss(b=2, num_img=5, shape=SHAPE,
                                             seed=0)
        return exp.sample_x(a=1, seed=0, add_bias=True).apply_mask(
            self._core())

    def test_round_trip_is_exact(self):
        """Inflating then cropping back returns the experiment it started as.

        The single strongest statement about a source: the numbering comes
        back identical (get_mask_idx is a function of the mask) and so does
        every value, so a fit that inflates internally tests exactly the
        voxels it was handed.
        """
        crop = self._gauss()
        back = crop.inflate(halo_vox=2).apply_mask(crop.mask_idx > -1)
        assert np.array_equal(back.mask_idx, crop.mask_idx)
        assert np.array_equal(back.y, crop.y)

    def test_added_voxels_come_from_the_source(self):
        """The halo holds what the source reads there, nothing invented."""
        crop = self._gauss()
        wide = crop.inflate(halo_vox=1)
        mask_add = (wide.mask_idx > -1) & (crop.mask_idx == -1)

        assert mask_add.any()
        assert np.array_equal(wide.y[:, :, wide.mask_idx[mask_add]],
                              crop.source.load(mask_add))

    def test_growth_stops_at_the_source_support(self, nii_paths):
        """A halo reaches as far as the images do and no further."""
        df, mask_path, _ = nii_paths
        exp = ExperimentImageOnly.from_paths(df, mask=mask_path)
        crop = exp.apply_mask(self._core())

        wide = crop.inflate(halo_vox=10)
        assert np.array_equal(wide.mask_idx > -1, exp.source.mask)
        assert not (wide.mask_idx > -1)[0, 0, 0]

    def test_screened_voxels_come_back_as_context(self, nii_paths):
        """A voxel drop_constant_vox removed returns as context, not data.

        A filter reading the parent images sees it, so the halo has to as
        well; mask_dead still names it, so cropping to what was analysed
        leaves it behind again.
        """
        df, mask_path, _ = nii_paths
        exp = ExperimentImageOnly.from_paths(df, mask=mask_path)
        with pytest.warns(UserWarning, match='constant across images'):
            screened = exp.drop_constant_vox()

        assert screened.mask_dead[CONST_VOX]
        assert screened.mask_idx[CONST_VOX] == -1

        wide = screened.inflate(halo_vox=1)
        assert wide.mask_idx[CONST_VOX] > -1
        assert np.allclose(wide.y[:, :, wide.mask_idx[CONST_VOX]], 7.0)
        back = wide.apply_mask(screened.mask_idx > -1)
        assert back.mask_idx[CONST_VOX] == -1

    def test_a_planted_offset_is_replayed(self):
        """An offset recorded by add_offset lands on the voxels it covers."""
        crop = self._gauss()
        support = self._core().copy()
        support[2, 1:3, 1:3] = False
        offset = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
        plant = crop.add_offset(offset, mask=support)

        # the round trip still holds with a plant in the way
        back = plant.inflate(halo_vox=2).apply_mask(crop.mask_idx > -1)
        assert np.array_equal(back.y, plant.y)

        # and the halo is unplanted, because the support does not reach it
        wide = plant.inflate(halo_vox=1)
        mask_add = (wide.mask_idx > -1) & (crop.mask_idx == -1)
        assert np.array_equal(wide.y[:, :, wide.mask_idx[mask_add]],
                              crop.source.load(mask_add))

    def test_a_patch_records_what_was_applied(self):
        """A support reaching past the crop is recorded clipped to it.

        add_offset reaches only voxels the experiment holds, so that is
        what the patch says: an inflate replays the act, it does not
        extrapolate the intent.
        """
        crop = self._gauss()
        offset = np.zeros((2, 5), dtype=np.float32)
        plant = crop.add_offset(offset, mask=np.ones(SHAPE, dtype=bool))
        assert np.array_equal(plant.patch_list[0]['mask'],
                              crop.mask_idx > -1)

    def test_scaled_inflate_preps_what_it_loads(self):
        """A scaled experiment grows in its own space, transform frozen."""
        from glow.experiment.exper import ExperimentScaled

        crop = self._gauss()
        offset = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
        scaled = ExperimentScaled.from_exp(crop.add_offset(
            offset, mask=self._core()))

        wide = scaled.inflate(halo_vox=1)
        assert np.array_equal(wide.pre_scale, scaled.pre_scale)

        mask_add = (wide.mask_idx > -1) & (crop.mask_idx == -1)
        assert np.allclose(wide.y[:, :, wide.mask_idx[mask_add]],
                           scaled.prep(crop.source.load(mask_add)))

        back = wide.apply_mask(scaled.mask_idx > -1)
        assert np.array_equal(back.y, scaled.y)

    def test_no_source_raises(self):
        """An experiment that cannot read its images refuses to grow."""
        crop = self._gauss()
        with pytest.raises(ValueError, match='no source'):
            crop.permute(1).inflate(halo_vox=1)

    def test_halo_xor_mask(self):
        """Exactly one of halo_vox and mask says how far to grow."""
        crop = self._gauss()
        with pytest.raises(AssertionError, match='xor'):
            crop.inflate()
        with pytest.raises(AssertionError, match='xor'):
            crop.inflate(halo_vox=1, mask=np.ones(SHAPE, dtype=bool))

    def test_an_explicit_mask_keeps_what_is_held(self):
        """Growing to a mask never drops a voxel already held."""
        crop = self._gauss()
        elsewhere = np.zeros(SHAPE, dtype=bool)
        elsewhere[0, 0, 1] = True

        wide = crop.inflate(mask=elsewhere)
        assert np.array_equal(wide.mask_idx > -1,
                              (crop.mask_idx > -1) | elsewhere)
