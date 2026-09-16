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
            path = tmp_path / f'{sbj}_{feat}.nii.gz'
            _write_nii(path, rng.standard_normal(SHAPE), affine)
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

    @pytest.mark.parametrize('op', ['take_img', 'bootstrap', 'offset',
                                    'permute'])
    def test_dropped_where_it_cannot_be_replayed(self, op):
        """Image selection, an offset and a permutation each drop it.

        A source reloads the images as they were: in the original order,
        with no planted offset, unpermuted. None of those can be brought
        into line with the columns already held, so the source goes rather
        than lie about what a reload would return.
        """
        exp = self._exp()
        if op == 'take_img':
            out = exp.split_img(seed=0)[0]
        elif op == 'bootstrap':
            out = exp.bootstrap_img(n=4, seed=0)
        elif op == 'offset':
            out = exp.add_offset(np.ones((1, 6)), mask=_checkerboard())
        else:
            out = exp.permute(1)
        assert out.source is None
