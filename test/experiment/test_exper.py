import pytest

from glow.experiment.exper import *
from .make_test_image import folder_test_data, img_feat_intensity

_COV = np.array([[2, 3], [3, 10]])


class TestExperimentOnlyImage:
    # one case per (mu, cov) combination.  from_gauss imposes the requested
    # mean exactly (``mean_exp``, zeros when mu is None) and, when a covariance
    # is requested, projects the sample to it exactly (``cov_exp``).  We do NOT
    # assert covariance for the default cov=None cases: there from_gauss only
    # draws standard-normal data, so the sample cov wanders with the draw and
    # any fixed-tolerance check would be flaky rather than a real guarantee.
    @pytest.mark.parametrize('b, mu, cov, mean_exp, cov_exp', [
        (None, None, None, np.zeros(1), None),
        (2, None, None, np.zeros(2), None),
        (None, np.ones(3), None, np.ones(3), None),
        (None, None, _COV, np.zeros(2), _COV),
        (None, np.ones(2), _COV, np.ones(2), _COV),
    ])
    def test_from_gauss(self, b, mu, cov, mean_exp, cov_exp):
        exp = ExperimentImageOnly.from_gauss(b=b, mu=mu, cov=cov)
        n_feat = exp.y.shape[0]
        y = exp.y.reshape((n_feat, -1))

        # mean is imposed exactly
        assert np.allclose(y.mean(axis=1), mean_exp, atol=1e-5)
        # covariance is imposed exactly when (and only when) one is requested
        if cov_exp is not None:
            assert np.allclose(np.atleast_2d(np.cov(y)), cov_exp, atol=1e-5)

    def test_from_search(self):
        # nii
        exp0 = ExperimentImageOnly.from_search(folder=folder_test_data,
                                               sbj_regex=r'img\d',
                                               img_glob_dict={
                                                   'feat0': '*feat0.nii.gz',
                                                   'feat1': '*feat1.nii.gz'})
        # jpg
        exp1 = ExperimentImageOnly.from_search(folder=folder_test_data,
                                               sbj_regex=r'img\d',
                                               img_glob_dict={
                                                   'feat0': '*feat0.jpg',
                                                   'feat1': '*feat1.jpg'})

        for exp in (exp0, exp1):
            # ensure values arrived at their proper place in y
            for img_idx, feat_intense in img_feat_intensity.items():
                for feat_idx, intense in feat_intense.items():
                    assert (exp.y[feat_idx, img_idx, :] == intense).all()

    def test_search_files_bids_repeated_sbj(self, tmp_path):
        # BIDS-derivatives paths repeat the subject id in both the directory
        # and the filename, so a natural regex matches it twice; _search_files
        # must dedupe rather than trip its uniqueness assert.
        for sbj in ('sub-100307', 'sub-200614'):
            dwi = tmp_path / sbj / 'dwi'
            dwi.mkdir(parents=True)
            for feat in ('fa', 'md'):
                (dwi / f'{sbj}_space-MNI152_param-{feat}_dwimap.nii.gz').touch()

        df = ExperimentImageOnly._search_files(
            folder=tmp_path,
            sbj_regex=r'sub-\d+',
            img_glob_dict={'fa': '**/*param-fa*.nii.gz',
                           'md': '**/*param-md*.nii.gz'})

        assert sorted(df.index) == ['sub-100307', 'sub-200614']
        assert sorted(df.columns) == ['fa', 'md']

    def test_search_files_ambiguous_sbj_raises(self, tmp_path):
        # a path carrying two genuinely different subject ids stays ambiguous;
        # the dedupe must not paper over it.
        dwi = tmp_path / 'sub-100307' / 'dwi'
        dwi.mkdir(parents=True)
        (dwi / 'sub-200614_param-fa_dwimap.nii.gz').touch()

        with pytest.raises(AssertionError):
            ExperimentImageOnly._search_files(
                folder=tmp_path,
                sbj_regex=r'sub-\d+',
                img_glob_dict={'fa': '**/*param-fa*.nii.gz'})

    # NOTE: the EffectSynthetic.fit effect_llr round-trip is covered in
    # test/effect/ (test_impose.py::test_compute_offset and
    # test_eff_synthetic.py::test_effect_llr_preserved), so the former
    # test_impose_effect here was redundant and has been removed.

    def test_sample_x(self):
        seed = 0
        exp = Experiment.from_gauss(seed=seed)
        num_img = exp.y.shape[1]

        # a=<n features>, add_bias=True prepends a bias row of ones, so x is
        # (n + 1, num_img) and contrast gains a leading False (bias term).
        n = exp.x.shape[0]
        exp_a = exp.sample_x(a=n, add_bias=True)
        assert exp_a.x.shape == (n + 1, num_img)
        assert len(exp_a.contrast) == n + 1
        assert np.all(exp_a.x[0] == 1)
        assert not exp_a.contrast[0]
        assert np.all(exp_a.contrast[1:])

        # contrast=<existing>, add_bias=True: a is taken from contrast.size,
        # bias prepended the same way.
        c = exp.contrast
        exp_c = exp.sample_x(contrast=c, add_bias=True)
        assert exp_c.x.shape == (c.size + 1, num_img)
        assert len(exp_c.contrast) == c.size + 1
        assert np.all(exp_c.x[0] == 1)
        assert not exp_c.contrast[0]
        assert np.array_equal(exp_c.contrast[1:], c)

        # a=4, add_bias=True -> x is (5, num_img), contrast (5,)
        exp_4 = exp.sample_x(a=4, add_bias=True)
        assert exp_4.x.shape == (5, num_img)
        assert len(exp_4.contrast) == 5
        assert np.all(exp_4.x[0] == 1)

    @pytest.mark.parametrize('str_test_glob', ['*test_bw.png', '*test.png'])
    def test_bootstrap_img_count(self, str_test_glob):
        """bootstrap_img returns exactly n resampled images"""
        n = 100
        exp = ExperimentImageOnly.from_search(folder=folder_test_data,
                                              sbj_regex='squares',
                                              img_glob_dict={
                                                  'feat': str_test_glob})
        exp = exp.bootstrap_img(n=n, seed=0)
        assert exp.y.shape[1] == n

    @pytest.mark.parametrize('str_test_glob', ['*test_bw.png', '*test.png'])
    def test_bootstrap_img_no_feature_mixing(self, str_test_glob):
        """bootstrap_img preserves per-feature data (no reshape mixing)"""
        rng = np.random.default_rng(seed=0)
        exp = ExperimentImageOnly.from_search(folder=folder_test_data,
                                              sbj_regex='squares',
                                              img_glob_dict={
                                                  'feat': str_test_glob})
        exp = exp.bootstrap_img(n=100, seed=0)
        b = exp.y.shape[0]

        # Give each feature a distinct, very large mean (the 1e6 factor
        # makes the per-feature means differ by ~1e6, far larger than the
        # unit-scale image values).  If bootstrap_img's internal
        # reshape/cov ever mixed features, the means would bleed into each
        # other and this large separation would expose it; a small offset
        # could be masked by the image's own variation.
        new_mean = rng.standard_normal(size=b) * 1e6
        diff = new_mean - exp.y.mean(axis=(1, 2))
        exp.y += diff[:, np.newaxis, np.newaxis]
        exp = exp.bootstrap_img(n=10, seed=0)
        assert np.allclose(new_mean, exp.y.mean(axis=(1, 2)))

    def test_bootstrap_img_takes_whole_images(self):
        """A resampled image brings its design column and label with it

        Resampling y alone would leave x describing the images the draw
        replaced, silently, whenever n differs from the original count.
        """
        num_img = 12
        exp = Experiment.from_gauss(a=1, b=2, shape=(4, 4),
                                    num_img=num_img, seed=0)
        boot = exp.bootstrap_img(n=30, seed=0)

        assert boot.x.shape == (exp.x.shape[0], 30)
        assert len(boot.meta['subjects']) == 30

        # the subject labels name which draw each image came from, so
        # they pin y and x to the same one
        idx = np.array([int(s.split('_')[1]) for s in boot.meta['subjects']])
        assert np.array_equal(boot.y, exp.y[:, idx, :])
        assert np.array_equal(boot.x, exp.x[:, idx])

    def test_bootstrap_img_noise_keeps_dtype(self):
        """Noise is added without promoting y out of its dtype"""
        exp = ExperimentImageOnly.from_gauss(b=2, shape=(4, 4), num_img=8,
                                             seed=0)
        boot = exp.bootstrap_img(n=20, seed=0, noise_scale=.5)

        assert boot.y.dtype == exp.y.dtype
        assert boot.y.shape[1] == 20


class TestExperiment:

    def test_permute(self):
        perm_idx = 1
        shape = 10, 10
        num_img = 5
        exp = Experiment.from_gauss(shape=shape, seed=0, num_img=num_img)

        # manually permute in a loop, check that einsum does the same
        exp_permuted = exp.permute(perm_idx=perm_idx)

        freed_lane = get_freed_lane(x=exp.x, contrast=exp.contrast,
                                    perm_idx=perm_idx)
        for vox_idx in range(np.prod(shape)):
            y_permute_exp = exp.y[..., vox_idx] @ freed_lane
            assert np.allclose(exp_permuted.y[..., vox_idx], y_permute_exp)


class TestExperimentScaled:
    # atol=1e-5 throughout: the default float32 experiment dtype caps eigh /
    # cov residuals around ~1e-7 (float64 gave ~1e-15), but the semantics --
    # "the basis was diagonalised to working precision" -- are unchanged.

    @staticmethod
    def _scaled():
        exp = Experiment.from_gauss(shape=(10, 10), seed=0, b=3, num_img=5)
        return exp, ExperimentScaled.from_exp(exp)

    def test_mean_zeroed(self):
        """scaling centres every feature at zero mean"""
        _exp, exp_scale = self._scaled()
        assert np.allclose(exp_scale.y.mean(axis=(1, 2)), 0, atol=1e-5)

    def test_whitened_off_diagonal_zero(self):
        """scaling whitens features (zero off-diagonal covariance)"""
        _exp, exp_scale = self._scaled()
        b = exp_scale.y.shape[0]
        cov_after = np.cov(exp_scale.y.reshape((b, -1), order='F'))
        off_diag = ~np.eye(b).astype(bool)
        assert np.allclose(cov_after[off_diag], 0, atol=1e-5)

    def test_prep_inv_round_trip(self):
        """prep_inv inverts prep, recovering the original y"""
        exp, exp_scale = self._scaled()
        assert np.allclose(exp.y, exp_scale.prep_inv(exp_scale.y),
                           rtol=1e-4, atol=1e-5)

    def test_prep_maps_pca_directions_to_basis(self):
        """prep maps each PCA direction onto a standard basis vector"""
        exp, exp_scale = self._scaled()
        b = exp_scale.y.shape[0]

        cov = np.cov(exp.y.reshape(b, -1))
        scale = np.diag(1 / np.diag(cov) ** .5)
        cov_scale = scale @ cov @ scale.T
        _evals, evecs = np.linalg.eigh(cov_scale)

        for evec, e in zip(evecs.T, np.eye(b)):
            e = e[:, np.newaxis, np.newaxis]
            e_preimage = np.diag(1 / np.diag(scale)) @ evec
            e_preimage = (e_preimage[:, np.newaxis, np.newaxis] +
                          exp_scale.mean_orig)
            assert np.allclose(exp_scale.prep(e_preimage), e,
                               rtol=1e-4, atol=1e-5)


class TestExperimentScaledZeroVariance:
    """ExperimentScaled should raise on zero-variance features"""

    def test_zero_variance_raises(self):
        import pytest
        exp = Experiment.from_gauss(a=2, b=2, shape=(5, 5), num_img=20, seed=0)
        # zero out one feature entirely → zero variance
        exp.y[1, :, :] = 0.0

        with pytest.raises(ValueError, match='zero-variance'):
            ExperimentScaled.from_exp(exp)


class TestDropConstantVox:
    """drop_constant_vox removes the voxels with nothing to test."""

    DEAD_V = 7

    @staticmethod
    def build_exp(shape=(4, 4, 4), b=3, num_img=20, seed=0):
        """A small experiment whose voxels all vary."""
        return Experiment.from_gauss(a=2, b=b, shape=shape,
                                     num_img=num_img, seed=seed)

    @staticmethod
    def quiet_drop(exp):
        """drop_constant_vox with the expected warning suppressed."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConstantVoxelWarning)
            return exp.drop_constant_vox()

    def kill(self, exp, feat=slice(None), value=1.234):
        """Flatten one voxel across images, on one feature or all."""
        exp.y[feat, :, self.DEAD_V] = value
        return exp

    def test_healthy_keeps_everything(self):
        """Data that varies passes through with nothing dropped."""
        exp = self.build_exp()
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConstantVoxelWarning)
            out = exp.drop_constant_vox()
        assert out.num_vox_dropped == 0
        assert out.y.shape == exp.y.shape
        assert not out.mask_dead.any()

    def test_constant_voxel_removed(self):
        """The dead voxel leaves y, and mask_idx is renumbered over it."""
        exp = self.kill(self.build_exp())
        num_vox = exp.y.shape[2]
        with pytest.warns(ConstantVoxelWarning, match='dropped 1 of'):
            out = exp.drop_constant_vox()
        assert out.num_vox_dropped == 1
        assert out.y.shape[2] == num_vox - 1
        # renumbered contiguously, so downstream indexing still works
        kept = out.mask_idx[out.mask_idx > -1]
        np.testing.assert_array_equal(np.sort(kept), np.arange(num_vox - 1))

    def test_mask_dead_marks_the_right_voxel(self):
        """mask_dead is in image space, and names the voxel that went."""
        exp = self.kill(self.build_exp())
        dead_pos = np.argwhere(exp.mask_idx == self.DEAD_V)[0]
        out = self.quiet_drop(exp)
        assert out.mask_dead.shape == exp.mask_idx.shape
        assert out.mask_dead[tuple(dead_pos)]
        assert out.mask_dead.sum() == 1
        assert out.mask_idx[tuple(dead_pos)] == -1

    def test_one_flat_feature_is_enough(self):
        """A single flat feature makes E rank-deficient, so the voxel goes."""
        exp = self.kill(self.build_exp(), feat=1)
        assert self.quiet_drop(exp).num_vox_dropped == 1

    def test_float32_cliff(self):
        """A voxel below the float32 underflow cliff is caught."""
        exp = self.build_exp()
        exp.y = exp.y.astype(np.float32)
        rng = np.random.default_rng(2)
        exp.y[:, :, self.DEAD_V] = (
            1.0 + rng.standard_normal((3, 20)) * 1e-9).astype(np.float32)
        assert self.quiet_drop(exp).num_vox_dropped == 1

    def test_all_constant_raises(self):
        """A wholly flat experiment is an error, not an empty analysis."""
        exp = self.build_exp()
        exp.y[:] = 1.0
        with pytest.raises(ValueError, match='every voxel is constant'):
            exp.drop_constant_vox()

    def test_source_experiment_untouched(self):
        """Dropping returns a new experiment and leaves this one alone."""
        exp = self.kill(self.build_exp())
        num_vox = exp.y.shape[2]
        self.quiet_drop(exp)
        assert exp.y.shape[2] == num_vox
        assert exp.mask_dead is None
        assert exp.num_vox_dropped == 0

    def test_mask_dead_survives_the_analysis_path(self):
        """The record follows the experiment through permute and scaling.

        mask_dead has to be a named __init__ parameter for this: both
        rebuild through the constructor, and **kwargs would swallow it.
        """
        out = self.quiet_drop(self.kill(self.build_exp()))
        assert out.permute(3).num_vox_dropped == 1
        assert ExperimentScaled.from_exp(out).num_vox_dropped == 1
        assert out.apply_mask(out.mask_idx > -1).num_vox_dropped == 1

    def test_repr_carries_the_count(self):
        """The count rides the repr, which is what the recorder stores."""
        exp = self.build_exp()
        assert 'num_vox_dropped' not in repr(exp), 'not screened yet'

        clean = self.quiet_drop(exp)
        assert 'num_vox_dropped=0' in repr(clean)

        dropped = self.quiet_drop(self.kill(self.build_exp()))
        assert 'num_vox_dropped=1' in repr(dropped)

    def test_repr_distinguishes_unscreened_from_clean(self):
        """None means never screened; all-False means screened and clean."""
        exp = self.build_exp()
        assert exp.mask_dead is None
        assert 'num_vox_dropped' not in repr(exp)
        assert repr(self.quiet_drop(exp)).endswith('num_vox_dropped=0)')


class TestSplitImg:
    """split_img cuts images into a segmentation fold and a test fold."""

    NUM_IMG = 20

    @staticmethod
    def build_exp(num_img=NUM_IMG, a=1, shape=(4, 4), b=2, seed=0):
        """A small experiment whose subject labels track image order."""
        return Experiment.from_gauss(a=a, b=b, shape=shape,
                                     num_img=num_img, seed=seed)

    @staticmethod
    def img_idx(exp):
        """Recover original image indices from the subject labels.

        from_gauss names image i 'subject_00i', so the labels are an
        independent record of which images a fold took -- one that has to
        agree with y and x for the split to be self-consistent.
        """
        return np.array([int(s.split('_')[1]) for s in exp.meta['subjects']])

    def test_partition_is_exact(self):
        """Ungrouped, the folds are disjoint, exhaustive and exactly sized"""
        exp = self.build_exp()
        seg, test = exp.split_img(frac_segment=.6, seed=0)

        idx_seg, idx_test = self.img_idx(seg), self.img_idx(test)
        assert not set(idx_seg) & set(idx_test)
        assert set(idx_seg) | set(idx_test) == set(range(self.NUM_IMG))
        assert seg.y.shape[1] == len(idx_seg) == 12
        assert test.y.shape[1] == len(idx_test) == 8

    def test_folds_carry_their_own_images(self):
        """y and x follow the same images the subject labels name"""
        exp = self.build_exp()
        for fold in exp.split_img(seed=0):
            idx = self.img_idx(fold)
            assert np.array_equal(fold.y, exp.y[:, idx, :])
            assert np.array_equal(fold.x, exp.x[:, idx])
            assert np.array_equal(fold.contrast, exp.contrast)

    def test_voxels_are_shared(self):
        """Both folds keep every voxel, so one fold's tree fits the other"""
        exp = self.build_exp()
        seg, test = exp.split_img(seed=0)

        for fold in (seg, test):
            assert np.array_equal(fold.mask_idx, exp.mask_idx)
            assert fold.y.shape[2] == exp.y.shape[2]
            assert fold.y.shape[0] == exp.y.shape[0]

    def test_seed_reproduces(self):
        """Same seed gives the same partition, a different seed does not"""
        exp = self.build_exp()
        idx = self.img_idx(exp.split_img(seed=3)[0])

        assert np.array_equal(idx, self.img_idx(exp.split_img(seed=3)[0]))
        assert not np.array_equal(idx, self.img_idx(exp.split_img(seed=4)[0]))

    def test_split_is_blind_to_y(self):
        """The partition depends on the images' identity, never their values

        Selecting on y would put back the selection bias the split exists
        to remove, so a wholly different y must not move the cut.
        """
        exp = self.build_exp()
        other = exp._copy_with(y=exp.y * -3 + 1)

        assert np.array_equal(self.img_idx(exp.split_img(seed=0)[0]),
                              self.img_idx(other.split_img(seed=0)[0]))

    @pytest.mark.parametrize('seed', range(5))
    def test_group_is_never_broken(self, seed):
        """Every image sharing a label lands in the same fold"""
        exp = self.build_exp()
        group = np.repeat(np.arange(self.NUM_IMG // 2), 2)
        seg, test = exp.split_img(seed=seed, group=group)

        set_seg = set(group[self.img_idx(seg)])
        set_test = set(group[self.img_idx(test)])
        assert not set_seg & set_test
        assert set_seg | set_test == set(group)

    def test_group_approximates_the_fraction(self):
        """Whole groups move, so the realized fraction only approximates"""
        exp = self.build_exp()
        group = np.repeat(np.arange(self.NUM_IMG // 4), 4)
        seg, _test = exp.split_img(frac_segment=.5, seed=0, group=group)

        assert abs(seg.y.shape[1] - self.NUM_IMG / 2) <= 4

    def test_group_of_one_matches_ungrouped(self):
        """A group per image is the ungrouped case, on the same code path"""
        exp = self.build_exp()
        alone = np.arange(self.NUM_IMG)

        assert np.array_equal(self.img_idx(exp.split_img(seed=0)[0]),
                              self.img_idx(exp.split_img(seed=0,
                                                         group=alone)[0]))

    def test_rank_deficient_fold_raises(self):
        """A covariate stranded in one fold is caught, not left to decompose"""
        exp = self.build_exp()
        rare = np.zeros(self.NUM_IMG)
        rare[7] = 1
        exp.x = np.vstack([np.ones(self.NUM_IMG), rare]).astype(exp.y.dtype)
        exp.contrast = np.array([False, True])

        with pytest.raises(ValueError, match='lost rank'):
            exp.split_img(seed=0)

    def test_no_residual_space_raises(self):
        """A fold with num_img <= a leaves nothing to estimate error from"""
        exp = self.build_exp(num_img=8, a=5)

        with pytest.raises(ValueError, match='no residual space'):
            exp.split_img(frac_segment=.5, seed=0)

    def test_empty_fold_raises(self):
        """One group holding every image cannot be cut in two"""
        exp = self.build_exp()

        with pytest.raises(ValueError, match='leaves a fold empty'):
            exp.split_img(seed=0, group=np.zeros(self.NUM_IMG))

    @pytest.mark.parametrize('frac', [0, 1, -.5, 1.5])
    def test_frac_out_of_range_rejected(self, frac):
        with pytest.raises(AssertionError, match='frac_segment'):
            self.build_exp().split_img(frac_segment=frac, seed=0)

    def test_group_length_checked(self):
        with pytest.raises(AssertionError, match='one label per image'):
            self.build_exp().split_img(seed=0, group=np.arange(3))

    def test_image_only_splits_without_a_design(self):
        """ExperimentImageOnly splits too -- there is just no x to check"""
        exp = ExperimentImageOnly.from_gauss(shape=(4, 4), b=2,
                                             num_img=self.NUM_IMG, seed=0)
        seg, test = exp.split_img(frac_segment=.5, seed=0)

        assert seg.y.shape[1] == test.y.shape[1] == self.NUM_IMG // 2
        assert not set(self.img_idx(seg)) & set(self.img_idx(test))

    def test_scaled_refuses_to_split(self):
        """Splitting after scaling would leak the test fold's whitening"""
        exp = ExperimentScaled.from_exp(self.build_exp())

        with pytest.raises(TypeError, match='split before scaling'):
            exp.split_img(seed=0)
