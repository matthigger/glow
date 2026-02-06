from collections import namedtuple

from glow.experiment.exper import *
from .make_test_image import folder_test_data, img_feat_intensity


class TestExperimentOnlyImage:
    def test_from_gauss(self):
        Case = namedtuple('Case', ['b', 'mu', 'cov'])
        _cov = np.array([[2, 3], [3, 10]])
        case_list = [Case(b=None, mu=None, cov=None),
                     Case(b=2, mu=None, cov=None),
                     Case(b=None, mu=np.ones(3), cov=None),
                     Case(b=None, mu=None, cov=_cov),
                     Case(b=None, mu=np.ones(2), cov=_cov),
                     ]

        for case in case_list:
            exp = ExperimentImageOnly.from_gauss(b=case.b,
                                                 mu=case.mu,
                                                 cov=case.cov)
            b, num_img, num_vox = exp.y.shape
            y = exp.y.reshape((b, -1))

            if case.mu is None:
                assert np.allclose(y.mean(axis=1), np.zeros(b), rtol=1e-5, atol=1e-5)
            else:
                assert np.allclose(y.mean(axis=1), case.mu, rtol=1e-5, atol=1e-5)

            if case.cov is not None:
                assert np.allclose(np.cov(y), case.cov, rtol=1e-5, atol=1e-5)

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

    def test_impose_effect(self):
        seed = 0
        exp = Experiment.from_gauss(seed=seed)
        extenter = glow.effect.ExtenterSphere(radius=3)

        for hotel_tr in [0, 1, 100]:
            _exp, effect = exp.impose_effect(seed=seed, extenter=extenter,
                                             hotel_tr=hotel_tr)
            assert np.isclose(effect.hotel_tr, hotel_tr)

    def test_sample_x(self):
        seed = 0
        exp = Experiment.from_gauss(seed=seed)
        exp.sample_x(a=exp.x.shape[0], add_bias=True)
        exp.sample_x(contrast=exp.contrast, add_bias=True)
        exp.sample_x(a=4, add_bias=True)

    def test_bootstrap_img(self):
        n = 100
        rng = np.random.default_rng(seed=0)
        for str_test_glob in ('*test_bw.png', '*test.png'):
            exp = ExperimentImageOnly.from_search(folder=folder_test_data,
                                                  sbj_regex='squares',
                                                  img_glob_dict={
                                                      'feat': str_test_glob})

            exp = exp.bootstrap_img(n=n, seed=0)
            b, num_img, num_vox = exp.y.shape
            assert num_img == n

            # validate reshaping (ensure cov compute is correct and doesn't
            # mix features, as reshape can do)
            new_mean = rng.standard_normal(size=b) * 1e6
            diff = new_mean - exp.y.mean(axis=(1, 2))
            exp.y += diff[:, np.newaxis, np.newaxis]
            exp = exp.bootstrap_img(n=10, seed=0)
            assert np.allclose(new_mean, exp.y.mean(axis=(1, 2)))


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
    def test_init(self):
        shape = 10, 10
        num_img = 5
        b = 3
        exp = Experiment.from_gauss(shape=shape, seed=0, b=b, num_img=num_img)

        exp_scale = ExperimentScaled.from_exp(exp)

        assert np.allclose(exp_scale.y.mean(axis=(1, 2)), 0), 'non-zero mean'

        cov_after = np.cov(exp_scale.y.reshape((b, -1), order='F'))
        off_diag = ~np.eye(b).astype(bool)
        assert np.allclose(cov_after[off_diag], 0), 'non-zero correlation'

        assert np.allclose(exp.y, exp_scale.prep_inv(exp_scale.y))

        # ensure that each pca direction is transformed properly
        cov = np.cov(exp.y.reshape(b, -1))
        scale = np.diag(1 / np.diag(cov) ** .5)
        cov_scale = scale @ cov @ scale.T
        evals, evecs = np.linalg.eigh(cov_scale)

        for evec, e in zip(evecs.T, np.eye(b)):
            e = e[:, np.newaxis, np.newaxis]
            e_preimage = np.diag(1 / np.diag(scale)) @ evec
            e_preimage = (e_preimage[:, np.newaxis, np.newaxis] +
                          exp_scale.mean_orig)
            assert np.allclose(exp_scale.prep(e_preimage), e)


class TestImposeEffectWithNoise:
    """test impose_effect with noise_scale parameter"""
    
    def test_impose_effect_with_noise(self):
        """test that noise_scale > 0 code path executes """
        seed = 0
        exp = Experiment.from_gauss(seed=seed, shape=(5, 5), num_img=20)
        extenter = glow.effect.ExtenterSphere(radius=2)
        
        # impose effect with noise (should not raise error)
        exp_with_noise, effect_with_noise = exp.impose_effect(
            seed=seed,
            extenter=extenter,
            hotel_tr=2.0,
            noise_scale=0.5  # triggers lines 171-175
        )
        
        # should complete successfully
        assert effect_with_noise.mask.sum() > 0
        assert effect_with_noise.hotel_tr > 0


class TestExperimentScaledZeroVariance:
    """ExperimentScaled should raise on zero-variance features"""

    def test_zero_variance_raises(self):
        import pytest
        exp = Experiment.from_gauss(a=2, b=2, shape=(5, 5), num_img=20, seed=0)
        # zero out one feature entirely → zero variance
        exp.y[1, :, :] = 0.0

        with pytest.raises(ValueError, match='zero-variance'):
            ExperimentScaled.from_exp(exp)
