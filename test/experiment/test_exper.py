from hglm.effect import ExtenterSphere
from hglm.experiment import Permuter
from hglm.experiment.exper import *
from test.helper import generate_dummy_data
from .make_test_image import folder_test_data, img_feat_intensity


def get_rand_exp(shape=(10, 10, 10), **kwargs):
    reg_size = np.prod(shape)
    mask_idx = np.arange(reg_size).reshape(shape)

    x, y, contrast = generate_dummy_data(num_vox=reg_size, **kwargs)
    return Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx)


class TestExperimentOnlyImage:
    def test_from_search(self):
        # nii
        exp0 = ExperimentImageOnly.from_search(folder=folder_test_data,
                                               sbj_regex='img\d',
                                               img_glob_dict={
                                                   'feat0': '*feat0.nii.gz',
                                                   'feat1': '*feat1.nii.gz'})
        # jpg
        exp1 = ExperimentImageOnly.from_search(folder=folder_test_data,
                                               sbj_regex='img\d',
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
        exp = get_rand_exp(seed=seed)
        extenter = ExtenterSphere(radius=3)

        for p_val in np.logspace(-3, -.0001, 4):
            _exp, effect = exp.impose_effect(seed=seed, extenter=extenter,
                                             p_val=p_val)

            assert np.isclose(effect.p_val, p_val)

    def test_sample_x(self):
        seed = 0
        exp = get_rand_exp(seed=seed)
        exp.sample_x(a=exp.x.shape[0])
        exp.sample_x(contrast=exp.contrast)
        exp.sample_x(a=4)

    def test_bootstrap_img(self):
        n = 100
        rng = np.random.default_rng(seed=0)
        for str_test_glob in ('*test_bw.png', '*test.png'):
            exp = ExperimentImageOnly.from_search(folder=folder_test_data,
                                                  sbj_regex='squares',
                                                  img_glob_dict={
                                                      'feat': str_test_glob})

            exp.bootstrap_img(n=n, seed=0)
            b, num_img, num_vox = exp.y.shape
            assert num_img == n

            # validate reshaping (ensure cov compute is correct and doesn't
            # mix features, as reshape can do)
            new_mean = rng.standard_normal(size=b) * 1e6
            diff = new_mean - exp.y.mean(axis=(1, 2))
            exp.y += diff[:, np.newaxis, np.newaxis]
            exp.bootstrap_img(n=10, seed=0)
            assert np.allclose(new_mean, exp.y.mean(axis=(1, 2)))

    def test_add_effect(self):
        # get arbitrary experiment & effect region (whole image)
        shape = 10, 10
        mask = np.ones(shape, dtype=bool)
        exp = get_rand_exp(shape=shape, add_effect=True, seed=0)
        eff = Effect.from_exp_mask(exp=exp, mask=mask)

        # remove the effect
        exp2 = exp.rm_effect(effect=eff)

        # validate that exp2 has f=0 for whole region
        eff2 = Effect.from_exp_mask(exp=exp2, mask=mask)
        assert np.isclose(eff2.f_stat, 0)


class TestExperiment:

    def test_permute(self):
        perm_idx = 1
        shape = 10, 10
        num_img = 5
        exp = get_rand_exp(shape=shape, seed=0, num_img=num_img)

        # prep
        x = exp.x[~exp.contrast, :], exp.x
        h = [np.linalg.pinv(_x) @ _x for _x in x]

        # test 1: block_exchange=True
        # manually permute in a loop, check that einsum does the same
        exp_permuted = exp.permute(perm_idx=perm_idx, block_exchange=True)

        perm = Permuter(x=exp.x[~exp.contrast, :])
        freed_lane = perm.get_freed_lane(perm_idx)
        for vox_idx in range(np.prod(shape)):
            y_permute_exp = exp.y[..., vox_idx] @ freed_lane
            assert np.allclose(exp_permuted.y[..., vox_idx], y_permute_exp)

        # test 2: block_exchange=False
        exp_permuted = exp.permute(perm_idx=perm_idx, block_exchange=False)

        for vox_idx in range(np.prod(shape)):
            freed_lane = perm.get_freed_lane(perm_idx + vox_idx)
            y_permute_exp = exp.y[..., vox_idx] @ freed_lane

            assert np.allclose(exp_permuted.y[..., vox_idx], y_permute_exp)


class TestExperimentWhitened:
    def test_init(self):
        shape = 10, 10
        num_img = 5
        exp = get_rand_exp(shape=shape, seed=0, num_img=num_img)
        exp_white = ExperimentWhitened.from_exp(exp)

        # confirm desired covariance in exp_white
        b = exp_white.y.shape[0]
        y = exp_white.y.reshape((b, -1), order='F')
        assert np.allclose(np.cov(y), np.eye(b))
