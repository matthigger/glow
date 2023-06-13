from hrba.experiment import *
from hrba.graph import get_miss_hits
from .make_test_image import folder_test_data


class TestEpoch:
    def test_cluster(self):
        # in a population of identical test images, clustering should
        # segment based on color

        # load single image, bootstrap a few more (no noise), sample rand x
        exp = Experiment.from_search(folder=folder_test_data,
                                     sbj_regex='sbj\d',
                                     img_glob_dict={'color': '*test.png'})
        exp.bootstrap_img(n=10, noise_scale=0)
        exp.sample_x(a=2)

        # cluster (should collect all areas of consistent color)
        dendro = Epoch.cluster(exp=exp, n_permute=0)[0]

        # assumptions: test image has 1 color per greyscale value and each
        # color is contiguous
        y_sbj0_grey = np.linalg.norm(exp.y[:, 0, :], axis=0)
        img_grey = np.zeros(exp.mask_idx.shape)
        img_grey[exp.mask_idx >= 0] = y_sbj0_grey
        for grey_val in set(y_sbj0_grey):
            mask = img_grey == grey_val
            miss_hit_dict = get_miss_hits(mask=mask, mask_idx=exp.mask_idx,
                                          dendro=dendro)
            perfect_miss_hit = (0, mask.sum())
            for miss_hit in miss_hit_dict.values():
                if perfect_miss_hit == tuple(miss_hit):
                    break
            else:
                raise AssertionError('some color not segmented perfectly')

    def test_model_adjust_f(self):
        # "right" answer: log10 f stat = log10 size * 1 + error
        # where error has std_dev equal to region size
        n_size = 4
        n_perm = 3
        size = np.tile(np.arange(2, 2 + n_size), (n_perm, 1))

        # build noise to be zero mean and scale with region size
        rng = np.random.default_rng(seed=0)
        error = rng.standard_normal((n_perm, n_size))
        for col_idx, _size in enumerate(size[0, :]):
            # zero mean
            error[:, col_idx] -= error[:, col_idx].mean()

            # scale to var = log10(_size)
            std = error[:, col_idx].std(ddof=0)
            error[:, col_idx] *= np.log10(_size) ** .5 / std

        f_stat = 10 ** (np.log10(size) + error)

        model_f_mu, model_f_var, z_stat = Epoch.model_adjust_f(size=size,
                                                               f_stat=f_stat)

        # model: log10 f = log10 size + eps
        assert np.isclose(model_f_mu.coef_, 1)
        assert np.isclose(model_f_mu.intercept_, 0)

        # model: variance of eps = log10 size
        assert np.isclose(model_f_var.coef_, 1)
        assert np.isclose(model_f_var.intercept_, 0)

        # check z stat compute
        z_stat_exp = (f_stat - np.log10(size)) / np.log10(size) ** .5
        assert np.allclose(z_stat, z_stat_exp)

    def test_get_pval(self):
        z_stat = np.array([[7, 3, 1, 0],
                           [0, 0, 0, 0],
                           [3, 3, 3, 3],
                           [5, 5, 5, 5]])
        p_val_exp = np.array([1, 3, 3, 4]) / 4

        p_val = Epoch.get_pval(z_stat)
        assert np.allclose(p_val, p_val_exp)
