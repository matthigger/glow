from shutil import rmtree

from hrba.experiment.epoch import *
from hrba.experiment.exper import Experiment
from hrba.graph import get_miss_hits
from .make_test_image import folder_test_data
from ..helper import generate_dummy_data

num_vox = 4
n_permute = 3
x, y, contrast = generate_dummy_data(b=1, reg_size=num_vox, seed=0)
mask_idx = np.arange(num_vox).reshape((2, 2))
exp = Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx)
children = np.arange(num_vox * 2 - 2).reshape((-1, 2))
child_dict = {idx: children for idx in range(n_permute)}


class TestEpoch:
    def test_to_nii(self):
        epoch = EpochHRBA(exp=exp, n_permute=0)
        folder = epoch.to_nii()
        rmtree(folder)

    def test_get_pval(self):
        z_stat = np.array([[7, 3, 1, 0],
                           [0, 0, 0, 0],
                           [3, 3, 3, 3],
                           [5, 5, 5, 5]])
        p_val_exp = np.array([1, 3, 3, 4]) / 4

        p_val = Epoch.get_pval(z_stat)
        assert np.allclose(p_val, p_val_exp)

    def test_get_f_stat(self):
        Epoch.get_llr(exp=exp, child_dict=child_dict)
        Epoch.get_llr(exp=exp, n_permute=n_permute)


class TestEpochHRBA:
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
        children = EpochHRBA.cluster(exp=exp, n_permute=0)[0]

        # assumptions: test image has 1 color per greyscale value and each
        # color is contiguous
        y_sbj0_grey = np.linalg.norm(exp.y[:, 0, :], axis=0)
        img_grey = np.zeros(exp.mask_idx.shape)
        img_grey[exp.mask_idx >= 0] = y_sbj0_grey
        for grey_val in set(y_sbj0_grey):
            mask = img_grey == grey_val
            miss_hit_dict = get_miss_hits(mask=mask, mask_idx=exp.mask_idx,
                                          children=children)
            perfect_miss_hit = (0, mask.sum())
            for miss_hit in miss_hit_dict.values():
                if perfect_miss_hit == tuple(miss_hit):
                    break
            else:
                raise AssertionError('some color not segmented perfectly')

    def test_discover(self):
        num_vox = 4
        x, y, contrast = generate_dummy_data(b=1, reg_size=num_vox, seed=0)
        mask_idx = np.arange(num_vox)
        exp = Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx)
        children = np.array([[0, 1],
                             [2, 3],
                             [4, 5]])

        # only 1 sig region
        pval = np.array([.1, 1, 1, 1, 1, 1, 1])
        eff_list = EpochHRBA.discover(pval=pval, stat=-pval, children=children, \
                                      exp=exp, alpha=.5)
        assert len(eff_list) == 1
        assert eff_list[0].reg_idx == 0

        # 2 sig regions which intersect
        pval = np.array([.1, 1, 1, 1, .2, 1, 1])
        eff_list = EpochHRBA.discover(pval=pval, stat=-pval, children=children, \
                                      exp=exp, alpha=.5)
        assert len(eff_list) == 1
        assert eff_list[0].reg_idx == 0

        # 2 sig regions which don't intersect
        pval = np.array([.1, .2, 1, 1, 1, 1, 1])
        eff_list = EpochHRBA.discover(pval=pval, stat=-pval, children=children, \
                                      exp=exp, alpha=.5)
        assert len(eff_list) == 2
        assert eff_list[0].reg_idx == 0
        assert eff_list[1].reg_idx == 1

    def test_adjust_by_size(self):
        # "right" answer: log10 f stat = log10 size * 1 + error
        # where error has std_dev of 1
        n_size = 4
        n_perm = 3
        size = np.tile(np.arange(2, 2 + n_size), (n_perm, 1))

        # build noise to be zero mean and std dev 1
        rng = np.random.default_rng(seed=0)
        error = rng.standard_normal((n_perm, n_size))
        for idx in range(n_size):
            error[:, idx] -= error[:, idx].mean()
            error[:, idx] *= 1 / error[:, idx].std()

        f_stat = 10 ** (np.log10(size) + error)

        f_stat_adjust, lin_reg = EpochHRBA.adjust_by_size(size=size,
                                                          stat=f_stat,
                                                          ignore_row0=False)

        # model: log10 f = log10 size + eps
        assert np.isclose(lin_reg.coef_, 1)
        assert np.isclose(lin_reg.intercept_, 0)

        # check z stat compute
        z_stat_exp = np.log10(f_stat) - np.log10(size)
        z_stat_exp /= np.std(z_stat_exp)
        assert np.allclose(f_stat_adjust, z_stat_exp)
        assert np.isclose(np.std(f_stat_adjust), 1)
