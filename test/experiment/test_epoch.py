from hrba.experiment import *
from hrba.experiment.epoch import *
from hrba.graph import get_f1
from hrba.sample_effect import ExtenterSphere
from .make_test_image import folder_test_data
from .test_exper import get_rand_exp
from ..helper import generate_dummy_data

num_vox = 4
n_permute = 3
x, y, contrast = generate_dummy_data(b=1, reg_size=num_vox, seed=0)
mask_idx = np.arange(num_vox).reshape((2, 2))
exp = Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx)
children = np.arange(num_vox * 2 - 2).reshape((-1, 2))
child_dict = {idx: children for idx in range(n_permute)}


class TestEpoch:
    def test_get_pval(self):
        z_stat = np.array([[7, 3, 1, 0],
                           [0, 0, 0, 0],
                           [3, 3, 3, 3],
                           [2, 2, 2, 5]])
        p_val_exp = np.array([1, 3, 3, 4]) / 4

        p_val = Epoch.get_pval(stat=z_stat)
        assert np.allclose(p_val, p_val_exp)

        mask_exclude = np.array([[1, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 1]]).astype(bool)
        p_val_exp = np.array([np.nan, 2, 3, 4]) / 4
        p_val = Epoch.get_pval(stat=z_stat, mask_exclude=mask_exclude)
        assert np.allclose(p_val, p_val_exp, equal_nan=True)


class TestEpochHRBA:
    def test_cluster(self):
        # in a population of identical test images, clustering should
        # segment based on color

        # load single image, bootstrap a few more (no noise), sample rand x
        exp = ExperimentImageOnly.from_search(folder=folder_test_data,
                                              sbj_regex='sbj\d',
                                              img_glob_dict={
                                                  'color': '*test.png'})
        exp.bootstrap_img(n=10, noise_scale=0)
        exp = exp.sample_x(a=2)

        # cluster (should collect all areas of consistent color)
        children = EpochHRBA.cluster(exp=exp)

        # assumptions: test image has 1 color per greyscale value and each
        # color is contiguous
        y_sbj0_grey = np.linalg.norm(exp.y[:, 0, :], axis=0)
        img_grey = np.zeros(exp.mask_idx.shape)
        img_grey[exp.mask_idx >= 0] = y_sbj0_grey
        for grey_val in set(y_sbj0_grey):
            mask = img_grey == grey_val
            f1 = get_f1(mask=mask, mask_idx=exp.mask_idx, children=children)
            assert np.isclose(max(f1), 1)

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


class TestBigEffect:
    """ given strong effect, discover it"""
    # build experiment with strong effect to be found (whole region)
    num_img = 100
    shape = 5, 5, 5
    a = 2
    b = 1
    exp = get_rand_exp(shape=shape, a=a, b=b, seed=0, num_img=num_img)
    exp, effect = exp.impose_effect(seed=0, extenter=ExtenterSphere(radius=1),
                                    p_val=.0001)

    def test_hrba(self):
        epoch = EpochHRBA(TestBigEffect.exp, n_permute=100)
        np.testing.assert_allclose(epoch.effect_list[0].mask,
                                   TestBigEffect.effect.mask)

    def test_tfce(self):
        epoch = EpochTFCE(TestBigEffect.exp, n_permute=100)
        np.testing.assert_allclose(epoch.effect_list[0].mask,
                                   TestBigEffect.effect.mask)