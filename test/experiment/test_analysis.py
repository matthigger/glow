import pytest
from scipy import ndimage

from hglm.effect import ExtenterSphere
from hglm.experiment import *
from hglm.experiment.analysis import *
from hglm.graph import get_f1
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


class TestAnalysis:
    def test_get_pval(self):
        z_stat = np.array([[7, 3, 1, 0],
                           [0, 0, 0, 0],
                           [3, 3, 3, 3],
                           [2, 2, 2, 5]])
        p_val_exp = np.array([1, 3, 3, 4]) / 4

        p_val = Analysis.get_pval(stat=z_stat)
        assert np.allclose(p_val, p_val_exp)

        mask_exclude = np.array([[1, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 1]]).astype(bool)
        p_val_exp = np.array([np.nan, 2, 3, 4]) / 4
        p_val = Analysis.get_pval(stat=z_stat, mask_exclude=mask_exclude)
        assert np.allclose(p_val, p_val_exp, equal_nan=True)


class TestAnalysishglm:
    def test_cluster(self):
        # in a population of identical test images, clustering should
        # segment based on color

        # load single image, bootstrap a few more (no noise), sample rand x
        exp = ExperimentImageOnly.from_search(folder=folder_test_data,
                                              sbj_regex='squares',
                                              img_glob_dict={
                                                  'color': '*test.png'})

        exp.bootstrap_img(n=10, noise_scale=0)
        exp = exp.sample_x(a=2)

        # cleave mask into many pieces (test case not supported)
        mask = exp.mask_idx > -1
        mask[:, mask.shape[1] // 2] = False
        mask[mask.shape[0] // 2, :] = False
        exp_cleave = exp.apply_mask(mask)

        with pytest.raises(NotImplementedError) as e:
            AnalysisHGLM.cluster(exp=exp_cleave)

        for _exp in (exp,):
            # cluster (should collect all areas of consistent color)
            children = AnalysisHGLM.cluster(exp=_exp)

            f1_list = list()
            # cast to uint8 to avoid floating point precision comparison error
            y0 = _exp.y[:, 0, :].astype(np.uint8)
            unique_colors = np.unique(np.unique(y0, axis=-1), axis=-1)
            for color in unique_colors.T:
                # build img_color, True at every voxel which given color
                color_mask = (y0.T == color).all(axis=-1)
                img_color = np.zeros(_exp.mask_idx.shape, dtype=bool)
                for idx in np.where(color_mask)[0]:
                    img_color[_exp.mask_idx == idx] = True

                # ensure that each contiguous region which is uniformly some
                # color shows up in the tree somewhere
                label, n_regions = ndimage.label(img_color)
                for idx in range(1, n_regions + 1):
                    mask = label == idx
                    f1 = get_f1(mask=mask, mask_idx=_exp.mask_idx,
                                children=children)
                    f1_list.append(max(f1))

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
        eff_list = AnalysisHGLM.discover(pval=pval, stat=-pval,
                                         children=children, exp=exp, alpha=.5)
        assert len(eff_list) == 1
        assert eff_list[0].reg_idx == 0

        # 2 sig regions which intersect
        pval = np.array([.1, 1, 1, 1, .2, 1, 1])
        eff_list = AnalysisHGLM.discover(pval=pval, stat=-pval,
                                         children=children, \
                                         exp=exp, alpha=.5)
        assert len(eff_list) == 1
        assert eff_list[0].reg_idx == 0

        # 2 sig regions which don't intersect
        pval = np.array([.1, .2, 1, 1, 1, 1, 1])
        eff_list = AnalysisHGLM.discover(pval=pval, stat=-pval,
                                         children=children, \
                                         exp=exp, alpha=.5)
        assert len(eff_list) == 2
        assert eff_list[0].reg_idx == 0
        assert eff_list[1].reg_idx == 1


class TestBigEffect:
    """ given strong effect, discover it"""
    # build experiment with strong effect to be found (whole region)
    num_img = 100
    shape = 5, 5
    a = 2
    b = 1
    exp = get_rand_exp(shape=shape, a=a, b=b, seed=0, num_img=num_img)
    exp, effect = exp.impose_effect(seed=0, extenter=ExtenterSphere(radius=1),
                                    p_val=.0001)

    def test_hglm(self):
        analysis = AnalysisHGLM(TestBigEffect.exp, n_perm=10, alpha=.1)

        # check that target region segmented properly
        f1 = get_f1(mask=TestBigEffect.effect.mask,
                    mask_idx=analysis.exp.mask_idx,
                    children=analysis.child_dict[0])
        assert np.isclose(f1.max(), 1), 'target region not segmented'

        # appropriate effect discovered as most significant effect
        np.testing.assert_allclose(analysis.effect_list[0].mask,
                                   TestBigEffect.effect.mask)

    def test_tfce(self):
        analysis = AnalysisTFCE(TestBigEffect.exp, n_perm=10, alpha=.1)
        np.testing.assert_allclose(analysis.effect_list[0].mask,
                                   TestBigEffect.effect.mask)
