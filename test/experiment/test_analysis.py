from glow.effect import ExtenterSphere
from glow.experiment import *
from glow.experiment.analysis import *
from glow.graph import get_f1_sens_spec


class TestAnalysis:
    def test_get_pval(self):
        z_stat = np.array([[7, 3, 1, 0],
                           [0, 0, 0, 0],
                           [3, 3, 3, 3],
                           [2, 2, 2, 5]])
        pval_exp = np.array([1, 3, 3, 4]) / 4

        pval = Analysis.get_pval(stat=z_stat)
        assert np.allclose(pval, pval_exp)


class TestBigEffect:
    """ given strong effect, discover it"""
    # build experiment with strong effect to be found
    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=100, seed=0)
    exp, effect = exp.impose_effect(seed=0,
                                    extenter=ExtenterSphere(radius=2),
                                    hotel_tr=2)

    def test_glow(self):
        analysis = AnalysisGLOW(TestBigEffect.exp, n_perm=25, alpha_fwer=.1)

        # check that target region segmented properly
        f1 = get_f1_sens_spec(mask=TestBigEffect.effect.mask,
                              mask_idx=analysis.exp.mask_idx,
                              children=analysis.child_dict[0])[0]
        assert np.isclose(f1.max(), 1), 'target region not segmented'

        # appropriate effect discovered as most significant effect
        np.testing.assert_allclose(analysis.effect_list[0].mask,
                                   TestBigEffect.effect.mask)

    def test_tfce(self):
        for tfce_flag in (0, 1):
            analysis = AnalysisVBA(TestBigEffect.exp, n_perm=25,
                                   alpha_fwer=.1,
                                   tfce_flag=tfce_flag)
            mask_all = sum(eff.mask for eff in analysis.effect_list)
            np.testing.assert_allclose(mask_all,
                                       TestBigEffect.effect.mask)
