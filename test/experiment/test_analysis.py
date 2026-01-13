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

    def test_vba(self):
        conn = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]])
        kwargs_list = [dict(tfce_flag=False, cet_flag=False),
                       dict(tfce_flag=True, cet_flag=False),
                       dict(tfce_flag=False, cet_flag=True, conn=conn,
                            mask_eff=TestBigEffect.effect.mask)]
        for kwargs in kwargs_list:
            analysis = AnalysisVBA(TestBigEffect.exp, n_perm=25,
                                   alpha_fwer=.1, **kwargs)
        mask_all = sum(eff.mask for eff in analysis.effect_list)
        np.testing.assert_allclose(mask_all,
                                   TestBigEffect.effect.mask)
    
    def test_glow_with_prune(self):
        """test GLOW with pruning enabled"""
        analysis = AnalysisGLOW(
            TestBigEffect.exp, 
            n_perm=10, 
            n_perm_prune=15,
            alpha_fwer=.1,
            alpha_prune=.05
        )
        
        # should still find the effect
        assert len(analysis.effect_list) > 0
    
    def test_glow_with_adjustment(self):
        """test GLOW with adjustment permutations"""
        analysis = AnalysisGLOW(
            TestBigEffect.exp, 
            n_perm=10,
            n_perm_adj=15,
            alpha_fwer=.1
        )
        
        # should have both regular and adjustment permutations
        assert len(analysis.child_dict) >= 10
    
    def test_analysis_get_stat(self):
        """test custom get_stat function"""
        from glow.experiment.mancova import get_pillai
        
        # use pillai instead of default hotelling
        analysis = AnalysisGLOW(
            TestBigEffect.exp,
            n_perm=5,
            alpha_fwer=.1,
            get_stat=get_pillai
        )
        
        # should still work
        assert hasattr(analysis, 'effect_list')
        assert hasattr(analysis, 'stat')


class TestAnalysisEdgeCases:
    """test edge cases and error handling"""
    
    def test_all_regions_too_small(self):
        """test when all regions are filtered out by min_size"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp, _ = exp.impose_effect(seed=0,
                                   extenter=ExtenterSphere(radius=1),
                                   hotel_tr=1.5)
        
        # set min_size so large that all regions are filtered
        analysis = AnalysisGLOW(
            exp,
            n_perm=5,
            alpha_fwer=.1,
            min_size=1000000  # impossibly large
        )
        
        # should run without error
        assert hasattr(analysis, 'pval')
        assert hasattr(analysis, 'effect_list')
        
        # all pvals should be nan
        assert np.all(np.isnan(analysis.pval))
        
        # no effects should be found
        assert len(analysis.effect_list) == 0
    
    def test_small_experiment(self):
        """test with minimal experiment size"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(3, 3), num_img=10, seed=0)
        
        analysis = AnalysisGLOW(
            exp,
            n_perm=3,
            alpha_fwer=.5,  # lenient for small sample
            min_size=1
        )
        
        # should run without error even with small size
        assert hasattr(analysis, 'pval')
        assert analysis.pval.shape[0] > 0
    
    def test_different_alpha_values(self):
        """test with different alpha thresholds"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp, _ = exp.impose_effect(seed=0,
                                   extenter=ExtenterSphere(radius=1),
                                   hotel_tr=1.5)
        
        # strict alpha
        analysis_strict = AnalysisGLOW(
            exp,
            n_perm=5,
            alpha_fwer=.01
        )
        
        # lenient alpha
        analysis_lenient = AnalysisGLOW(
            exp,
            n_perm=5,
            alpha_fwer=.5
        )
        
        # lenient should find same or more effects
        assert len(analysis_lenient.effect_list) >= len(analysis_strict.effect_list)