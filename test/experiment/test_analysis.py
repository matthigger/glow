import copy

from glow.effect import ExtenterSphere
from glow.experiment import *
from glow.experiment.analysis import *
from glow.graph import get_f1_sens_spec, iter_topo
from glow.mask import get_mask_idx


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

        # true effect should be among the discovered effects
        assert any(np.array_equal(eff.mask, TestBigEffect.effect.mask)
                   for eff in analysis.effect_list), \
            'true effect not found among discovered effects'

    def test_vba(self):
        kwargs_list = [dict(tfce_flag=False),
                       dict(tfce_flag=True)]
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

    def test_glow_node(self):
        """node pruning with exp_eff=1 should find exactly one effect."""
        analysis = AnalysisGLOW(
            TestBigEffect.exp,
            n_perm=25,
            alpha_fwer=.1,
            prune_method='node',
            prune_geom_exp_eff=1
        )

        # dp_info should be populated with correct lambda
        assert hasattr(analysis, 'dp_info')
        assert np.isclose(analysis.dp_info['lam'], np.log(2))

        # should discover exactly one effect
        assert len(analysis.effect_list) == 1, \
            f'expected 1 effect, got {len(analysis.effect_list)}'

        # that one effect should match the true effect
        assert np.array_equal(analysis.effect_list[0].mask,
                              TestBigEffect.effect.mask), \
            'node effect does not match the true effect'
    
    def test_glow_with_adjustment(self):
        """test GLOW with adjustment permutations"""
        analysis = AnalysisGLOW(
            TestBigEffect.exp, 
            n_perm=10,
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


class TestParallelExecution:
    """test parallel execution paths"""
    
    def test_glow_parallel_permutations(self):
        """test parallel permutation execution in AnalysisGLOW"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp, _ = exp.impose_effect(seed=0,
                                   extenter=ExtenterSphere(radius=1),
                                   hotel_tr=1.5)
        
        # run with parallel execution
        analysis_parallel = AnalysisGLOW(
            exp,
            n_perm=10,
            alpha_fwer=.1,
            n_jobs_perm=2  # parallel execution
        )
        
        # run with serial execution
        analysis_serial = AnalysisGLOW(
            exp,
            n_perm=10,
            alpha_fwer=.1,
            n_jobs_perm=0  # serial execution
        )
        
        # results should be identical
        assert np.allclose(analysis_parallel.pval, analysis_serial.pval)
        assert len(analysis_parallel.effect_list) == len(analysis_serial.effect_list)
    


class TestPvalFloor:
    """test that permutation p-values are floored at 1/num_perm"""

    def test_pval_never_zero(self):
        """observed stat equal to permutation max should not yield pval=0"""
        # row 0 is unpermuted (observed), rows 1-4 are permutations
        # region 0 has the global max in the observed row
        z_stat = np.array([[10.0, 0.0],
                           [1.0, 0.0],
                           [2.0, 0.0],
                           [3.0, 0.0],
                           [4.0, 0.0]])
        pval = Analysis.get_pval(stat=z_stat)

        # p-value for region 0 must be >= 1/num_perm, never zero
        num_perm = z_stat.shape[0]
        assert pval[0] >= 1 / num_perm, \
            f'pval should be >= {1/num_perm}, got {pval[0]}'
        assert pval[0] > 0, f'pval must never be exactly zero, got {pval[0]}'


class TestZeroStdGuard:
    """test z-normalization when adjustment permutations have zero variance"""

    def test_constant_stat_region(self):
        """regions with constant stat across adjustment perms should not produce inf/nan"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp, _ = exp.impose_effect(seed=0,
                                   extenter=ExtenterSphere(radius=1),
                                   hotel_tr=1.5)

        analysis = AnalysisGLOW(exp, n_perm=5, alpha_fwer=0.05,
                                min_size=1)

        assert not np.any(np.isinf(analysis.llr_adjusted)), \
            'llr_adjusted contains inf (likely zero-std division)'
        assert not np.any(np.isnan(analysis.llr_adjusted)), \
            'llr_adjusted contains nan (likely zero-std division)'


class TestNaNHandling:
    """test handling of NaN statistics"""
    
    def test_get_pval_with_nan_stats(self):
        """test get_pval handles NaN stats correctly"""
        # create stat array with some NaN values
        z_stat = np.array([[7.0, np.nan, 3.0, 1.0],
                           [5.0, np.nan, 2.0, 0.0],
                           [6.0, np.nan, 4.0, 2.0]])
        
        pval = Analysis.get_pval(stat=z_stat)
        
        # second region should have NaN pval
        assert np.isnan(pval[1])
        
        # other regions should have valid pvals
        assert not np.isnan(pval[0])
        assert not np.isnan(pval[2])
        assert not np.isnan(pval[3])


# ---------------------------------------------------------------------------
# In-memory checkpoint helper for testing (no S3 dependency)
# ---------------------------------------------------------------------------

class MemoryCheckpoint:
    """Minimal checkpoint implementation backed by an in-memory dict."""

    def __init__(self, state=None):
        self._state = state
        self.save_count = 0
        self.delete_called = False

    def load(self):
        return copy.deepcopy(self._state)

    def save(self, child_dict, stat, perm_idx):
        self._state = {
            'child_dict': copy.deepcopy(child_dict),
            'stat': stat.copy(),
            'last_perm_idx': perm_idx,
        }
        self.save_count += 1

    def delete(self):
        self.delete_called = True
        self._state = None


class TestCheckpoint:
    """Test AnalysisGLOW checkpoint / resume behaviour."""

    # shared small experiment
    exp = Experiment.from_gauss(a=2, b=1, shape=(3, 3), num_img=20, seed=0)

    def test_no_checkpoint_by_default(self):
        """AnalysisGLOW works when checkpoint is None (the default)."""
        analysis = AnalysisGLOW(self.exp, n_perm=5, alpha_fwer=.5)
        assert hasattr(analysis, 'pval')
        assert len(analysis.child_dict) == 6  # 0..5

    def test_checkpoint_save_called(self):
        """save() is called during the permutation loop."""
        # use n_perm high enough to trigger at least one save
        # _CHECKPOINT_INTERVAL is 25, so n_perm=50 gives 51 permutations
        ckpt = MemoryCheckpoint()
        analysis = AnalysisGLOW(
            self.exp, n_perm=50, alpha_fwer=.5, checkpoint=ckpt)

        assert ckpt.save_count >= 1, 'checkpoint.save() was never called'
        assert ckpt.delete_called, 'checkpoint.delete() should be called on completion'

    def test_checkpoint_delete_on_completion(self):
        """delete() is called after all permutations finish."""
        ckpt = MemoryCheckpoint()
        AnalysisGLOW(self.exp, n_perm=5, alpha_fwer=.5, checkpoint=ckpt)
        assert ckpt.delete_called

    def test_checkpoint_resume(self):
        """Resuming from a partial checkpoint produces the same result."""
        n_perm = 10

        # full run (no checkpoint) as reference
        ref = AnalysisGLOW(self.exp, n_perm=n_perm, alpha_fwer=.5)

        # build partial checkpoint from the first 5 permutations
        partial_child_dict = {i: ref.child_dict[i] for i in range(5)}
        b, num_img, num_vox = self.exp.y.shape
        num_reg = num_vox + ref.child_dict[0].shape[0]
        partial_stat = np.full((n_perm + 1, num_reg), fill_value=-1.0)
        for i in range(5):
            partial_stat[i, :] = ref.stat[i, :]

        resume_state = {
            'child_dict': partial_child_dict,
            'stat': partial_stat,
            'last_perm_idx': 4,
        }

        ckpt = MemoryCheckpoint(state=resume_state)
        resumed = AnalysisGLOW(
            self.exp, n_perm=n_perm, alpha_fwer=.5, checkpoint=ckpt)

        # child_dict should be complete
        assert set(resumed.child_dict.keys()) == set(range(n_perm + 1))

        # stat arrays should match
        np.testing.assert_array_equal(resumed.stat, ref.stat)

        # final p-values should match
        np.testing.assert_array_equal(resumed.pval, ref.pval)


class TestForest:
    """AnalysisGLOW on a non-contiguous mask (forest of 2 trees)."""

    def test_forest_completes(self):
        # 2D mask: two disconnected 5x3 blobs with a gap
        mask = np.zeros((5, 9), dtype=bool)
        mask[:, :3] = True
        mask[:, 6:] = True
        num_vox = int(mask.sum())
        mask_idx = get_mask_idx(mask)

        rng = np.random.default_rng(seed=42)
        b, num_img = 1, 20
        y = rng.standard_normal((b, num_img, num_vox))
        x = rng.standard_normal((1, num_img))
        contrast = np.array([True])

        exp = Experiment(x=x, contrast=contrast, y=y,
                         mask_idx=mask_idx, add_bias=True)
        ana = AnalysisGLOW(exp, n_perm=10, alpha_fwer=.5)

        children = ana.child_dict[0]
        assert children.shape == (num_vox - 2, 2), \
            f'expected {num_vox - 2} internal nodes, got {children.shape[0]}'

        all_nodes = list(iter_topo(children=children, num_leaf=num_vox))
        expected_total = num_vox + children.shape[0]
        assert len(all_nodes) == expected_total, \
            f'iter_topo yielded {len(all_nodes)}, expected {expected_total}'
