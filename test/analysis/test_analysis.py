from glow.effect import ExtenterSphere, EffectSynthetic
from glow.experiment import *
from glow.analysis import *
from glow.graph import get_dice_sens_spec, iter_topo
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
    exp, effect = EffectSynthetic.impose(exp, seed=0,
                                    extenter=ExtenterSphere(radius=2),
                                    effect_llr=0.5)

    def test_glow(self):
        analysis = AnalysisGLOW(TestBigEffect.exp, n_perm_fwer=25, alpha_fwer=.1)

        # check that target region segmented properly
        dice = get_dice_sens_spec(mask=TestBigEffect.effect.mask,
                              mask_idx=analysis.exp.mask_idx,
                              children=analysis.children)[0]
        assert np.isclose(dice.max(), 1), 'target region not segmented'

        # should discover at least one effect overlapping the target
        assert len(analysis.effect_list) >= 1, \
            'no effects discovered'

    def test_vba(self):
        kwargs_list = [dict(tfce_flag=False),
                       dict(tfce_flag=True)]
        for kwargs in kwargs_list:
            analysis = AnalysisVBA(TestBigEffect.exp, n_perm_fwer=25,
                                   alpha_fwer=.1, **kwargs)
        mask_all = sum(eff.mask for eff in analysis.effect_list)
        np.testing.assert_allclose(mask_all,
                                   TestBigEffect.effect.mask)
    
    def test_vba_wilks(self):
        """VBA with Wilks' Lambda (1 - Wilks) should detect effects.

        get_wilks returns 1 - Lambda, so larger = more evidence against H0,
        compatible with max-stat and TFCE without sign correction.
        """
        from glow.analysis.mancova import get_wilks
        # non-TFCE
        ana = AnalysisVBA(TestBigEffect.exp, n_perm_fwer=25,
                          alpha_fwer=.5, tfce_flag=False,
                          get_stat=get_wilks)
        assert np.nanmin(ana.pval) <= 0.5, (
            f'Wilks VBA produced no small p-values '
            f'(min={np.nanmin(ana.pval):.3f})')

        # TFCE: 1-Wilks is non-negative, so TFCE works directly
        ana_tfce = AnalysisVBA(TestBigEffect.exp, n_perm_fwer=25,
                               alpha_fwer=.5, tfce_flag=True,
                               get_stat=get_wilks)
        assert np.nanmin(ana_tfce.pval) <= 0.5, (
            f'Wilks VBA-TFCE produced no small p-values '
            f'(min={np.nanmin(ana_tfce.pval):.3f})')

    def test_glow_with_prune(self):
        """test GLOW with default pruning (llr_z, lam=0)"""
        analysis = AnalysisGLOW(
            TestBigEffect.exp,
            n_perm_fwer=10,
            alpha_fwer=.1,
        )

        # should still find the effect
        assert len(analysis.effect_list) > 0

    def test_glow_node(self):
        """Greedy pruning should find the true effect."""
        analysis = AnalysisGLOW(
            TestBigEffect.exp,
            n_perm_fwer=25,
            alpha_fwer=.1,
        )

        assert hasattr(analysis, 'prune_info')
        assert len(analysis.effect_list) >= 1, \
            f'expected at least 1 effect, got {len(analysis.effect_list)}'
    
    def test_glow_with_adjustment(self):
        """test GLOW with adjustment permutations"""
        analysis = AnalysisGLOW(
            TestBigEffect.exp,
            n_perm_fwer=10,
            alpha_fwer=.1
        )
        
        assert hasattr(analysis, 'children')
        assert hasattr(analysis, 'pval')
    
    def test_analysis_get_stat(self):
        """test custom get_stat function"""
        from glow.analysis.mancova import get_pillai
        
        # use pillai instead of default hotelling
        analysis = AnalysisGLOW(
            TestBigEffect.exp,
            n_perm_fwer=5,
            alpha_fwer=.1,
            get_stat=get_pillai
        )
        
        # should still work
        assert hasattr(analysis, 'effect_list')
        assert hasattr(analysis, 'stat')


class TestZScoreStat:
    """z_score_stat lives on Analysis and is inherited by subclasses."""

    def test_shape_preserved(self):
        stat = np.random.default_rng(0).standard_normal((11, 50))
        z = Analysis.z_score_stat(stat)
        assert z.shape == stat.shape

    def test_all_rows_standardised(self):
        """All rows (observed + null) should have per-voxel mean ~0, std ~1."""
        stat = np.random.default_rng(0).standard_normal((51, 200))
        z = Analysis.z_score_stat(stat)
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(z.std(axis=0, ddof=1), 1, atol=1e-12)

    def test_constant_row_safe(self):
        """A constant row should not produce inf or nan."""
        stat = np.ones((5, 10))
        z = Analysis.z_score_stat(stat)
        assert not np.any(np.isinf(z))
        assert not np.any(np.isnan(z))

    def test_inherited_by_subclasses(self):
        """Subclasses should not override z_score_stat."""
        stat = np.random.default_rng(0).standard_normal((5, 20))
        np.testing.assert_array_equal(
            AnalysisVBA.z_score_stat(stat),
            Analysis.z_score_stat(stat))
        np.testing.assert_array_equal(
            AnalysisCET.z_score_stat(stat),
            Analysis.z_score_stat(stat))


class TestCET:
    def test_big_effect(self):
        """CET discovers the strong effect."""
        ana = AnalysisCET(TestBigEffect.exp, n_perm_fwer=25,
                          alpha_fwer=.1, cft_pval=0.01)
        assert len(ana.effect_list) >= 1

    def test_big_effect_z(self):
        """CET with z_flag runs without error and sets the flag."""
        ana = AnalysisCET(TestBigEffect.exp, n_perm_fwer=25,
                          alpha_fwer=.5, cft_pval=0.05, z_flag=True)
        assert ana.z_flag is True
        assert hasattr(ana, 'pval')
        assert hasattr(ana, 'effect_list')

    def test_null_no_discoveries(self):
        """Under the null (no effect), CET should not discover at alpha=0.05."""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=100, seed=0)
        ana = AnalysisCET(exp, n_perm_fwer=25, alpha_fwer=.05)
        assert len(ana.effect_list) == 0

    def test_pval_bounds(self):
        """p-values in [1/n_perm, 1]."""
        n = 25
        ana = AnalysisCET(TestBigEffect.exp, n_perm_fwer=n, alpha_fwer=.1)
        assert (ana.pval >= 1 / n).all()
        assert (ana.pval <= 1.0).all()

    def test_cluster_members_share_pval(self):
        """All voxels in a discovered cluster should have the same p-value."""
        ana = AnalysisCET(TestBigEffect.exp, n_perm_fwer=25,
                          alpha_fwer=.5, cft_pval=0.01)
        for eff in ana.effect_list:
            vox_idx = ana.exp.mask_idx[eff.mask]
            pvals = ana.pval[vox_idx]
            assert np.all(pvals == pvals[0])


class TestAnalysisEdgeCases:
    """test edge cases and error handling"""
    
    def test_all_regions_too_small(self):
        """test when all regions are filtered out by min_vox"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp, _ = EffectSynthetic.impose(exp, seed=0,
                                   extenter=ExtenterSphere(radius=1),
                                   effect_llr=0.5)

        # set min_vox so large that all regions are filtered
        analysis = AnalysisGLOW(
            exp,
            n_perm_fwer=5,
            alpha_fwer=.1,
            min_vox=1000000  # impossibly large
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
            n_perm_fwer=3,
            alpha_fwer=.5,  # lenient for small sample
            min_vox=1
        )
        
        # should run without error even with small size
        assert hasattr(analysis, 'pval')
        assert analysis.pval.shape[0] > 0
    
    def test_different_alpha_values(self):
        """test with different alpha thresholds"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp, _ = EffectSynthetic.impose(exp, seed=0,
                                   extenter=ExtenterSphere(radius=1),
                                   effect_llr=0.5)
        
        # strict alpha
        analysis_strict = AnalysisGLOW(
            exp,
            n_perm_fwer=5,
            alpha_fwer=.01
        )

        # lenient alpha
        analysis_lenient = AnalysisGLOW(
            exp,
            n_perm_fwer=5,
            alpha_fwer=.5
        )
        
        # lenient should find same or more effects
        assert len(analysis_lenient.effect_list) >= len(analysis_strict.effect_list)


class TestParallelExecution:
    """test parallel execution paths"""
    
    def test_glow_parallel_permutations(self):
        """test parallel permutation execution in AnalysisGLOW"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp, _ = EffectSynthetic.impose(exp, seed=0,
                                   extenter=ExtenterSphere(radius=1),
                                   effect_llr=0.5)
        
        # run with parallel execution
        analysis_parallel = AnalysisGLOW(
            exp,
            n_perm_fwer=10,
            alpha_fwer=.1,
            n_jobs_perm=2  # parallel execution
        )

        # run with serial execution
        analysis_serial = AnalysisGLOW(
            exp,
            n_perm_fwer=10,
            alpha_fwer=.1,
            n_jobs_perm=0  # serial execution
        )
        
        # results should be identical (bit-equal; NaN ↔ NaN allowed)
        np.testing.assert_array_equal(analysis_parallel.pval,
                                       analysis_serial.pval)
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
        exp, _ = EffectSynthetic.impose(exp, seed=0,
                                   extenter=ExtenterSphere(radius=1),
                                   effect_llr=0.5)

        analysis = AnalysisGLOW(exp, n_perm_fwer=5, alpha_fwer=0.05,
                                min_vox=1)

        assert not np.any(np.isinf(analysis.llr_z_0)), \
            'llr_z_0 contains inf (likely zero-std division)'
        assert not np.any(np.isnan(analysis.llr_z_0)), \
            'llr_z_0 contains nan (likely zero-std division)'


class TestMinVox:
    """min_vox gates the FWER comparison set."""

    def test_no_significant_region_below_min_vox(self):
        """No significant region in effect_list should have size < min_vox."""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                     num_img=50, seed=0)
        exp, _ = EffectSynthetic.impose(exp, seed=0,
                                    extenter=ExtenterSphere(radius=2),
                                    effect_llr=0.5)

        min_vox = 4
        ana = AnalysisGLOW(
            exp, n_perm_fwer=10, n_perm_inner=20,
            alpha_fwer=.5, min_vox=min_vox)

        # any sig_reg_list entry must have size >= min_vox
        for reg_idx in ana.sig_reg_list:
            assert ana.size[reg_idx] >= min_vox, (
                f'reg {reg_idx} has size {ana.size[reg_idx]} '
                f'< min_vox={min_vox} but appears in sig_reg_list')

        # regions with size < min_vox must have nan p-values
        below = ana.size < min_vox
        assert np.all(np.isnan(ana.pval[below])), \
            'regions below min_vox should have NaN p-values'

    def test_min_vox_affects_threshold(self):
        """Increasing min_vox should monotonically not raise the threshold.

        With more small-region noise excluded from the max-z null,
        the FWER threshold should drop or stay equal — never rise.
        """
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                     num_img=50, seed=0)
        exp, _ = EffectSynthetic.impose(exp, seed=0,
                                    extenter=ExtenterSphere(radius=2),
                                    effect_llr=0.5)

        ana_low = AnalysisGLOW(
            exp, n_perm_fwer=10, n_perm_inner=20,
            alpha_fwer=.1, min_vox=1)
        ana_high = AnalysisGLOW(
            exp, n_perm_fwer=10, n_perm_inner=20,
            alpha_fwer=.1, min_vox=4)

        # threshold under min_vox=4 should be <= threshold under min_vox=1
        # (using the same outer-perm seeds → comparable max-z draws)
        assert ana_high.adj_crit <= ana_low.adj_crit + 1e-9, (
            f'min_vox=4 threshold {ana_high.adj_crit:.3f} > '
            f'min_vox=1 threshold {ana_low.adj_crit:.3f}')


class TestPerRegionZConsistency:
    """``llr_z_0`` must equal (stat - mu_per_region) / sigma_per_region.

    Catches handoff bugs between the per-worker computation and the
    synth step's reuse of the worker's stored z-array.
    """

    def test_z_matches_stat_minus_mu_over_sigma(self):
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=50, seed=0)
        exp, _ = EffectSynthetic.impose(exp, seed=0,
                                   extenter=ExtenterSphere(radius=2),
                                   effect_llr=0.5)
        ana = AnalysisGLOW(exp, n_perm_fwer=5, n_perm_inner=20,
                           alpha_fwer=.5, min_vox=1)

        # the worker stores mu / sigma / z; synth attaches them back
        mu = ana._mu_per_region
        sigma = ana._sigma_per_region
        stat = ana.stat
        z_stored = ana.llr_z_0

        # only check entries where sigma is well above the floor and
        # neither input is NaN (matches the worker's sanitisation).
        ok = np.isfinite(stat) & np.isfinite(mu) & (sigma > 1e-9)
        z_expected = (stat[ok] - mu[ok]) / sigma[ok]
        np.testing.assert_allclose(z_stored[ok], z_expected,
                                   rtol=1e-9, atol=1e-9)


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

class TestResume:
    """Test AnalysisGLOW perm_dir-based resume behaviour."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(3, 3), num_img=20, seed=0)

    def test_no_perm_dir_by_default(self):
        """AnalysisGLOW works when perm_dir is None (temp dir, auto-clean)."""
        analysis = AnalysisGLOW(self.exp, n_perm_fwer=5, alpha_fwer=.5)
        assert hasattr(analysis, 'pval')
        assert hasattr(analysis, 'children')

    def test_perm_dir_resume(self):
        """Partial results in perm_dir are reused, completing the run."""
        import pickle
        import tempfile
        n_perm_fwer = 10
        n_perm_inner = 20
        min_vox = 1  # tiny test exp; min_vox=4 would mask everything

        # full run as reference
        ref = AnalysisGLOW(self.exp, n_perm_fwer=n_perm_fwer,
                            n_perm_inner=n_perm_inner,
                            min_vox=min_vox, alpha_fwer=.5)

        # write first 5 permutations into a temp perm_dir.  The pickle
        # must include mu/sigma/z/max_z so the synth step can reuse it,
        # so rerun_permutation gets the same n_perm_inner / min_vox.
        perm_dir = tempfile.mkdtemp(prefix='glow_test_resume_')
        for p in range(5):
            r = AnalysisGLOW.rerun_permutation(
                self.exp, p, n_perm_inner=n_perm_inner, min_vox=min_vox)
            with open(f'{perm_dir}/{p:06d}_result.pkl', 'wb') as f:
                pickle.dump(r, f)

        # resume from partial perm_dir
        resumed = AnalysisGLOW(
            self.exp, n_perm_fwer=n_perm_fwer,
            n_perm_inner=n_perm_inner, min_vox=min_vox,
            alpha_fwer=.5, perm_dir=perm_dir)

        np.testing.assert_array_equal(resumed.pval, ref.pval)

        import shutil
        shutil.rmtree(perm_dir, ignore_errors=True)

    def test_perm_dir_keeps_files(self):
        """When perm_dir is provided, outer-perm files are kept after run."""
        import tempfile
        perm_dir = tempfile.mkdtemp(prefix='glow_test_keep_')
        n_perm_fwer = 5

        AnalysisGLOW(self.exp, n_perm_fwer=n_perm_fwer,
                      n_perm_inner=10,
                      alpha_fwer=.5, perm_dir=perm_dir)

        from pathlib import Path
        result_files = list(Path(perm_dir).glob('*_result.pkl'))
        # only outer perms (0..n_perm_fwer) are written to perm_dir;
        # inner perms run inside each outer-perm worker (no per-perm files)
        assert len(result_files) == n_perm_fwer + 1

        import shutil
        shutil.rmtree(perm_dir, ignore_errors=True)


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
        ana = AnalysisGLOW(exp, n_perm_fwer=10, alpha_fwer=.5)

        children = ana.children
        assert children.shape == (num_vox - 2, 2), \
            f'expected {num_vox - 2} internal nodes, got {children.shape[0]}'

        all_nodes = list(iter_topo(children=children, num_leaf=num_vox))
        expected_total = num_vox + children.shape[0]
        assert len(all_nodes) == expected_total, \
            f'iter_topo yielded {len(all_nodes)}, expected {expected_total}'


class TestStreamingFidelity:
    """Verify that two identical runs produce the same results."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=50, seed=0)
    exp, effect = EffectSynthetic.impose(exp, seed=0,
                                    extenter=ExtenterSphere(radius=2),
                                    effect_llr=0.5)

    def test_reproducible(self):
        """Two runs with the same data must produce identical p-values."""
        n_perm_fwer = 25
        alpha_fwer = 0.1

        ana_a = AnalysisGLOW(self.exp, n_perm_fwer=n_perm_fwer,
                             alpha_fwer=alpha_fwer, verbose=False)
        ana_b = AnalysisGLOW(self.exp, n_perm_fwer=n_perm_fwer,
                             alpha_fwer=alpha_fwer, verbose=False)

        np.testing.assert_array_equal(ana_a.pval, ana_b.pval)
        assert set(ana_a.sig_reg_list) == set(ana_b.sig_reg_list)

        masks_a = sorted([e.mask.tobytes() for e in ana_a.effect_list])
        masks_b = sorted([e.mask.tobytes() for e in ana_b.effect_list])
        assert masks_a == masks_b

    def test_rerun_permutation(self):
        """rerun_permutation reproduces the same result as a full run."""
        import pickle, tempfile
        n_perm_fwer = 10
        perm_dir = tempfile.mkdtemp(prefix='glow_test_rerun_')

        AnalysisGLOW(self.exp, n_perm_fwer=n_perm_fwer, alpha_fwer=.1,
                      perm_dir=perm_dir)

        with open(f'{perm_dir}/{3:06d}_result.pkl', 'rb') as f:
            stored = pickle.load(f)
        rerun = AnalysisGLOW.rerun_permutation(self.exp, perm_idx=3)

        np.testing.assert_array_equal(stored['stat'], rerun['stat'])
        np.testing.assert_array_equal(stored['size'], rerun['size'])
        np.testing.assert_array_equal(stored['children'], rerun['children'])

        import shutil
        shutil.rmtree(perm_dir, ignore_errors=True)


class TestFromPrecomputed:
    """Test factory classmethods for constructing analysis from pre-computed data."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=100, seed=0)
    exp_eff, effect = EffectSynthetic.impose(exp, seed=0,
                                         extenter=ExtenterSphere(radius=2),
                                         effect_llr=0.5)

    def test_vba_from_precomputed(self):
        from glow.analysis.mancova import get_wilks
        ana = AnalysisVBA(self.exp_eff, n_perm_fwer=25, alpha_fwer=.5,
                          get_stat=get_wilks)
        ana2 = AnalysisVBA.from_precomputed(
            exp=self.exp_eff, get_stat=get_wilks,
            stat=ana.stat, alpha_fwer=.5)
        np.testing.assert_array_equal(ana2.pval, ana.pval)
        assert len(ana2.effect_list) == len(ana.effect_list)

    def test_cet_from_precomputed(self):
        from glow.analysis.mancova import get_wilks
        ana = AnalysisCET(self.exp_eff, n_perm_fwer=25, alpha_fwer=.5,
                          cft_pval=0.01, get_stat=get_wilks)
        ana2 = AnalysisCET.from_precomputed(
            exp=self.exp_eff, get_stat=get_wilks,
            stat=ana.stat, cft=ana.cft, cft_pval=ana.cft_pval,
            alpha_fwer=.5)
        np.testing.assert_array_equal(ana2.pval, ana.pval)

    def test_glow_from_precomputed_has_attrs(self):
        from glow.analysis.mancova import get_llr
        ana = AnalysisGLOW.from_precomputed(
            exp=self.exp_eff, get_stat=get_llr, verbose=True)
        assert hasattr(ana, 'exp')
        assert ana.get_stat is get_llr
        assert ana.verbose is True

    def test_discover_mask_on_base(self):
        """Verify Analysis.discover_mask works (it was moved from AnalysisVBA)."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=10, seed=0)
        effects = Analysis.discover_mask(mask=mask, exp=exp)
        assert len(effects) == 1
        np.testing.assert_array_equal(effects[0].mask, mask)


class TestInnerPermRace:
    """Lockstep inner-FL race over a pre-built survivor kernel."""

    @staticmethod
    def _setup(num_img=40, shape=(8, 8), effect_llr=0.6, radius=3,
                seed=0):
        import glow.graph
        from glow.analysis.cluster import cluster
        from glow.analysis.mancova import decompose

        exp = Experiment.from_gauss(a=2, b=1, shape=shape,
                                    num_img=num_img, seed=seed)
        exp, _ = EffectSynthetic.impose(
            exp, seed=seed,
            extenter=ExtenterSphere(radius=radius),
            effect_llr=effect_llr)

        children = cluster(exp=exp, mode='q1')
        num_vox = exp.y.shape[2]
        layer = glow.graph.compute_tree_layers(children, num_vox)
        q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
        llr_outer, size = glow.graph.compute_llr_batched(
            exp, children=children, q0=q0, q1=q1, layer=layer)
        return exp, children, q0, q1, llr_outer, size

    @staticmethod
    def _make_draw_one(exp, children, q0, q1, size, min_vox=4,
                       base=100_000):
        import glow.graph

        q0_proj = q0.T @ q0
        n_img = exp.y.shape[1]
        eye_n = np.eye(n_img, dtype=exp.y.dtype)
        num_reg = children.shape[0] + exp.y.shape[2]

        survivor_idx = np.where(size >= min_vox)[0]
        kernels = glow.graph.build_survivor_kernels(
            exp.y, children, survivor_idx, q0)

        def draw_one(i):
            rng = np.random.default_rng(base + i)
            perm = np.argsort(rng.permutation(n_img))
            freed_lane = (eye_n - q0_proj)[:, perm] + q0_proj
            return glow.graph.compute_llr_inner_kernel(
                kernels, q0, q1, freed_lane, perm, num_reg,
                min_size=min_vox)

        return draw_one

    def test_leader_is_argmax_z(self):
        """Tournament leader equals argmax of final z on the active set."""
        from glow.analysis._glow import _run_inner_race

        exp, children, q0, q1, llr_outer, size = self._setup()
        draw_one = self._make_draw_one(exp, children, q0, q1, size)

        out = _run_inner_race(
            draw_one, n_max=100, llr_outer=llr_outer, size=size,
            min_vox=4, race_batch=25, race_k_sigma=3.0,
            z_threshold=None)

        assert out['leader'] >= 0, 'leader not assigned'

        sigma_safe = np.where(out['sigma'] < 1e-12, 1.0, out['sigma'])
        z = (llr_outer - out['mu']) / sigma_safe
        eligible = (size >= 4) & np.isfinite(z)
        expected = int(np.argmax(np.where(eligible, z, -np.inf)))
        assert out['leader'] == expected, (
            f"leader={out['leader']} but argmax z = {expected}")

    def test_max_z_matches_no_race_reference(self):
        """Race's leader z exactly matches a buffer reference over
        the same number of perms (lockstep perms are bit-identical)."""
        from glow.analysis._glow import _run_inner_race

        exp, children, q0, q1, llr_outer, size = self._setup()
        draw_one = self._make_draw_one(exp, children, q0, q1, size)

        out = _run_inner_race(
            draw_one, n_max=100, llr_outer=llr_outer, size=size,
            min_vox=4, race_batch=25, race_k_sigma=4.0)
        leader = out['leader']
        n_used = out['n_inner_used']
        z_race = ((llr_outer[leader] - out['mu'][leader])
                   / max(out['sigma'][leader], 1e-12))

        # Reference: rerun the same first n_used perms, take mean /
        # std at the leader.
        draws = np.array([draw_one(i)[leader] for i in range(n_used)])
        finite = np.isfinite(draws)
        mu_ref = draws[finite].mean()
        sigma_ref = draws[finite].std(ddof=1)
        z_ref = (llr_outer[leader] - mu_ref) / max(sigma_ref, 1e-12)

        np.testing.assert_allclose(z_race, z_ref, rtol=1e-10,
                                    atol=1e-10)

    def test_race_vs_slow_mu_sigma(self):
        """Race-with-no-trim matches a slow buffer pass over the same
        seeds.  Replaces the deleted Path-C reference in
        ``AnalysisGLOW._process_permutation`` — proves the race's
        Welford accounting equals nanmean / nanstd over a materialised
        ``(n_perm_inner, num_reg)`` buffer."""
        from glow.analysis._glow import _run_inner_race
        from ._slow_inner import compute_slow_inner
        import glow.graph

        exp, children, q0, q1, llr_outer, size = self._setup()
        num_vox = exp.y.shape[2]
        layer = glow.graph.compute_tree_layers(children, num_vox)

        n_perm_inner = 30
        base = 100_000
        min_vox = 4

        slow = compute_slow_inner(
            exp, children, q0, q1, layer,
            n_perm_inner=n_perm_inner, base_seed=base, min_vox=min_vox)

        # race_init = n_max forces every perm through draw_one_slow
        # during warmup; the trim/swap loop never executes.
        def draw_one(i):
            _exp_i = exp.permute(base + i)
            llr_i, _ = glow.graph.compute_llr_batched(
                _exp_i, children=children, q0=q0, q1=q1,
                min_size=min_vox, layer=layer)
            return llr_i

        race = _run_inner_race(
            draw_one, n_max=n_perm_inner, llr_outer=llr_outer,
            size=size, min_vox=min_vox,
            race_init=n_perm_inner, race_batch=25, race_k_sigma=3.0,
            z_threshold=None, on_warmup_done=None)

        assert race['n_inner_used'] == n_perm_inner, (
            f'race ran {race["n_inner_used"]} perms; expected '
            f'{n_perm_inner} (race_init = n_max should warmup all)')

        finite_mu = np.isfinite(slow['mu']) & np.isfinite(race['mu'])
        assert finite_mu.any(), 'no finite mu cells to compare'
        np.testing.assert_allclose(
            race['mu'][finite_mu], slow['mu'][finite_mu],
            rtol=1e-10, atol=1e-12,
            err_msg='race mu disagrees with buffer reference')

        finite_sigma = (np.isfinite(slow['sigma'])
                        & np.isfinite(race['sigma']))
        np.testing.assert_allclose(
            race['sigma'][finite_sigma], slow['sigma'][finite_sigma],
            rtol=1e-8, atol=1e-10,
            err_msg='race sigma disagrees with buffer reference')

    def test_threshold_mode_sig_regions_above_threshold(self):
        """Threshold mode: every flagged region has final z > threshold."""
        from glow.analysis._glow import _run_inner_race

        exp, children, q0, q1, llr_outer, size = self._setup()
        draw_one = self._make_draw_one(exp, children, q0, q1, size)

        z_thresh = 2.0
        out = _run_inner_race(
            draw_one, n_max=80, llr_outer=llr_outer, size=size,
            min_vox=4, race_batch=10, race_k_sigma=4.0,
            z_threshold=z_thresh)

        sigma_safe = np.where(out['sigma'] < 1e-12, 1.0, out['sigma'])
        z_final = (llr_outer - out['mu']) / sigma_safe
        for r in out['sig_regions']:
            assert z_final[r] > z_thresh, (
                f'sig region {r} has final z={z_final[r]:.3f} '
                f'≤ threshold {z_thresh}')

    def test_glow_end_to_end_with_race(self):
        """Race-on AnalysisGLOW discovers the synthetic effect."""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=80, seed=0)
        exp, effect = EffectSynthetic.impose(
            exp, seed=0, extenter=ExtenterSphere(radius=2),
            effect_llr=0.5)

        analysis = AnalysisGLOW(exp, n_perm_fwer=25, alpha_fwer=0.1)
        assert len(analysis.effect_list) >= 1, \
            'race-on found no effects despite strong signal'

    def test_warmup_hook_swaps_draw_one(self):
        """on_warmup_done's returned callable replaces draw_one after
        the first post-warmup trim — that's the 2-stage swap that
        makes the race cheap at large num_vox."""
        from glow.analysis._glow import _run_inner_race

        num_reg = 50
        rng = np.random.default_rng(0)
        llr_outer = rng.uniform(0.0, 3.0, num_reg)
        size = np.full(num_reg, 10, dtype=np.int64)
        calls = {'init': 0, 'post': 0}

        def draw_init(i):
            calls['init'] += 1
            return np.random.default_rng(i).normal(0, 1, num_reg)

        def draw_post(i):
            calls['post'] += 1
            return np.random.default_rng(i + 100000).normal(0, 1, num_reg)

        race_init = 20
        out = _run_inner_race(
            draw_init, n_max=200, llr_outer=llr_outer, size=size,
            min_vox=4, race_init=race_init, race_batch=25,
            race_k_sigma=3.0,
            on_warmup_done=lambda active: draw_post)

        assert calls['init'] == race_init, (
            f'warmup draw_one called {calls["init"]} times, '
            f'expected {race_init}')
        assert calls['post'] > 0, \
            'post-warmup draw_one never called; hook did not swap'
