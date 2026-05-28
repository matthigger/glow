from glow.effect import ExtenterSphere, EffectSynthetic
from glow.experiment import *
from glow.analysis import *
from glow.graph import get_dice_sens_spec, iter_postorder
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
    effect = EffectSynthetic(extenter=ExtenterSphere(radius=2),
                             effect_llr=0.5, seed=0)
    exp = effect.fit(exp)

    def test_glow(self):
        analysis = AnalysisGLOW(TestBigEffect.exp, n_perm_fwer=25, alpha_fwer=.1).fit()

        # check that target region segmented properly
        dice = get_dice_sens_spec(mask=TestBigEffect.effect.mask_,
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
                                   alpha_fwer=.1, **kwargs).fit()
        mask_all = sum(eff.mask for eff in analysis.effect_list)
        np.testing.assert_allclose(mask_all,
                                   TestBigEffect.effect.mask_)
    
    def test_vba_wilks(self):
        """VBA with Wilks' Lambda (1 - Wilks) should detect effects.

        get_wilks returns 1 - Lambda, so larger = more evidence against H0,
        compatible with max-stat and TFCE without sign correction.
        """
        from glow.analysis.mancova import get_wilks
        # non-TFCE
        ana = AnalysisVBA(TestBigEffect.exp, n_perm_fwer=25,
                          alpha_fwer=.5, tfce_flag=False,
                          get_stat=get_wilks).fit()
        assert np.nanmin(ana.pval) <= 0.5, (
            f'Wilks VBA produced no small p-values '
            f'(min={np.nanmin(ana.pval):.3f})')

        # TFCE: 1-Wilks is non-negative, so TFCE works directly
        ana_tfce = AnalysisVBA(TestBigEffect.exp, n_perm_fwer=25,
                               alpha_fwer=.5, tfce_flag=True,
                               get_stat=get_wilks).fit()
        assert np.nanmin(ana_tfce.pval) <= 0.5, (
            f'Wilks VBA-TFCE produced no small p-values '
            f'(min={np.nanmin(ana_tfce.pval):.3f})')

    def test_glow_with_prune(self):
        """test GLOW with default pruning (llr_z, lam=0)"""
        analysis = AnalysisGLOW(
            TestBigEffect.exp,
            n_perm_fwer=10,
            alpha_fwer=.1,
        ).fit()

        # should still find the effect
        assert len(analysis.effect_list) > 0

    def test_glow_node(self):
        """Greedy pruning should find the true effect."""
        analysis = AnalysisGLOW(
            TestBigEffect.exp,
            n_perm_fwer=25,
            alpha_fwer=.1,
        ).fit()

        assert len(analysis.effect_list) >= 1, \
            f'expected at least 1 effect, got {len(analysis.effect_list)}'

    def test_glow_with_adjustment(self):
        """test GLOW with adjustment permutations"""
        analysis = AnalysisGLOW(
            TestBigEffect.exp,
            n_perm_fwer=10,
            alpha_fwer=.1
        ).fit()

        assert hasattr(analysis, 'children')
        assert hasattr(analysis, 'pval')


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
                          alpha_fwer=.1, cft_pval=0.01).fit()
        assert len(ana.effect_list) >= 1

    def test_big_effect_z(self):
        """CET with z_flag runs without error and sets the flag."""
        ana = AnalysisCET(TestBigEffect.exp, n_perm_fwer=25,
                          alpha_fwer=.5, cft_pval=0.05, z_flag=True).fit()
        assert ana.z_flag is True
        assert hasattr(ana, 'pval')
        assert hasattr(ana, 'effect_list')

    def test_null_no_discoveries(self):
        """Under the null (no effect), CET should not discover at alpha=0.05."""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=100, seed=0)
        ana = AnalysisCET(exp, n_perm_fwer=25, alpha_fwer=.05).fit()
        assert len(ana.effect_list) == 0

    def test_pval_bounds(self):
        """p-values in [1/n_perm, 1]."""
        n = 25
        ana = AnalysisCET(TestBigEffect.exp, n_perm_fwer=n, alpha_fwer=.1).fit()
        assert (ana.pval >= 1 / n).all()
        assert (ana.pval <= 1.0).all()

    def test_cluster_members_share_pval(self):
        """All voxels in a discovered cluster should have the same p-value."""
        ana = AnalysisCET(TestBigEffect.exp, n_perm_fwer=25,
                          alpha_fwer=.5, cft_pval=0.01).fit()
        for eff in ana.effect_list:
            vox_idx = ana.exp.mask_idx[eff.mask]
            pvals = ana.pval[vox_idx]
            assert np.all(pvals == pvals[0])


class TestAnalysisEdgeCases:
    """test edge cases and error handling"""
    
    def test_all_regions_too_small(self):
        """test when all regions are filtered out by min_vox"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=1),
                              effect_llr=0.5, seed=0).fit(exp)

        # set min_vox so large that all regions are filtered
        analysis = AnalysisGLOW(
            exp,
            n_perm_fwer=5,
            alpha_fwer=.1,
            min_vox=1000000  # impossibly large
        ).fit()
        
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
        ).fit()
        
        # should run without error even with small size
        assert hasattr(analysis, 'pval')
        assert analysis.pval.shape[0] > 0
    
    def test_different_alpha_values(self):
        """test with different alpha thresholds"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=1),
                              effect_llr=0.5, seed=0).fit(exp)
        
        # strict alpha
        analysis_strict = AnalysisGLOW(
            exp,
            n_perm_fwer=5,
            alpha_fwer=.01
        ).fit()

        # lenient alpha
        analysis_lenient = AnalysisGLOW(
            exp,
            n_perm_fwer=5,
            alpha_fwer=.5
        ).fit()
        
        # lenient should find same or more effects
        assert len(analysis_lenient.effect_list) >= len(analysis_strict.effect_list)


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
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=1),
                              effect_llr=0.5, seed=0).fit(exp)

        analysis = AnalysisGLOW(exp, n_perm_fwer=5, alpha_fwer=0.05,
                                min_vox=1).fit()

        assert not np.any(np.isinf(analysis.z)), \
            'z contains inf (likely zero-std division)'
        assert not np.any(np.isnan(analysis.z)), \
            'z contains nan (likely zero-std division)'


class TestMinVox:
    """min_vox gates the FWER comparison set."""

    def test_no_significant_region_below_min_vox(self):
        """No significant region in effect_list should have size < min_vox."""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                     num_img=50, seed=0)
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=2),
                              effect_llr=0.5, seed=0).fit(exp)

        min_vox = 4
        ana = AnalysisGLOW(
            exp, n_perm_fwer=10, n_perm_inner=20,
            alpha_fwer=.5, min_vox=min_vox).fit()

        # every significant region must have size >= min_vox
        sig = np.where(ana.pval <= ana.alpha_fwer)[0]
        for reg_idx in sig:
            assert ana.size[reg_idx] >= min_vox, (
                f'reg {reg_idx} has size {ana.size[reg_idx]} '
                f'< min_vox={min_vox} but appears significant')

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
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=2),
                              effect_llr=0.5, seed=0).fit(exp)

        ana_low = AnalysisGLOW(
            exp, n_perm_fwer=10, n_perm_inner=20,
            alpha_fwer=.1, min_vox=1).fit()
        ana_high = AnalysisGLOW(
            exp, n_perm_fwer=10, n_perm_inner=20,
            alpha_fwer=.1, min_vox=4).fit()

        # threshold under min_vox=4 should be <= threshold under min_vox=1
        # (using the same outer-perm seeds → comparable max-z draws)
        crit_high = np.quantile(ana_high.max_z_null, 1 - ana_high.alpha_fwer,
                                method='higher')
        crit_low = np.quantile(ana_low.max_z_null, 1 - ana_low.alpha_fwer,
                               method='higher')
        assert crit_high <= crit_low + 1e-9, (
            f'min_vox=4 threshold {crit_high:.3f} > '
            f'min_vox=1 threshold {crit_low:.3f}')


class TestPerRegionZConsistency:
    """z must equal (stat - mu) / std on the observed tree."""

    def test_z_matches_stat_minus_mu_over_std(self):
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=50, seed=0)
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=2),
                              effect_llr=0.5, seed=0).fit(exp)
        ana = AnalysisGLOW(exp, n_perm_fwer=5, n_perm_inner=20,
                           alpha_fwer=.5, min_vox=1).fit()

        mu = ana.mu
        std = ana.std
        llr = ana.llr
        z_stored = ana.z

        # only check entries where std is well above the floor and
        # neither input is NaN (matches the worker's sanitisation).
        ok = np.isfinite(llr) & np.isfinite(mu) & (std > 1e-9)
        z_expected = (llr[ok] - mu[ok]) / std[ok]
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
        ana = AnalysisGLOW(exp, n_perm_fwer=10, alpha_fwer=.5).fit()

        children = ana.children
        assert children.shape == (num_vox - 2, 2), \
            f'expected {num_vox - 2} internal nodes, got {children.shape[0]}'

        all_nodes = list(iter_postorder(children=children, num_leaf=num_vox))
        expected_total = num_vox + children.shape[0]
        assert len(all_nodes) == expected_total, \
            f'iter_postorder yielded {len(all_nodes)}, expected {expected_total}'


class TestStreamingFidelity:
    """Verify that two identical runs produce the same results."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=50, seed=0)
    effect = EffectSynthetic(extenter=ExtenterSphere(radius=2),
                             effect_llr=0.5, seed=0)
    exp = effect.fit(exp)

    def test_reproducible(self):
        """Two runs with the same data must produce identical p-values."""
        n_perm_fwer = 25
        alpha_fwer = 0.1

        ana_a = AnalysisGLOW(self.exp, n_perm_fwer=n_perm_fwer,
                             alpha_fwer=alpha_fwer).fit()
        ana_b = AnalysisGLOW(self.exp, n_perm_fwer=n_perm_fwer,
                             alpha_fwer=alpha_fwer).fit()

        np.testing.assert_array_equal(ana_a.pval, ana_b.pval)
        np.testing.assert_array_equal(ana_a.max_z_null, ana_b.max_z_null)

        masks_a = sorted([e.mask.tobytes() for e in ana_a.effect_list])
        masks_b = sorted([e.mask.tobytes() for e in ana_b.effect_list])
        assert masks_a == masks_b


class TestAnalysisScaling:
    """Analysis must scale the experiment if it isn't already scaled."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
    assert not isinstance(exp, ExperimentScaled)

    def _check_scales(self, AnalysisCls, **kwargs):
        # unscaled in → wrapped in ExperimentScaled
        ana = AnalysisCls(self.exp, **kwargs)
        assert isinstance(ana.exp, ExperimentScaled)

        # already-scaled in → kept as-is (not re-wrapped)
        exp_scaled = ExperimentScaled.from_exp(self.exp)
        ana_pre = AnalysisCls(exp_scaled, **kwargs)
        assert ana_pre.exp is exp_scaled

    def test_glow_scales(self):
        self._check_scales(AnalysisGLOW, n_perm_fwer=2)

    def test_vba_scales(self):
        self._check_scales(AnalysisVBA, n_perm_fwer=2)

    def test_cet_scales(self):
        self._check_scales(AnalysisCET, n_perm_fwer=2)


class TestFromPrecomputed:
    """Test factory classmethods for constructing analysis from pre-computed data."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=100, seed=0)
    effect = EffectSynthetic(extenter=ExtenterSphere(radius=2),
                             effect_llr=0.5, seed=0)
    exp_eff = effect.fit(exp)

    def test_discover_mask_on_base(self):
        """Verify Analysis.discover_mask works (it was moved from AnalysisVBA)."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=10, seed=0)
        effects = Analysis.discover_mask(mask=mask, exp=exp)
        assert len(effects) == 1
        np.testing.assert_array_equal(effects[0].mask, mask)

