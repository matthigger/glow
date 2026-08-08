import pytest

from glow.effect import ExtenterSphere, EffectSynthetic
from glow.experiment import *
from glow.analysis import *
from glow.analysis.mancova import get_hotel_tr, get_pillai, get_wilks
from glow.graph import confusion_counts_tree
from glow.mask import get_mask_idx, stats_from_counts


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
    effect = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                             effect_llr=0.5)
    exp, mask_target = effect.fit(exp)

    def test_glow(self):
        analysis = AnalysisGLOW(n_perm_fwer=25, alpha_fwer=.1).fit(
            TestBigEffect.exp)

        # the Ward tree must carry the target as one of its nodes before a
        # discovery over that tree could recover it
        counts = confusion_counts_tree(mask=TestBigEffect.mask_target,
                                      mask_idx=TestBigEffect.exp.mask_idx,
                                      children=analysis.children)
        dice = stats_from_counts(**counts)['dice']
        assert np.isclose(dice.max(), 1), 'target region not segmented'

        # and the discovered effects must be exactly it
        mask_all = sum(eff.mask for eff in analysis.effect_list)
        np.testing.assert_array_equal(mask_all, TestBigEffect.mask_target)

    @pytest.mark.parametrize('tfce_flag', [False, True])
    def test_vba(self, tfce_flag):
        analysis = AnalysisVBA(n_perm_fwer=25,
                               alpha_fwer=.1, tfce_flag=tfce_flag).fit(
            TestBigEffect.exp)
        mask_all = sum(eff.mask for eff in analysis.effect_list)
        np.testing.assert_array_equal(mask_all, TestBigEffect.mask_target)


class TestStatDefaults:
    """The stat each arm takes when the caller names none.

    These are the arms the paper reports, so a silent change to them
    changes every published VBA / CET / TFCE number. Every other test
    passes get_stat= explicitly, which would leave such a change green --
    hence pinning the resolved attributes here.

    The choices come from the vba_stat bake-off: the raw Hotelling-Lawley
    trace for VBA and CET, and the z-scored 1 - Wilks for TFCE, whose
    single height grid needs one voxel's stat to mean what another's does.
    """

    def test_vba_defaults_to_raw_hotelling(self):
        ana = AnalysisVBA(n_perm_fwer=2)
        assert ana.get_stat is get_hotel_tr
        assert ana.z_flag is False
        assert ana.tfce_flag is False

    def test_vba_tfce_defaults_to_z_scored_wilks(self):
        ana = AnalysisVBA(n_perm_fwer=2, tfce_flag=True)
        assert ana.get_stat is get_wilks
        assert ana.z_flag is True

    def test_cet_defaults_to_raw_hotelling(self):
        ana = AnalysisCET(n_perm_fwer=2)
        assert ana.get_stat is get_hotel_tr
        assert ana.z_flag is False

    @pytest.mark.parametrize('tfce_flag', [False, True])
    def test_explicit_arguments_still_win(self, tfce_flag):
        """A named stat / z_flag overrides the arm default, either way."""
        ana = AnalysisVBA(n_perm_fwer=2, tfce_flag=tfce_flag,
                          get_stat=get_pillai, z_flag=not tfce_flag)
        assert ana.get_stat is get_pillai
        assert ana.z_flag is (not tfce_flag)


class TestZScoreStat:
    """z_score_stat lives on Analysis."""

    def test_constant_row_safe(self):
        """A constant row should not produce inf or nan."""
        stat = np.ones((5, 10))
        z = Analysis.z_score_stat(stat)
        assert not np.any(np.isinf(z))
        assert not np.any(np.isnan(z))


class TestCET:
    def test_big_effect(self):
        """CET recovers the strong effect exactly."""
        ana = AnalysisCET(n_perm_fwer=25,
                          alpha_fwer=.1, cft_pval=0.01).fit(TestBigEffect.exp)
        mask_all = sum(eff.mask for eff in ana.effect_list)
        np.testing.assert_array_equal(mask_all, TestBigEffect.mask_target)

    def test_cluster_members_share_pval(self):
        """All voxels in a discovered cluster should have the same p-value."""
        ana = AnalysisCET(n_perm_fwer=25,
                          alpha_fwer=.5, cft_pval=0.01).fit(TestBigEffect.exp)
        for eff in ana.effect_list:
            vox_idx = TestBigEffect.exp.mask_idx[eff.mask]
            pvals = ana.pval[vox_idx]
            assert np.all(pvals == pvals[0])


class TestAnalysisEdgeCases:
    """test edge cases and error handling"""
    
    def test_all_regions_too_small(self):
        """test when all regions are filtered out by min_vox"""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=1, seed=0),
                              effect_llr=0.5).fit(exp)[0]

        # set min_vox so large that all regions are filtered
        analysis = AnalysisGLOW(
            n_perm_fwer=5,
            alpha_fwer=.1,
            min_vox=1000000  # impossibly large
        ).fit(exp)
        
        # should run without error
        assert hasattr(analysis, 'pval')
        assert hasattr(analysis, 'effect_list')
        
        # all pvals should be nan
        assert np.all(np.isnan(analysis.pval))
        
        # no effects should be found
        assert len(analysis.effect_list) == 0


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
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=1, seed=0),
                              effect_llr=0.5).fit(exp)[0]

        analysis = AnalysisGLOW(n_perm_fwer=5, alpha_fwer=0.05,
                                min_vox=1).fit(exp)

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
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                              effect_llr=0.5).fit(exp)[0]

        min_vox = 4
        ana = AnalysisGLOW(
            n_perm_fwer=10, n_perm_inner=20,
            alpha_fwer=.5, min_vox=min_vox).fit(exp)

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


class TestPerRegionZConsistency:
    """z must equal (stat - mu) / std on the observed tree."""

    def test_z_matches_stat_minus_mu_over_std(self):
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=50, seed=0)
        exp = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                              effect_llr=0.5).fit(exp)[0]
        ana = AnalysisGLOW(n_perm_fwer=5, n_perm_inner=20,
                           alpha_fwer=.5, min_vox=1).fit(exp)

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
        ana = AnalysisGLOW(n_perm_fwer=10, alpha_fwer=.5).fit(exp)

        # GLOW completes on a forest: 2 components → num_vox - 2 internal nodes
        children = ana.children
        assert children.shape == (num_vox - 2, 2), \
            f'expected {num_vox - 2} internal nodes, got {children.shape[0]}'


class TestStreamingFidelity:
    """Verify that two identical runs produce the same results."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=50, seed=0)
    effect = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                             effect_llr=0.5)
    exp = effect.fit(exp)[0]

    def test_reproducible(self):
        """Two runs with the same data must produce identical p-values."""
        n_perm_fwer = 25
        alpha_fwer = 0.1

        ana_a = AnalysisGLOW(n_perm_fwer=n_perm_fwer,
                             alpha_fwer=alpha_fwer).fit(self.exp)
        ana_b = AnalysisGLOW(n_perm_fwer=n_perm_fwer,
                             alpha_fwer=alpha_fwer).fit(self.exp)

        np.testing.assert_array_equal(ana_a.pval, ana_b.pval)
        np.testing.assert_array_equal(ana_a.max_z_null, ana_b.max_z_null)

        masks_a = sorted([e.mask.tobytes() for e in ana_a.effect_list])
        masks_b = sorted([e.mask.tobytes() for e in ana_b.effect_list])
        assert masks_a == masks_b


class TestNJobsDeterminism:
    """VBA / CET fits are identical serial vs joblib-parallel.

    Each permutation row is seeded by its index (exp.permute(k)), so the
    stat matrix, p-values, and discovered effects must not depend on
    n_jobs. Covers plain VBA, z-scored VBA, VBA+TFCE (parallel TFCE
    loop), and CET (parallel stat walk).
    """

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=50, seed=0)
    effect = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                             effect_llr=0.5)
    exp = effect.fit(exp)[0]

    @pytest.mark.parametrize('make_ana', [
        lambda: AnalysisVBA(n_perm_fwer=10, alpha_fwer=.1),
        lambda: AnalysisVBA(n_perm_fwer=10, alpha_fwer=.1, z_flag=True),
        lambda: AnalysisVBA(n_perm_fwer=10, alpha_fwer=.1, tfce_flag=True),
        lambda: AnalysisCET(n_perm_fwer=10, alpha_fwer=.1, cft_pval=.05),
    ])
    def test_serial_matches_parallel(self, make_ana):
        ana_serial = make_ana().fit(self.exp, n_jobs=1)
        ana_par = make_ana().fit(self.exp, n_jobs=2)

        # bit-identical stat matrix and p-values
        np.testing.assert_array_equal(ana_serial.stat, ana_par.stat)
        np.testing.assert_array_equal(ana_serial.pval, ana_par.pval)

        # identical discovered effects
        masks_serial = sorted(e.mask.tobytes() for e in ana_serial.effect_list)
        masks_par = sorted(e.mask.tobytes() for e in ana_par.effect_list)
        assert masks_serial == masks_par


class TestAnalysisScaling:
    """fit scales the experiment via the idempotent ExperimentScaled.from_exp.

    The experiment is no longer stored on the Analysis (it is passed to fit);
    each fit calls ExperimentScaled.from_exp, which scales a raw experiment
    and returns an already-scaled one unchanged.
    """

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=20, seed=0)
    assert not isinstance(exp, ExperimentScaled)

    def test_from_exp_wraps_unscaled(self):
        # raw experiment → scaled
        assert isinstance(ExperimentScaled.from_exp(self.exp), ExperimentScaled)

    def test_from_exp_is_idempotent(self):
        # already-scaled in → returned as-is (no double-scaling)
        exp_scaled = ExperimentScaled.from_exp(self.exp)
        assert ExperimentScaled.from_exp(exp_scaled) is exp_scaled

    @pytest.mark.parametrize('AnalysisCls, kwargs', [
        (AnalysisGLOW, dict(n_perm_fwer=2)),
        (AnalysisVBA, dict(n_perm_fwer=2)),
        (AnalysisCET, dict(n_perm_fwer=2)),
    ])
    def test_fit_accepts_raw_and_scaled(self, AnalysisCls, kwargs):
        """Scaling inside fit is idempotent, so both inputs give one answer.

        from_exp returns an already-scaled experiment unchanged, so passing
        the raw and the pre-scaled form must not merely both run -- they
        must produce the same p-values.
        """
        ana_raw = AnalysisCls(**kwargs).fit(self.exp)
        ana_scaled = AnalysisCls(**kwargs).fit(
            ExperimentScaled.from_exp(self.exp))

        np.testing.assert_array_equal(ana_raw.pval, ana_scaled.pval)


class TestDiscoverMask:
    """Analysis.discover_mask splits a mask into connected-component effects."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=100, seed=0)
    effect = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                             effect_llr=0.5)
    exp_eff = effect.fit(exp)[0]

    def test_single_connected_mask_is_one_effect(self):
        """A single connected blob yields exactly one effect equal to the mask."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=10, seed=0)
        effects = Analysis.discover_mask(mask=mask, exp=exp)
        assert len(effects) == 1
        np.testing.assert_array_equal(effects[0].mask, mask)



class TestNanSafeReductions:
    """Every reduction over the stat matrix skips NaN rather than spreading it.

    NaN is the matrix's sentinel for a region with no usable statistic.
    Nothing repairs it -- no stat function can return +-inf, and the
    voxels with no variance are gone before an analysis sees the data
    (Experiment.drop_constant_vox) -- so each reader has to handle it.
    """

    @staticmethod
    def build_stat(n_perm=20, num_vox=1000):
        """(n_perm+1, num_vox) positive stats, a blob in the observed row."""
        rng = np.random.default_rng(0)
        stat = np.abs(rng.standard_normal((n_perm + 1, num_vox))) * 3
        stat[0, 400:460] += 8
        return stat

    def test_all_nan_column_stays_nan_silently(self):
        """An all-NaN column comes back NaN, with no RuntimeWarning."""
        stat = self.build_stat()
        stat[:, 3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            z = Analysis.z_score_stat(stat)
        assert np.isnan(z[:, 3]).all()
        assert np.isfinite(np.delete(z, 3, axis=1)).all()

    def test_nan_column_does_not_move_the_others(self):
        """Blanking one voxel does not shift any other voxel's z."""
        stat = self.build_stat()
        clean = Analysis.z_score_stat(stat.copy())
        stat[:, 3] = np.nan
        z = Analysis.z_score_stat(stat)
        np.testing.assert_allclose(np.delete(z, 3, axis=1),
                                   np.delete(clean, 3, axis=1))

    def test_all_nan_perm_excluded_from_null(self):
        """A permutation with no valid voxel drops out of the max-stat null.

        nanmax would call that row's max NaN, which sorts to the top of
        the null and silently raises every p-value.
        """
        stat = self.build_stat()
        stat[7, :] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            pval = Analysis.get_pval(stat)
        assert np.isfinite(np.nanmin(pval))
        assert np.nanmin(pval) < 1.0

    def test_all_nan_matrix_gives_all_nan(self):
        """Nothing valid anywhere is NaN p-values, not a ZeroDivisionError."""
        stat = np.full((21, 1000), np.nan)
        assert np.isnan(Analysis.get_pval(stat)).all()


class TestCetNanThreshold:
    """CET's threshold and p-values ignore the voxels with no statistic."""

    @staticmethod
    def build():
        """A stat matrix and its mask, one voxel carrying no statistic."""
        rng = np.random.default_rng(0)
        mask_idx = np.arange(1000).reshape((10, 10, 10))
        stat = np.abs(rng.standard_normal((201, 1000))) * 3
        stat[0, 400:460] += 8
        stat[:, 3] = np.nan
        return stat, mask_idx

    def test_cft_finite_with_nan_present(self):
        """np.quantile would return NaN here, and then detect nothing."""
        stat, mask_idx = self.build()
        cft = np.nanquantile(stat[1:, :].ravel(), 1 - 0.001)
        assert np.isfinite(cft)
        pval = AnalysisCET._get_pval_cet(stat, mask_idx, cft)
        assert np.nanmin(pval) < 1.0

    def test_dropped_voxel_gets_nan_not_one(self):
        """A voxel with no statistic leaves the family, not fails in it."""
        stat, mask_idx = self.build()
        cft = np.nanquantile(stat[1:, :].ravel(), 1 - 0.001)
        pval = AnalysisCET._get_pval_cet(stat, mask_idx, cft)
        assert np.isnan(pval[3])
        assert np.isfinite(np.delete(pval, 3)).all()


class TestScreenCutsAcrossArms:
    """One screen upstream, so every method tests the same voxels.

    The point of dropping in pre-processing rather than inside a fit:
    GLOW and the voxel-wise arms control FWER over one family, instead
    of each arriving at its own by whatever its statistic happened to
    return on a voxel with nothing in it.
    """

    DEAD_V = 10

    @pytest.fixture
    def exp(self):
        """A screened experiment, one voxel flat across images."""
        exp = Experiment.from_gauss(a=2, b=2, shape=(6, 6, 6), num_img=40,
                                    seed=0)
        exp.y[:, :, self.DEAD_V] = 3.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return exp.drop_constant_vox()

    @pytest.fixture(params=['vba', 'tfce', 'cet', 'glow'])
    def ana(self, request):
        """One recipe per arm under test."""
        if request.param == 'vba':
            return AnalysisVBA(n_perm_fwer=30)
        if request.param == 'tfce':
            return AnalysisVBA(n_perm_fwer=30, tfce_flag=True)
        if request.param == 'cet':
            return AnalysisCET(n_perm_fwer=30)
        return AnalysisGLOW(n_perm_fwer=25, n_perm_inner=25)

    def test_same_family_every_arm(self, exp, ana):
        """215 of 216 voxels, whichever method is fit."""
        assert exp.y.shape[2] == 215
        assert exp.num_vox_dropped == 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ana.fit(exp)
        assert ana.pval is not None

    def test_fit_finds_nothing_to_repair(self, exp, ana):
        """No statistic comes back non-finite once the screen has run."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ana.fit(exp)
        stat = getattr(ana, 'stat', None)
        if stat is not None:
            assert np.isfinite(stat).all()
