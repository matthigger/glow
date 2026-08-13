"""The split architecture: one Ward tree, on a fold the statistics never see.

These pin the properties the FWER argument rests on, which are structural
rather than numerical and so would survive any amount of green elsewhere:

  - the tree is built once, on the segmentation fold alone;
  - every statistic comes from the test fold;
  - the draw matrix's row 0 is the observed draw, and it contributes to
    the column moments that standardize every row -- the asymmetry that
    broke VBA+z (see Analysis.z_score_stat);
  - the split is reproducible from the recipe.

test_fwer_calibration.py covers the distributional consequence (the
observed draw's rank is uniform among the permuted draws); this file
covers the plumbing that makes it true.

Run:
    ~/venv_glow/bin/pytest test/analysis/test_glow_split.py -v
"""
import warnings

import numpy as np
import pytest

import glow.graph
from glow.analysis import _glow, draws
from glow.analysis._glow import AnalysisGLOW
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.mancova import decompose
from glow.experiment.exper import Experiment, ExperimentScaled


NUM_IMG = 40


def _exp(seed=0, num_img=NUM_IMG, shape=(6, 6, 6), b=2):
    """A float64 experiment.

    from_gauss samples float32. fp64 here because the checks below hold
    cpu_reliable against compute_llr_batched -- two independent
    implementations of the statistic, which agree to fp64 round-off but
    only to ~1e-3 in float32 (the same reason test_draws.py anchors
    its equivalence tests on fp64 preps).
    """
    exp = Experiment.from_gauss(b=b, a=2, seed=seed, shape=shape,
                                num_img=num_img)
    return Experiment(x=exp.x.astype(np.float64), contrast=exp.contrast,
                      mask_idx=exp.mask_idx, y=exp.y.astype(np.float64))


def _ana(**over):
    kw = dict(n_perm_fwer=8, min_vox=2, cluster_mode=ClusterMode.FOCUS,
              frac_segment=.5, split_seed=0)
    kw.update(over)
    return AnalysisGLOW(**kw)


def _test_fold(ana, exp):
    """Rebuild the fit's tree and test fold from the recipe alone."""
    exp_seg, exp_test = exp.split_img(frac_segment=ana.frac_segment,
                                      seed=ana.split_seed)
    children = cluster(ExperimentScaled.from_exp(exp_seg),
                       mode=ana.cluster_mode)
    return children, ExperimentScaled.from_exp(exp_test)


def _draws(ana, exp):
    """Rebuild a fit's whole draw matrix from the recipe alone.

    fit keeps only row 0 and the column moments -- the matrix is
    (n_perm_fwer + 1, num_reg) and too heavy to hold -- so the checks
    that need every row rebuild it here. That the rebuild reproduces the
    fit is itself pinned, by test_draws_come_from_the_test_fold.
    """
    children, exp_test = _test_fold(ana, exp)
    q0, q1, _ = decompose(x=exp_test.x, contrast=exp_test.contrast)
    return draws.cpu_reliable(
        exp=exp_test, base_seed=0, n_perm=ana.n_perm_fwer + 1,
        q0=q0, q1=q1, children=children, min_vox=ana.min_vox)


# ---------- the tree is built once, on the segmentation fold -----------------
def test_ward_runs_once(monkeypatch):
    """One tree per fit -- not one per outer perm, as before the split."""
    calls = []
    real = _glow.cluster
    monkeypatch.setattr(
        _glow, 'cluster',
        lambda exp, **kw: (calls.append(exp), real(exp, **kw))[1])

    _ana().fit(_exp())
    assert len(calls) == 1, f'Ward ran {len(calls)} times, expected once'


def test_tree_is_built_on_the_segmentation_fold_only(monkeypatch):
    """Ward sees fold A, and fold A alone -- never the whole cohort."""
    seen = []
    real = _glow.cluster
    monkeypatch.setattr(
        _glow, 'cluster',
        lambda exp, **kw: (seen.append(exp), real(exp, **kw))[1])

    _ana(frac_segment=.5).fit(_exp())
    assert seen[0].y.shape[1] == NUM_IMG // 2
    # and it is scaled on that fold's images alone
    assert isinstance(seen[0], ExperimentScaled)


def test_draws_come_from_the_test_fold():
    """The whole draw matrix reproduces on the test fold, from the recipe.

    The fit keeps no matrix to compare against, so this holds the rebuilt
    one against everything the fit did keep of it: row 0 exactly, and the
    column moments, which every row enters.
    """
    exp = _exp()
    ana = _ana().fit(exp)

    children, exp_test = _test_fold(ana, exp)
    np.testing.assert_array_equal(ana.children, children)
    assert exp_test.y.shape[1] == NUM_IMG - NUM_IMG // 2

    draws = _draws(ana, exp)
    np.testing.assert_allclose(ana.llr, draws[0], rtol=0, atol=0,
                               equal_nan=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        np.testing.assert_allclose(ana.mu, np.nanmean(draws, axis=0),
                                   rtol=0, atol=0, equal_nan=True)


# ---------- the one draw matrix ---------------------------------------------
def test_only_the_observed_row_is_kept():
    """The fit keeps row 0 and the moments, never the matrix itself.

    llr and fwer.stat_obs are copies, not rows sliced out of it: a view
    would hold the whole (n_perm_fwer + 1, num_reg) matrix alive through
    .base -- gigabytes at full-brain num_vox, for one row of it.
    """
    exp = _exp()
    ana = _ana(n_perm_fwer=8).fit(exp)

    num_reg = 2 * exp.y.shape[2] - 1
    assert ana.llr.shape == (num_reg,)
    assert ana.fwer.stat_obs.shape == (num_reg,)
    assert ana.llr.base is None
    assert ana.fwer.stat_obs.base is None
    assert not hasattr(ana, 'draws')


def test_observed_row_is_the_unpermuted_draw():
    """llr is permute(0) -- so it matches an independent LLR backend."""
    exp = _exp()
    ana = _ana(n_perm_fwer=8).fit(exp)

    children, exp_test = _test_fold(ana, exp)
    q0, q1, _ = decompose(x=exp_test.x, contrast=exp_test.contrast)
    llr, size = glow.graph.compute_llr_batched(
        exp_test, children=children, q0=q0, q1=q1)
    ok = ana.size >= ana.min_vox
    np.testing.assert_allclose(ana.llr[ok], llr[ok], rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(ana.size, size)


def test_observed_row_contributes_to_the_moments():
    """mu / std are taken over ALL rows, the observed one included.

    Standardizing row 0 by moments it did not contribute to is what broke
    VBA+z (inflated to ~11% at B=49); see Analysis.z_score_stat.
    """
    exp = _exp()
    ana = _ana().fit(exp)
    draws = _draws(ana, exp)
    # a region below min_vox is NaN in every row, so nanmean/nanstd warn
    # on it -- the column is meant to stay NaN (as z_score_stat notes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu_all = np.nanmean(draws, axis=0)
        std_all = np.nanstd(draws, axis=0, ddof=1)
        mu_null_only = np.nanmean(draws[1:], axis=0)

    np.testing.assert_allclose(ana.mu, mu_all, rtol=0, atol=0,
                               equal_nan=True)
    np.testing.assert_allclose(ana.std, std_all, rtol=0, atol=0,
                               equal_nan=True)
    # excluding row 0 would move them, so this is a real constraint
    assert not np.allclose(ana.mu, mu_null_only, equal_nan=True)


def test_z_is_the_shared_standardization():
    """GLOW z-scores through Analysis.z_score_stat, not a local copy."""
    exp = _exp()
    ana = _ana().fit(exp)
    z, _, _ = AnalysisGLOW.z_score_stat(_draws(ana, exp))
    np.testing.assert_allclose(ana.fwer.stat_obs, z[0], rtol=0, atol=0,
                               equal_nan=True)

    reg_active = ana.size >= ana.min_vox
    np.testing.assert_allclose(ana.fwer.max_stat,
                               np.nanmax(z[:, reg_active], axis=1),
                               rtol=0, atol=0)


def test_max_stat_has_one_entry_per_draw():
    ana = _ana(n_perm_fwer=8).fit(_exp())
    assert ana.fwer.max_stat.shape == (9,)


def test_inactive_regions_have_no_pval():
    """min_vox keeps small regions out of the comparison set entirely."""
    ana = _ana(min_vox=4).fit(_exp())
    small = ana.size < 4
    assert np.isnan(ana.fwer.pval[small]).all()
    assert np.isfinite(ana.fwer.pval[~small]).any()


# ---------- the split is part of the recipe ---------------------------------
def test_split_knobs_are_recipe_fields():
    """frac_segment / split_seed reach RECORD_FIELDS, so they key the cache."""
    assert 'frac_segment' in AnalysisGLOW.RECORD_FIELDS
    assert 'split_seed' in AnalysisGLOW.RECORD_FIELDS
    assert 'frac_segment=0.5' in repr(_ana())


def test_no_inner_perm_knob_survives():
    """The inner null is gone; a stale caller should fail loudly."""
    assert 'n_perm_inner' not in AnalysisGLOW.RECORD_FIELDS
    with pytest.raises(TypeError):
        AnalysisGLOW(n_perm_fwer=4, n_perm_inner=8)


def test_split_seed_changes_the_partition_and_so_the_tree():
    exp = _exp()
    a = _ana(split_seed=0).fit(exp)
    b = _ana(split_seed=7).fit(exp)
    assert not np.array_equal(a.children, b.children)


def test_split_group_is_plumbed_through():
    """fit(split_group=...) reaches split_img, so related images hold."""
    exp = _exp()
    group = np.repeat(np.arange(NUM_IMG // 4), 4)

    ana = _ana().fit(exp, split_group=group)
    exp_seg, _ = exp.split_img(frac_segment=ana.frac_segment,
                               seed=ana.split_seed, group=group)
    children = cluster(ExperimentScaled.from_exp(exp_seg),
                       mode=ana.cluster_mode)
    np.testing.assert_array_equal(ana.children, children)

    assert not np.array_equal(ana.children, _ana().fit(exp).children)


def test_refuses_an_already_scaled_experiment():
    """Scaling before the split would let fold B pick fold A's transform."""
    with pytest.raises(TypeError, match='[Ss]plit before scaling'):
        _ana().fit(ExperimentScaled.from_exp(_exp()))


def test_fit_is_deterministic_and_returns_self():
    exp = _exp()
    a = _ana()
    assert a.fit(exp) is a
    b = _ana().fit(exp)
    np.testing.assert_allclose(a.fwer.max_stat, b.fwer.max_stat, rtol=0,
                               atol=0)
    np.testing.assert_allclose(a.fwer.pval, b.fwer.pval, rtol=0, atol=0,
                               equal_nan=True)


# ---------- the device argument ----------------------------------------------
# Which device 'auto' lands on depends on what is visible, so this only
# pins that the fit completes and keeps fit's contract either way. The A/B
# equivalence of the two backends lives in test_fit_gpu.py, which needs a
# device to say anything.
def test_auto_device_fits_and_returns_self():
    ana = _ana()
    assert ana.fit(_exp(), gpu='auto') is ana
