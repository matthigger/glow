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
import pickle
import warnings

import numpy as np
import pytest

import glow.graph
from glow.analysis import _glow_split, draws
from glow.analysis._glow_split import AnalysisGLOWSplit
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.mancova import decompose
from glow.experiment.exper import Experiment, ExperimentScaled


NUM_IMG = 40

# Tolerance for a fit held against a rebuild through draws.cpu_reliable.
# The fit draws through the batched kernel (draws.cpu_summary), so these are
# two independent implementations of the same statistic and agree to fp64
# round-off, not bitwise -- they were bitwise while the fit took the anchor
# itself. Nothing the cases below claim is a 13th-digit effect: getting the
# fold, the moments or the standardization wrong moves them structurally,
# which is why the checks survive the looser bound. Cell-by-cell agreement
# of the two backends is pinned in test_draws.py, not inferred here.
RTOL_ANCHOR = 1e-7
ATOL_ANCHOR = 1e-9

# Tolerance between the two reductions of one backend's own chunks:
# cpu_summary's streaming Chan accumulators against summarize_draws over the
# materialized matrix. Same cells in a different summation order, so this is
# pure fp64 round-off and far tighter than RTOL_ANCHOR -- kept separate so a
# real drift between the streaming and materializing paths still fails.
RTOL_REDUCE = 1e-12
ATOL_REDUCE = 1e-14


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
    return AnalysisGLOWSplit(**kw)


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

    Deliberately through cpu_reliable, which is not the backend the fit
    takes: that makes every comparison against this matrix a second
    implementation's opinion rather than a re-run of the same code, at the
    cost of holding it to round-off (RTOL_ANCHOR / ATOL_ANCHOR).
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
    real = _glow_split.cluster
    monkeypatch.setattr(
        _glow_split, 'cluster',
        lambda exp, **kw: (calls.append(exp), real(exp, **kw))[1])

    _ana().fit(_exp())
    assert len(calls) == 1, f'Ward ran {len(calls)} times, expected once'


def test_tree_is_built_on_the_segmentation_fold_only(monkeypatch):
    """Ward sees fold A, and fold A alone -- never the whole cohort."""
    seen = []
    real = _glow_split.cluster
    monkeypatch.setattr(
        _glow_split, 'cluster',
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
    np.testing.assert_allclose(ana.llr, draws[0], rtol=RTOL_ANCHOR,
                               atol=ATOL_ANCHOR, equal_nan=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        np.testing.assert_allclose(ana.mu, np.nanmean(draws, axis=0),
                                   rtol=RTOL_ANCHOR, atol=ATOL_ANCHOR,
                                   equal_nan=True)


# ---------- the one draw matrix ---------------------------------------------
def test_only_the_observed_row_is_kept():
    """The fit keeps row 0 and the moments, never the matrix itself.

    llr and fwer.stat_obs are copies, not rows sliced out of it: a view
    would hold the whole (n_perm_fwer + 1, num_reg) matrix alive through
    .base -- gigabytes at full-brain num_vox, for one row of it.

    keep_stat is the one way to hold it, and it is off here.
    """
    exp = _exp()
    ana = _ana(n_perm_fwer=8).fit(exp)

    num_reg = 2 * exp.y.shape[2] - 1
    assert ana.llr.shape == (num_reg,)
    assert ana.fwer.stat_obs.shape == (num_reg,)
    assert ana.llr.base is None
    assert ana.fwer.stat_obs.base is None
    assert ana.stat is None


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

    np.testing.assert_allclose(ana.mu, mu_all, rtol=RTOL_ANCHOR,
                               atol=ATOL_ANCHOR, equal_nan=True)
    np.testing.assert_allclose(ana.std, std_all, rtol=RTOL_ANCHOR,
                               atol=ATOL_ANCHOR, equal_nan=True)
    # excluding row 0 would move them, so this is a real constraint
    assert not np.allclose(ana.mu, mu_null_only, equal_nan=True)


def test_z_is_the_shared_standardization():
    """GLOW z-scores through Analysis.z_score_stat, not a local copy."""
    exp = _exp()
    ana = _ana().fit(exp)
    z, _, _ = AnalysisGLOWSplit.z_score_stat(_draws(ana, exp))
    np.testing.assert_allclose(ana.fwer.stat_obs, z[0], rtol=RTOL_ANCHOR,
                               atol=ATOL_ANCHOR, equal_nan=True)

    reg_active = ana.size >= ana.min_vox
    np.testing.assert_allclose(ana.fwer.max_stat,
                               np.nanmax(z[:, reg_active], axis=1),
                               rtol=RTOL_ANCHOR, atol=ATOL_ANCHOR)


def test_max_stat_has_one_entry_per_draw():
    ana = _ana(n_perm_fwer=8).fit(_exp())
    assert ana.fwer.max_stat.shape == (9,)


def test_inactive_regions_have_no_pval():
    """min_vox keeps small regions out of the comparison set entirely."""
    ana = _ana(min_vox=4).fit(_exp())
    small = ana.size < 4
    assert np.isnan(ana.fwer.pval[small]).all()
    assert np.isfinite(ana.fwer.pval[~small]).any()


# ---------- keep_stat: the same matrix, held rather than dropped -----------
def test_keep_stat_holds_the_whole_matrix():
    """.stat is the (n_perm_fwer + 1, num_reg) matrix, not a summary."""
    exp = _exp()
    ana = _ana(n_perm_fwer=8, keep_stat=True).fit(exp)
    assert ana.stat.shape == (9, 2 * exp.y.shape[2] - 1)


def test_keep_stat_is_the_matrix_the_summary_came_from():
    """Row 0 is llr and the columns are mu / std -- to the last bit.

    The point of keeping the matrix is that the summary can be read back
    off it, so a viewer showing a region's draws is showing the null that
    region's own z was measured against, not a second sample of it.
    """
    exp = _exp()
    ana = _ana(keep_stat=True).fit(exp)

    # a below-min_vox region is NaN in every row, so nanmean/nanstd warn
    # on it -- the column is meant to stay NaN (as z_score_stat notes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.nanmean(ana.stat, axis=0)
        std = np.nanstd(ana.stat, axis=0, ddof=1)

    for got, want in ((ana.llr, ana.stat[0]), (ana.mu, mu),
                      (ana.std, std)):
        np.testing.assert_allclose(got, want, rtol=0, atol=0,
                                   equal_nan=True)


def test_keep_stat_matches_the_independent_rebuild():
    """The kept matrix is the recipe's, cell for cell.

    Held to RTOL_ANCHOR because keep_stat forms the matrix through
    draws.cpu_batched while the rebuild goes through draws.cpu_reliable --
    the same cross-implementation comparison the constant is named for.
    """
    exp = _exp()
    ana = _ana(keep_stat=True).fit(exp)
    np.testing.assert_allclose(ana.stat, _draws(ana, exp),
                               rtol=RTOL_ANCHOR, atol=ATOL_ANCHOR,
                               equal_nan=True)


def test_keep_stat_changes_no_result():
    """A diagnostic, not a knob: no statistic moves beyond round-off.

    llr and mu are bitwise: row 0 and the chunk means are the same
    reduction either way. std is not, because keep_stat swaps
    cpu_summary's streaming Chan accumulators for summarize_draws over the
    kept matrix, and the two sum in different orders -- so std and the
    z-scores dividing by it are held to RTOL_REDUCE. What the fit reports
    -- the p-values and the discoveries -- stays exact.
    """
    exp = _exp()
    off = _ana().fit(exp)
    on = _ana(keep_stat=True).fit(exp)

    for name in ('llr', 'mu', 'size'):
        np.testing.assert_allclose(getattr(off, name), getattr(on, name),
                                   rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(off.std, on.std, rtol=RTOL_REDUCE,
                               atol=ATOL_REDUCE, equal_nan=True)
    for name in ('stat_obs', 'max_stat'):
        np.testing.assert_allclose(getattr(off.fwer, name),
                                   getattr(on.fwer, name),
                                   rtol=RTOL_REDUCE, atol=ATOL_REDUCE,
                                   equal_nan=True)
    np.testing.assert_allclose(off.fwer.pval, on.fwer.pval, rtol=0, atol=0,
                               equal_nan=True)
    np.testing.assert_array_equal(off.fwer.reg_sig, on.fwer.reg_sig)


def test_keep_stat_is_not_a_recipe_field():
    """It keys no cache: same recipe, same repr, whether on or off.

    Storage is not a result. In RECORD_FIELDS it would invalidate every
    cached benchmark record the first time anyone wanted to look at a
    null (see AnalysisGLOWSplit's docstring, and Analysis.fit on n_jobs/gpu).
    """
    assert 'keep_stat' not in AnalysisGLOWSplit.RECORD_FIELDS
    assert repr(_ana()) == repr(_ana(keep_stat=True))


# ---------- saving a kept matrix is loud, not silent ------------------------
def test_pickling_a_kept_stat_warns():
    """The size of the thing being written is announced before it is."""
    exp = _exp()
    ana = _ana(keep_stat=True).fit(exp)

    with pytest.warns(UserWarning, match='keep_stat'):
        pickle.dumps(ana)


def test_the_warning_names_the_matrix():
    """It reports the shape, dtype and footprint, not just that it is big."""
    exp = _exp()
    ana = _ana(keep_stat=True).fit(exp)

    with pytest.warns(UserWarning) as record:
        pickle.dumps(ana)

    msg = str(record[0].message)
    assert str(ana.stat.shape) in msg and str(ana.stat.dtype) in msg
    assert 'GiB' in msg


def test_an_ordinary_fit_pickles_quietly():
    """Nothing kept, nothing to warn about -- the default path is silent."""
    exp = _exp()
    ana = _ana().fit(exp)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pickle.dumps(ana)


def test_an_unfit_recipe_pickles_quietly():
    """joblib hashes the recipe to key a cache; that must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pickle.dumps(_ana(keep_stat=True))


def test_the_matrix_survives_the_round_trip():
    """The warning is a notice, not a drop: the viewer bundle needs it.

    The viewer reads .stat off a pickled bundle, so silently omitting it
    here would produce a bundle whose PERMUTATION panel cannot open.
    """
    exp = _exp()
    ana = _ana(keep_stat=True).fit(exp)

    with pytest.warns(UserWarning):
        blob = pickle.dumps(ana)
    back = pickle.loads(blob)

    np.testing.assert_allclose(back.stat, ana.stat, rtol=0, atol=0,
                               equal_nan=True)
    assert back.keep_stat is True


# ---------- the split is part of the recipe ---------------------------------
def test_split_knobs_are_recipe_fields():
    """frac_segment / split_seed reach RECORD_FIELDS, so they key the cache."""
    assert 'frac_segment' in AnalysisGLOWSplit.RECORD_FIELDS
    assert 'split_seed' in AnalysisGLOWSplit.RECORD_FIELDS
    assert 'frac_segment=0.5' in repr(_ana())


def test_no_inner_perm_knob():
    """One tree needs no inner null, so n_perm_inner is not a knob here.

    It belongs to AnalysisGLOW, whose tree changes per outer perm; asking
    this arm for it names the wrong arm and fails loudly.
    """
    assert 'n_perm_inner' not in AnalysisGLOWSplit.RECORD_FIELDS
    with pytest.raises(TypeError):
        AnalysisGLOWSplit(n_perm_fwer=4, n_perm_inner=8)


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


def test_fit_records_the_realized_partition():
    """img_segment names the images the tree was built on, group or not.

    A grouped split cannot be redrawn from frac_segment and split_seed --
    the labels are a fit argument the analysis does not keep -- so the mask
    is the only record of which images chose the tree.
    """
    exp = _exp()
    group = np.repeat(np.arange(NUM_IMG // 4), 4)

    for split_group in (None, group):
        ana = _ana().fit(exp, split_group=split_group)
        exp_seg, exp_test = exp.split_img(frac_segment=ana.frac_segment,
                                          seed=ana.split_seed,
                                          group=split_group)

        np.testing.assert_array_equal(exp_seg.x, exp.x[:, ana.img_segment])
        np.testing.assert_array_equal(exp_test.x, exp.x[:, ~ana.img_segment])


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


# ---------- the CPU backends against each other ------------------------------
# fit takes the batched kernel; cpu_anchor=True takes the slow per-region
# walk. Nothing downstream of the draws differs, so a whole fit must land on
# the same conclusions either way -- and that is what makes the anchor worth
# keeping in the tree at 35x the cost.
def test_the_anchor_fits_to_the_same_conclusions():
    """fit() and fit(cpu_anchor=True) agree on every reported output.

    p-values are compared exactly, not to a tolerance: they are counts of
    null draws at or above the observed, so round-off may not move one
    without something being wrong about the ordering.
    """
    exp = _exp()
    fast = _ana().fit(exp)
    anchor = _ana().fit(exp, cpu_anchor=True)

    np.testing.assert_array_equal(anchor.children, fast.children)
    np.testing.assert_allclose(anchor.llr, fast.llr, rtol=RTOL_ANCHOR,
                               atol=ATOL_ANCHOR, equal_nan=True)
    np.testing.assert_allclose(anchor.fwer.max_stat, fast.fwer.max_stat,
                               rtol=RTOL_ANCHOR, atol=ATOL_ANCHOR)
    np.testing.assert_allclose(anchor.fwer.pval, fast.fwer.pval, rtol=0,
                               atol=0, equal_nan=True)
    assert [e.reg_idx for e in anchor.effect_list] == \
           [e.reg_idx for e in fast.effect_list]


def test_the_anchor_returns_self():
    ana = _ana()
    assert ana.fit(_exp(), cpu_anchor=True) is ana


def test_the_anchor_refuses_an_explicit_device():
    """Two backends were named at once, so the fit says so rather than pick."""
    with pytest.raises(ValueError, match='cannot also honour'):
        _ana().fit(_exp(), cpu_anchor=True, gpu=True)


def test_the_anchor_outranks_an_auto_device():
    """'auto' is a default, cpu_anchor is an instruction; no raise."""
    ana = _ana()
    assert ana.fit(_exp(), cpu_anchor=True, gpu='auto') is ana


# ---------- the device argument ----------------------------------------------
# Which device 'auto' lands on depends on what is visible, so this only
# pins that the fit completes and keeps fit's contract either way. The A/B
# equivalence of the two backends lives in test_fit_gpu.py, which needs a
# device to say anything.
def test_auto_device_fits_and_returns_self():
    ana = _ana()
    assert ana.fit(_exp(), gpu='auto') is ana
