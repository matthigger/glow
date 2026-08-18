"""The per-perm architecture: a tree per outer perm, an inner null per tree.

These pin the properties that make AnalysisGLOW the arm it is, and that
separate it from AnalysisGLOWSplit (test_glow_split.py):

  - Ward runs once per outer permutation, on every image;
  - outer perm k clusters exp.permute(k), so the null trees are the
    permuted data's own;
  - each tree's regions are standardized against inner FL draws of that
    same permuted data, the identity draw included;
  - one entry of the FWER null per outer perm, entry 0 the observed;
  - the recipe carries n_perm_inner and no fold.

Run:
    ~/venv_glow/bin/pytest test/analysis/test_glow_perm.py -v
"""
import pickle
import warnings

import numpy as np
import pytest

import glow.graph
from glow.analysis import _glow, draws
from glow.analysis._glow import AnalysisGLOW
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.mancova import decompose
from glow.experiment.exper import Experiment, ExperimentScaled


NUM_IMG = 20

# Tolerance for a fit held against a rebuild through draws.cpu_reliable: the
# fit draws through the batched kernel, so the two are independent
# implementations of the same statistic and agree to fp64 round-off rather
# than bitwise (the same constants test_glow_split.py names).
RTOL_ANCHOR = 1e-7
ATOL_ANCHOR = 1e-9


def _exp(seed=0, num_img=NUM_IMG, shape=(6, 6, 6), b=2):
    """A float64 experiment (see test_glow_split._exp on why fp64)."""
    exp = Experiment.from_gauss(b=b, a=2, seed=seed, shape=shape,
                                num_img=num_img)
    return Experiment(x=exp.x.astype(np.float64), contrast=exp.contrast,
                      mask_idx=exp.mask_idx, y=exp.y.astype(np.float64))


def _ana(**over):
    kw = dict(n_perm_fwer=4, n_perm_inner=3, min_vox=2,
              cluster_mode=ClusterMode.FOCUS)
    kw.update(over)
    return AnalysisGLOW(**kw)


def _rebuild_outer(ana, exp, k):
    """Rebuild outer perm k's tree and inner draw matrix from the recipe.

    The fit keeps only the observed tree's arrays and one max-z per outer
    perm, so every check that needs a null tree rebuilds it here --
    deliberately through cpu_reliable, which is not the backend the fit
    takes, so each comparison is a second implementation's opinion.

    Returns:
        children (np.array): (num_reg - num_vox, 2) outer perm k's tree
        matrix (np.array): (n_perm_inner + 1, num_reg) its inner draws,
            row 0 the identity draw of the permuted data
    """
    exp_scaled = ExperimentScaled.from_exp(exp)
    exp_k = exp_scaled.permute(k) if k else exp_scaled
    children = cluster(exp_k, mode=ana.cluster_mode)
    q0, q1, _ = decompose(x=exp_k.x, contrast=exp_k.contrast)
    matrix = draws.cpu_reliable(
        exp=exp_k, base_seed=0, n_perm=ana.n_perm_inner + 1, q0=q0, q1=q1,
        children=children, min_vox=ana.min_vox)
    return children, matrix


def _max_z(matrix, size, min_vox):
    """Reduce one tree's inner matrix to the max-z its outer perm reports."""
    z, _, _ = AnalysisGLOW.z_score_stat(matrix)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmax(z[0][size >= min_vox]))


# ---------- one tree per outer perm, on every image --------------------------
def test_ward_runs_once_per_outer_perm(monkeypatch):
    """The tree is rebuilt inside every draw -- n_perm_fwer + 1 of them."""
    calls = []
    real = _glow.cluster
    monkeypatch.setattr(
        _glow, 'cluster',
        lambda exp, **kw: (calls.append(exp), real(exp, **kw))[1])

    ana = _ana(n_perm_fwer=4).fit(_exp())
    assert len(calls) == ana.n_perm_fwer + 1


def test_ward_sees_every_image(monkeypatch):
    """No fold is held out: the tree is built from the whole cohort.

    This is what the arm buys, so it is pinned rather than left to the
    absence of a split knob.
    """
    seen = []
    real = _glow.cluster
    monkeypatch.setattr(
        _glow, 'cluster',
        lambda exp, **kw: (seen.append(exp), real(exp, **kw))[1])

    _ana().fit(_exp())
    assert [e.y.shape[1] for e in seen] == [NUM_IMG] * len(seen)


def test_observed_tree_is_the_unpermuted_one():
    """children is cluster(exp) -- outer perm 0 is the observed data."""
    exp = _exp()
    ana = _ana().fit(exp)
    children, _ = _rebuild_outer(ana, exp, 0)
    np.testing.assert_array_equal(ana.children, children)


def test_null_trees_come_from_the_permuted_data(monkeypatch):
    """Outer perm k clusters exp.permute(k), not exp.

    A null tree built on the observed data would make every draw share the
    observed tree, which is the anti-conservative case the arm is not.
    """
    seen = []
    real = _glow.cluster
    monkeypatch.setattr(
        _glow, 'cluster',
        lambda exp, **kw: (seen.append(exp.y.copy()), real(exp, **kw))[1])

    exp = _exp()
    _ana(n_perm_fwer=2).fit(exp)

    exp_scaled = ExperimentScaled.from_exp(exp)
    np.testing.assert_allclose(seen[0], exp_scaled.y)
    for k in (1, 2):
        np.testing.assert_allclose(seen[k], exp_scaled.permute(k).y)
        assert not np.allclose(seen[k], exp_scaled.y)


# ---------- the inner null standardizes its own tree ------------------------
def test_observed_stats_are_the_inner_null_of_the_observed_tree():
    """llr, mu and std are row 0 and the columns of one inner matrix."""
    exp = _exp()
    ana = _ana().fit(exp)
    _, matrix = _rebuild_outer(ana, exp, 0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0, ddof=1)

    for got, want in ((ana.llr, matrix[0]), (ana.mu, mu), (ana.std, std)):
        np.testing.assert_allclose(got, want, rtol=RTOL_ANCHOR,
                                   atol=ATOL_ANCHOR, equal_nan=True)


def test_observed_row_contributes_to_the_inner_moments():
    """The identity draw enters mu / std, as z_score_stat requires."""
    exp = _exp()
    ana = _ana().fit(exp)
    _, matrix = _rebuild_outer(ana, exp, 0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu_null_only = np.nanmean(matrix[1:], axis=0)

    assert not np.allclose(ana.mu, mu_null_only, equal_nan=True)


def test_llr_matches_an_independent_backend():
    """The observed LLR is the permuted-free statistic on its own tree."""
    exp = _exp()
    ana = _ana().fit(exp)

    exp_scaled = ExperimentScaled.from_exp(exp)
    q0, q1, _ = decompose(x=exp_scaled.x, contrast=exp_scaled.contrast)
    llr, size = glow.graph.compute_llr_batched(
        exp_scaled, children=ana.children, q0=q0, q1=q1)

    ok = ana.size >= ana.min_vox
    np.testing.assert_allclose(ana.llr[ok], llr[ok], rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(ana.size, size)


def test_n_perm_inner_sets_the_inner_draw_count():
    """More inner draws move the moments -- the knob reaches the null."""
    exp = _exp()
    few = _ana(n_perm_inner=3).fit(exp)
    many = _ana(n_perm_inner=9).fit(exp)

    np.testing.assert_allclose(few.llr, many.llr, rtol=0, atol=0,
                               equal_nan=True)
    assert not np.allclose(few.mu, many.mu, equal_nan=True)


# ---------- one FWER entry per outer perm -----------------------------------
def test_max_stat_has_one_entry_per_outer_perm():
    ana = _ana(n_perm_fwer=4).fit(_exp())
    assert ana.fwer.max_stat.shape == (5,)


def test_every_null_entry_is_its_own_perms_max_z():
    """Entry k is the max z of outer perm k's own tree, entry 0 observed.

    The whole null, rebuilt one perm at a time: this is the claim the FWER
    p-values rest on, and the one place the per-perm comparison set (each
    tree's own size >= min_vox) is checked.
    """
    exp = _exp()
    ana = _ana(n_perm_fwer=3).fit(exp)

    for k in range(ana.n_perm_fwer + 1):
        children, matrix = _rebuild_outer(ana, exp, k)
        _, region_l, region_h = glow.graph.build_dfs_preorder(
            children=children, num_vox=exp.y.shape[2])
        want = _max_z(matrix, region_h - region_l, ana.min_vox)
        np.testing.assert_allclose(ana.fwer.max_stat[k], want,
                                   rtol=RTOL_ANCHOR, atol=ATOL_ANCHOR)


def test_observed_z_is_the_shared_standardization():
    """fwer.stat_obs is z_score_stat's row 0, not a local copy."""
    exp = _exp()
    ana = _ana().fit(exp)
    _, matrix = _rebuild_outer(ana, exp, 0)
    z, _, _ = AnalysisGLOW.z_score_stat(matrix)

    np.testing.assert_allclose(ana.fwer.stat_obs, z[0], rtol=RTOL_ANCHOR,
                               atol=ATOL_ANCHOR, equal_nan=True)
    np.testing.assert_allclose(ana.fwer.max_stat[0],
                               np.nanmax(z[0][ana.size >= ana.min_vox]),
                               rtol=RTOL_ANCHOR, atol=ATOL_ANCHOR)


def test_inactive_regions_have_no_pval():
    """min_vox keeps small regions out of the comparison set entirely."""
    ana = _ana(min_vox=4).fit(_exp())
    small = ana.size < 4
    assert np.isnan(ana.fwer.pval[small]).all()
    assert np.isfinite(ana.fwer.pval[~small]).any()


# ---------- the recipe: an inner null, and no fold --------------------------
def test_inner_perm_is_a_recipe_field():
    """n_perm_inner keys the cache: it changes what the fit computes."""
    assert 'n_perm_inner' in AnalysisGLOW.RECORD_FIELDS
    assert 'n_perm_inner=3' in repr(_ana())


def test_no_split_knobs():
    """There is no fold here, so naming one fails rather than being ignored."""
    for knob in ('frac_segment', 'split_seed'):
        with pytest.raises(TypeError):
            AnalysisGLOW(n_perm_fwer=4, **{knob: .5})


def test_prune_knobs_are_recipe_fields():
    """The rule is shared with the split arm, so both arms must key on it."""
    for knob in ('prune_rule', 'prune_lam', 'prune_exp_n_eff'):
        assert knob in AnalysisGLOW.RECORD_FIELDS
    assert 'prune_rule=greedy' in repr(_ana())


def test_the_rule_reaches_the_selection(monkeypatch):
    """_discover is shared, so the per-perm arm honours the rule as well."""
    seen = {}

    def spy(rule, *, sig_reg_list, children, stat, lam, exp_n_eff):
        seen.update(rule=rule, lam=lam, exp_n_eff=exp_n_eff)
        return [], {}

    monkeypatch.setattr(_glow, 'prune_by_rule', spy)
    _ana(prune_rule='dp', prune_lam=2.5).fit(_exp())
    assert seen == dict(rule='dp', lam=2.5, exp_n_eff=None)

    ana = _ana().fit(_exp())
    assert not hasattr(ana, 'img_segment')


def test_accepts_an_already_scaled_experiment():
    """No split means no pre-scaling leak, so a scaled exp is fine.

    AnalysisGLOWSplit refuses one (its folds would share a transform); this
    arm has no folds, and from_exp passes an ExperimentScaled through.
    """
    exp = _exp()
    raw = _ana().fit(exp)
    scaled = _ana().fit(ExperimentScaled.from_exp(exp))
    np.testing.assert_allclose(raw.llr, scaled.llr, rtol=0, atol=0,
                               equal_nan=True)


# ---------- keep_stat: the observed tree's own matrix ------------------------
def test_keep_stat_holds_the_observed_inner_matrix():
    """.stat is (n_perm_inner + 1, num_reg): one tree's null, not the fit's."""
    exp = _exp()
    ana = _ana(n_perm_inner=3, keep_stat=True).fit(exp)
    assert ana.stat.shape == (4, 2 * exp.y.shape[2] - 1)


def test_keep_stat_is_the_matrix_the_moments_came_from():
    """Row 0 is llr and the columns are mu / std -- to the last bit."""
    exp = _exp()
    ana = _ana(keep_stat=True).fit(exp)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.nanmean(ana.stat, axis=0)
        std = np.nanstd(ana.stat, axis=0, ddof=1)

    for got, want in ((ana.llr, ana.stat[0]), (ana.mu, mu), (ana.std, std)):
        np.testing.assert_allclose(got, want, rtol=0, atol=0, equal_nan=True)


def test_keep_stat_changes_no_pvalue():
    """A diagnostic, not a knob: the reported decisions are untouched."""
    exp = _exp()
    off = _ana().fit(exp)
    on = _ana(keep_stat=True).fit(exp)

    np.testing.assert_allclose(off.llr, on.llr, rtol=0, atol=0,
                               equal_nan=True)
    np.testing.assert_allclose(off.fwer.pval, on.fwer.pval, rtol=0, atol=0,
                               equal_nan=True)
    np.testing.assert_array_equal(off.fwer.reg_sig, on.fwer.reg_sig)


def test_keep_stat_is_not_a_recipe_field():
    """Storage is not a result: same recipe, same repr, either way."""
    assert 'keep_stat' not in AnalysisGLOW.RECORD_FIELDS
    assert repr(_ana()) == repr(_ana(keep_stat=True))


def test_pickling_a_kept_stat_warns():
    """The size of the thing being written is announced before it is."""
    ana = _ana(keep_stat=True).fit(_exp())
    with pytest.warns(UserWarning, match='keep_stat'):
        pickle.dumps(ana)


def test_an_ordinary_fit_pickles_quietly():
    ana = _ana().fit(_exp())
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pickle.dumps(ana)


# ---------- execution knobs change nothing ----------------------------------
def test_fit_is_deterministic_and_returns_self():
    exp = _exp()
    a = _ana()
    assert a.fit(exp) is a
    b = _ana().fit(exp)
    np.testing.assert_allclose(a.fwer.max_stat, b.fwer.max_stat, rtol=0,
                               atol=0)
    np.testing.assert_allclose(a.fwer.pval, b.fwer.pval, rtol=0, atol=0,
                               equal_nan=True)


def test_n_jobs_changes_no_result():
    """Outer perm k is seeded by k, so the workers cannot reorder anything."""
    exp = _exp()
    serial = _ana().fit(exp, n_jobs=1)
    parallel = _ana().fit(exp, n_jobs=2)

    np.testing.assert_allclose(serial.fwer.max_stat, parallel.fwer.max_stat,
                               rtol=0, atol=0)
    np.testing.assert_allclose(serial.mu, parallel.mu, rtol=0, atol=0,
                               equal_nan=True)


def test_auto_device_fits_and_returns_self():
    """Which device 'auto' lands on depends on what is visible; both fit."""
    ana = _ana()
    assert ana.fit(_exp(), gpu='auto') is ana
