"""Tests for glow._extra.benchmark.driver: the data x effect x analysis sweep.

The stage functions (data_factory / effect_factory / run_ana) and the shared
cache + recorder machinery are covered by test_data.py / test_run.py /
test_recorder.py, so these cover only what is novel to ``drive``: that it runs
the full cartesian product of its two upstream grids fanned across the fnc's
kwargs grid (one score per cell), that it threads each stage's output into the
next (the clean exp into the plant, the planted exp + its mask into fnc as the
target), and that every cell lands in the shared provenance DAG as a complete
data -> plant -> score chain.

Fresh seeds keep every cell a cache miss, so each really runs and records (a hit
would neither recompute nor record); the grids are tiny WGN + cheap VBA so the
sweep is fast.
"""
import random

import pytest
from joblib import parallel_config

from glow._extra.benchmark import data
from glow._extra.benchmark.driver import drive
from glow._extra.benchmark.run import run_ana
from glow.analysis import AnalysisVBA
from glow.effect import ExtenterSphere


def _fresh_seed() -> int:
    """A seed unlikely to be in the on-disk cache, so the cell is a miss."""
    return random.randrange(2 ** 31)


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Mirror the shared recorder's per-hash files to a tmp dir, not the real one."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)


_N_VOX_EFF = 10
_SCORE_KEYS = {'num_vox', 'min_pval', 'n_pred', 'pred', 'target'}


def _data_grid(n):
    """n WGN data cells, each a distinct fresh seed (so each misses)."""
    return [dict(source='wgn', shape=(5, 5, 5), b=2, num_img=16, a=2,
                 seed=_fresh_seed()) for _ in range(n)]


def _effect_grid(n):
    """n effect cells (distinct llr), each a fixed-size sphere plant."""
    return [dict(effect_llr=0.05 + i, extenter_cls=ExtenterSphere,
                 n_vox=_N_VOX_EFF, seed=0)
            for i in range(n)]


def _ana_grid(n):
    """n analysis cells (distinct n_perm_fwer, so distinct recipes / keys)."""
    return [dict(ana=AnalysisVBA(n_perm_fwer=15 + i)) for i in range(n)]


# ---------------------------------------------------------------------------
# return contract: one score dict per (data, effect, analysis) cell
# ---------------------------------------------------------------------------

class TestContract:
    def test_one_score_per_product_cell(self):
        scores = drive(_data_grid(2), _effect_grid(2), _ana_grid(2), run_ana)
        assert len(scores) == 2 * 2 * 2
        assert all(_SCORE_KEYS <= set(s) for s in scores)

    def test_non_rectangular_axes(self):
        # the product is n_data * n_effect * n_ana, not a square
        scores = drive(_data_grid(3), _effect_grid(1), _ana_grid(2), run_ana)
        assert len(scores) == 3 * 1 * 2

    def test_empty_grid_yields_no_scores(self):
        # an empty axis collapses the whole product to nothing
        assert drive(_data_grid(2), [], _ana_grid(2), run_ana) == []
        assert drive([], _effect_grid(2), _ana_grid(2), run_ana) == []


# ---------------------------------------------------------------------------
# wiring: each stage's output threads into the next
# ---------------------------------------------------------------------------

class TestWiring:
    def test_planted_mask_reaches_run_ana_as_target(self):
        # the effect_factory mask is passed as run_ana's target, so every cell's
        # target accounts for exactly the planted support (tp hit + fn missed).
        scores = drive(_data_grid(2), _effect_grid(1), _ana_grid(1), run_ana)
        for s in scores:
            t = s['target']
            assert t['tp'] + t['fn'] == _N_VOX_EFF


# ---------------------------------------------------------------------------
# provenance: every cell joins the shared recorder DAG as a full chain
# ---------------------------------------------------------------------------

class TestNullEffectCell:
    """A ``None`` effect cell is the null path: no plant, empty target."""

    def test_none_effect_targets_nothing(self):
        # nothing is planted, so run_ana sees the clean exp and an empty target:
        # every voxel is background (no tp to hit, no fn to miss)
        scores = drive(_data_grid(2), [None], _ana_grid(1), run_ana)
        assert len(scores) == 2 * 1
        for s in scores:
            assert s['target']['tp'] == 0 and s['target']['fn'] == 0

    def test_none_effect_records_no_plant_node(self):
        # the null row chains run_ana straight to the build -- effect_factory is
        # skipped entirely, so it records nothing
        data.RECORDER.records.clear()
        drive(_data_grid(2), [None], _ana_grid(1), run_ana)
        fns = [r['function'] for r in data.RECORDER.records.values()]
        assert fns.count('effect_factory') == 0
        assert fns.count('run_ana') == 2


class TestProvenanceDAG:
    def test_one_full_chain_row_per_cell(self):
        data.RECORDER.records.clear()
        data_grid = _data_grid(2)
        drive(data_grid, _effect_grid(2), _ana_grid(2), run_ana)

        df = data.RECORDER.flatten_to_df()
        # one run_ana leaf per cell; the build + plant feed each, so they are
        # ancestors (not leaves) and every row carries the whole chain
        assert len(df) == 2 * 2 * 2
        assert (df['run_ana.function'] == 'run_ana').all()
        assert (df['effect_factory.function'] == 'effect_factory').all()
        assert (df['data_factory_wgn.function'] == 'data_factory_wgn').all()
        # the swept data seeds chain onto the run_ana rows (only via shared-exp
        # DAG edges), so both data cells are represented
        assert set(df['data_factory_wgn.in.seed']) == {
            d['seed'] for d in data_grid}

    def test_each_stage_recorded_per_distinct_cell(self):
        # with fresh seeds every cell misses, so the record count per stage is
        # exactly its grid: data once per data cell, the plant once per
        # (data, effect), the fit once per (data, effect, analysis).
        data.RECORDER.records.clear()
        drive(_data_grid(2), _effect_grid(2), _ana_grid(2), run_ana)

        fns = [r['function'] for r in data.RECORDER.records.values()]
        assert fns.count('data_factory_wgn') == 2
        assert fns.count('effect_factory') == 2 * 2
        assert fns.count('run_ana') == 2 * 2 * 2


# ---------------------------------------------------------------------------
# parallel: split by data cell (n_jobs != 1). The threading backend shares this
# process, so it honours the monkeypatched recorder folder + in-memory records
# (loky workers re-import the real singletons); it still exercises the grouping
# and RECORDER.load -- the loky-only pickling is covered by the payload-size test.
# ---------------------------------------------------------------------------

class TestParallel:
    def test_parallel_matches_serial(self):
        # same grid, same results: parallel only changes who runs each data cell,
        # not what each computes (order is preserved data-cell-wise).
        grid, eff, ana = _data_grid(3), _effect_grid(2), _ana_grid(2)
        with parallel_config(backend='threading'):
            par = drive(grid, eff, ana, run_ana, n_jobs=2)
        ser = drive(grid, eff, ana, run_ana, n_jobs=1)
        assert len(par) == 3 * 2 * 2
        assert par == ser

    def test_parallel_records_full_chain(self):
        # every parallel cell still lands in the shared DAG: one full
        # data -> plant -> run_ana row per (data, effect, analysis) cell.
        data.RECORDER.records.clear()
        grid = _data_grid(3)
        with parallel_config(backend='threading'):
            drive(grid, _effect_grid(2), _ana_grid(2), run_ana, n_jobs=2)

        df = data.RECORDER.flatten_to_df()
        assert len(df) == 3 * 2 * 2
        assert (df['run_ana.function'] == 'run_ana').all()
        assert (df['effect_factory.function'] == 'effect_factory').all()
        assert set(df['data_factory_wgn.in.seed']) == {d['seed'] for d in grid}

    def test_parallel_task_payload_stays_small(self):
        # regression: the leaf fnc is pickled into each task by value. Because the
        # recorder snapshots its records (it retains no live Experiments), a task
        # payload stays small even after a heavy record exists -- not the ~1 MB a
        # pinned Experiment would add (see glow._extra.benchmark.recorder).
        import cloudpickle
        from joblib import delayed

        from glow._extra.benchmark.driver import _run_data_cell

        # parks a record in the shared recorder (an 829 KB array -> a snapshot)
        data.data_factory_wgn(shape=(12, 12, 12), b=3, num_img=40, a=2,
                              seed=_fresh_seed())
        payload = cloudpickle.dumps(delayed(_run_data_cell)(
            dict(source='wgn', shape=(5, 5, 5), b=2, num_img=16, a=2, seed=0),
            [None], _ana_grid(1), run_ana))
        assert len(payload) < 100_000
