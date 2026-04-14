"""Tests for aws/worker.py functions that run locally (no AWS needed).

Covers:
- get_array_info: recursive numpy array discovery
- process_permutation: fidelity against AnalysisGLOW._process_permutation
- Synthesis path: worker's reimplemented regression + finalization matches
  a local AnalysisGLOW run
"""

import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from glow.aws.worker import process_permutation, get_array_info
except ImportError:
    pytest.skip("worker dependencies not available", allow_module_level=True)

from glow.experiment import Experiment
from glow.experiment.analysis import AnalysisGLOW
from glow.experiment.mancova import get_llr


# ---------------------------------------------------------------------------
# get_array_info
# ---------------------------------------------------------------------------

class TestGetArrayInfo:

    def test_single_array(self):
        arr = np.zeros((3, 4))
        info = get_array_info(arr)
        assert len(info) == 1
        assert info[0]['shape'] == (3, 4)
        assert info[0]['dtype'] == 'float64'
        assert info[0]['size_mb'] == pytest.approx(3 * 4 * 8 / 1024 / 1024)

    def test_nested_dict(self):
        d = {'a': np.zeros(10), 'b': {'c': np.ones((2, 3))}}
        info = get_array_info(d)
        assert len(info) == 2
        names = {i['name'] for i in info}
        assert 'a' in names
        assert 'b.c' in names

    def test_list_of_arrays(self):
        lst = [np.zeros(3), np.ones(5)]
        info = get_array_info(lst)
        assert len(info) == 2

    def test_object_attributes(self):
        class Obj:
            pass
        obj = Obj()
        obj.arr = np.zeros(5)
        info = get_array_info(obj)
        arr_info = [i for i in info if 'arr' in i['name']]
        assert len(arr_info) >= 1

    def test_cycle_detection(self):
        d = {'x': np.zeros(1)}
        d['self'] = d
        info = get_array_info(d)
        # should not infinite loop; should find the array
        assert any(i['name'] == 'x' for i in info)

    def test_max_depth(self):
        d = {'a': {'b': {'c': {'d': np.zeros(1)}}}}
        info = get_array_info(d, max_depth=2)
        # array at depth 4 should not be reached
        assert len(info) == 0

    def test_non_container(self):
        info = get_array_info(42)
        assert info == []

    def test_prefix_propagation(self):
        d = {'inner': np.zeros(1)}
        info = get_array_info(d, prefix='root')
        assert info[0]['name'] == 'root.inner'


# ---------------------------------------------------------------------------
# process_permutation fidelity
# ---------------------------------------------------------------------------

class TestProcessPermutationFidelity:
    """Worker's process_permutation must match AnalysisGLOW.rerun_permutation."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=30, seed=0)

    def _check_perm(self, perm_idx):
        worker_r = process_permutation(self.exp, {}, perm_idx)
        local_r = AnalysisGLOW.rerun_permutation(self.exp, perm_idx,
                                                   get_stat=get_llr)

        np.testing.assert_array_equal(
            worker_r['children'], local_r['children'],
            err_msg=f'children mismatch at perm_idx={perm_idx}')
        np.testing.assert_allclose(
            np.asarray(worker_r['stat'], dtype=float),
            np.asarray(local_r['stat'], dtype=float),
            err_msg=f'stat mismatch at perm_idx={perm_idx}')
        np.testing.assert_array_equal(
            worker_r['size'], local_r['size'],
            err_msg=f'size mismatch at perm_idx={perm_idx}')

    def test_observed(self):
        """perm_idx=0 (unpermuted) should match."""
        self._check_perm(0)

    def test_permuted(self):
        """Permuted indices should match."""
        self._check_perm(1)
        self._check_perm(5)

    def test_stat_length(self):
        """Worker stat list should have one entry per region."""
        r = process_permutation(self.exp, {}, 0)
        num_vox = self.exp.y.shape[2]
        num_internal = r['children'].shape[0]
        assert len(r['stat']) == num_vox + num_internal

    def test_custom_get_stat(self):
        """Worker should respect get_stat from ana_kwargs."""
        from glow.experiment.mancova import get_wilks
        r_wilks = process_permutation(
            self.exp, {'get_stat': get_wilks}, 0)
        r_llr = process_permutation(self.exp, {}, 0)

        # same tree structure (data is identical)
        np.testing.assert_array_equal(r_wilks['children'],
                                       r_llr['children'])
        # but different stat values (Wilks != LLR)
        assert not np.allclose(
            np.asarray(r_wilks['stat']),
            np.asarray(r_llr['stat']))


# ---------------------------------------------------------------------------
# Synthesis fidelity
# ---------------------------------------------------------------------------

class TestSynthesisFidelity:
    """Worker synthesis path must produce identical results to local GLOW."""

    def test_synthesis_matches_local(self):
        """Replicate worker.run_synthesis_mode logic, compare to local run."""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=50, seed=42)
        n_perm_fwer = 10
        n_perm_fwer_size_adjust = 5
        alpha_fwer = 0.05
        min_size = 1
        model = 'power_law'

        # 1. Run local AnalysisGLOW with perm_dir for reference
        perm_dir = tempfile.mkdtemp(prefix='glow_test_synth_')
        try:
            ref = AnalysisGLOW(
                exp, n_perm_fwer=n_perm_fwer,
                n_perm_fwer_size_adjust=n_perm_fwer_size_adjust,
                alpha_fwer=alpha_fwer, min_size=min_size,
                perm_dir=perm_dir, size_adjust_model=model)

            # 2. Load saved permutation results
            results = {}
            for f in sorted(Path(perm_dir).glob('*_result.pkl')):
                pidx = int(f.name.split('_')[0])
                with open(f, 'rb') as fh:
                    results[pidx] = pickle.load(fh)

            # 3. Replicate synthesis path (same as worker.py, no S3)
            n_perm = n_perm_fwer + n_perm_fwer_size_adjust
            fit_start = n_perm_fwer + 1
            n_expected = n_perm + 1

            r0 = results[0]
            stat_0 = np.asarray(r0['stat'], dtype=float)
            children_0 = r0['children']
            size_0 = np.asarray(r0['size'], dtype=float)

            # accumulate regression from fit permutations only
            XtX, Xty = None, None
            for pidx in range(1, n_expected):
                r = results[pidx]
                stat_p = np.asarray(r['stat'], dtype=float)
                size_p = np.asarray(r['size'], dtype=float)
                if pidx >= fit_start:
                    XtX, Xty = AnalysisGLOW.accumulate_regression(
                        size_p, stat_p, model, XtX, Xty)

            mu_fn, _, beta = AnalysisGLOW.fit_size_regression_online(
                XtX, Xty, model, get_llr)

            # compute adjusted max-stats for FWER permutations
            reg_active = size_0 >= min_size
            stat_max_list = []
            for pidx in range(n_perm_fwer + 1):
                r = results[pidx]
                stat_p = np.asarray(r['stat'], dtype=float)
                size_p = np.asarray(r['size'], dtype=float)
                adj = stat_p - mu_fn(size_p)
                adj = np.nan_to_num(adj, nan=0.0, posinf=0.0,
                                    neginf=-30.0)
                if reg_active.any():
                    stat_max_list.append(
                        float(np.nanmax(adj[reg_active])))
                else:
                    stat_max_list.append(float('-inf'))
            stat_max_sorted = np.sort(stat_max_list)

            # finalize
            ana = AnalysisGLOW.from_precomputed(
                exp=exp, get_stat=get_llr,
                adj_model=model, adj_beta=beta)
            ana._finalize_analysis(
                exp, n_perm_fwer, stat_0, size_0, children_0,
                mu_fn, stat_max_sorted, alpha_fwer, min_size)

            # 4. Compare synthesis result to local reference
            np.testing.assert_array_equal(
                ana.pval, ref.pval,
                err_msg='p-values differ between synthesis and local')
            np.testing.assert_array_equal(
                ana.stat, ref.stat,
                err_msg='stat arrays differ')
            np.testing.assert_array_equal(
                ana.size, ref.size,
                err_msg='size arrays differ')
            np.testing.assert_array_equal(
                ana.children, ref.children,
                err_msg='children differ')
            assert set(ana.sig_reg_list) == set(ref.sig_reg_list), \
                'sig_reg_list differs'
            assert len(ana.effect_list) == len(ref.effect_list), \
                f'effect count: synth={len(ana.effect_list)}, ' \
                f'local={len(ref.effect_list)}'

        finally:
            shutil.rmtree(perm_dir, ignore_errors=True)

    def test_synthesis_with_effect(self):
        """Synthesis should also match under H1 (effect present)."""
        from glow.effect import ExtenterSphere
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=50, seed=7)
        exp, _ = exp.impose_effect(seed=7,
                                    extenter=ExtenterSphere(radius=2),
                                    effect_llr=0.5)
        n_perm_fwer = 10
        n_perm_fwer_size_adjust = 5
        alpha_fwer = 0.1
        model = 'power_law'

        perm_dir = tempfile.mkdtemp(prefix='glow_test_synth_h1_')
        try:
            ref = AnalysisGLOW(
                exp, n_perm_fwer=n_perm_fwer,
                n_perm_fwer_size_adjust=n_perm_fwer_size_adjust,
                alpha_fwer=alpha_fwer, perm_dir=perm_dir,
                size_adjust_model=model)

            # load and replay
            results = {}
            for f in sorted(Path(perm_dir).glob('*_result.pkl')):
                pidx = int(f.name.split('_')[0])
                with open(f, 'rb') as fh:
                    results[pidx] = pickle.load(fh)

            n_perm = n_perm_fwer + n_perm_fwer_size_adjust
            fit_start = n_perm_fwer + 1

            r0 = results[0]
            stat_0 = np.asarray(r0['stat'], dtype=float)
            children_0 = r0['children']
            size_0 = np.asarray(r0['size'], dtype=float)

            XtX, Xty = None, None
            for pidx in range(1, n_perm + 1):
                r = results[pidx]
                if pidx >= fit_start:
                    XtX, Xty = AnalysisGLOW.accumulate_regression(
                        np.asarray(r['size'], dtype=float),
                        np.asarray(r['stat'], dtype=float),
                        model, XtX, Xty)

            mu_fn, _, beta = AnalysisGLOW.fit_size_regression_online(
                XtX, Xty, model, get_llr)

            reg_active = size_0 >= 1
            stat_max_list = []
            for pidx in range(n_perm_fwer + 1):
                r = results[pidx]
                adj = (np.asarray(r['stat'], dtype=float)
                       - mu_fn(np.asarray(r['size'], dtype=float)))
                adj = np.nan_to_num(adj, nan=0.0, posinf=0.0,
                                    neginf=-30.0)
                stat_max_list.append(
                    float(np.nanmax(adj[reg_active]))
                    if reg_active.any() else float('-inf'))
            stat_max_sorted = np.sort(stat_max_list)

            ana = AnalysisGLOW.from_precomputed(
                exp=exp, get_stat=get_llr,
                adj_model=model, adj_beta=beta)
            ana._finalize_analysis(
                exp, n_perm_fwer, stat_0, size_0, children_0,
                mu_fn, stat_max_sorted, alpha_fwer, 1)

            np.testing.assert_array_equal(ana.pval, ref.pval)
            assert len(ana.effect_list) == len(ref.effect_list)

        finally:
            shutil.rmtree(perm_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# process_permutation output format
# ---------------------------------------------------------------------------

class TestProcessPermutationFormat:
    """Verify the output dict has the fields the synthesis code expects."""

    exp = Experiment.from_gauss(a=2, b=1, shape=(3, 3), num_img=20, seed=0)

    def test_required_keys(self):
        r = process_permutation(self.exp, {}, 0)
        assert set(r.keys()) >= {'perm_idx', 'children', 'stat', 'size'}

    def test_stat_asarray_works(self):
        """Synthesis code calls np.asarray(r['stat'], dtype=float)."""
        r = process_permutation(self.exp, {}, 0)
        arr = np.asarray(r['stat'], dtype=float)
        assert arr.ndim == 1
        assert np.isfinite(arr).all() or np.isnan(arr).any()

    def test_size_matches_node_sum(self):
        """size should equal node_sum(ones, children)."""
        import glow.graph
        r = process_permutation(self.exp, {}, 0)
        num_vox = self.exp.y.shape[2]
        expected = glow.graph.node_sum(
            np.ones(num_vox, dtype=int), r['children'])
        np.testing.assert_array_equal(r['size'], expected)
