"""Tests for glow.benchmark.runner helpers (_write_result, RunMancovaVba._emit_variant)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from glow.benchmark.file import OUT, ERROR
from glow.benchmark.runner import (_write_result, _score_and_emit,
                                    _run_shared_voxel_walk, _dispatch_shared,
                                    RunMancovaVba, RunAna)
from glow.experiment.exper import Experiment
from glow.analysis import AnalysisVBA, AnalysisCET, AnalysisVoxel


def _make_config(tmp_path):
    """Build a minimal config-like object sufficient for _write_result."""
    fake_runner = SimpleNamespace(hash=lambda _cfg, _label: 'abc123')
    config = SimpleNamespace(
        folder=tmp_path,
        runner=fake_runner,
    )
    return config


# -----------------------------------------------------------------------
# _write_result
# -----------------------------------------------------------------------

class TestWriteResult:
    def test_creates_json_file(self, tmp_path):
        config = _make_config(tmp_path)
        d = {'label': 'test', 'value': 42}
        uuid_str = _write_result(config, d)

        file_out = tmp_path / OUT / f'{uuid_str}_result.json'
        assert file_out.exists()

        with open(file_out) as f:
            loaded = json.load(f)
        assert loaded['label'] == 'test'
        assert loaded['value'] == 42
        assert loaded['uuid'] == uuid_str

    def test_subfolder_error(self, tmp_path):
        config = _make_config(tmp_path)
        d = {'error': 'something went wrong'}
        uuid_str = _write_result(config, d, subfolder=ERROR)

        file_out = tmp_path / ERROR / f'{uuid_str}_result.json'
        assert file_out.exists()

        with open(file_out) as f:
            loaded = json.load(f)
        assert loaded['error'] == 'something went wrong'

    def test_returns_uuid(self, tmp_path):
        config = _make_config(tmp_path)
        d = {'x': 1}
        uuid_str = _write_result(config, d)

        assert isinstance(uuid_str, str)
        assert len(uuid_str) == 8

        # uuid in the dict should match
        assert d['uuid'] == uuid_str

        # uuid in the file should match too
        file_out = tmp_path / OUT / f'{uuid_str}_result.json'
        with open(file_out) as f:
            loaded = json.load(f)
        assert loaded['uuid'] == uuid_str

    def test_unique_uuids(self, tmp_path):
        config = _make_config(tmp_path)
        uuids = [_write_result(config, {'i': i}) for i in range(10)]
        assert len(set(uuids)) == 10


# -----------------------------------------------------------------------
# _run_variant
# -----------------------------------------------------------------------

class TestRunVariant:
    """Integration test: build a tiny experiment, compute stats, run variant."""

    @pytest.fixture()
    def setup(self, tmp_path):
        """Create a small WGN experiment and compute a stat matrix."""
        exp = Experiment.from_gauss(num_img=10, shape=(3, 3), b=2, seed=0)
        # impose a trivial effect so we have a non-None effect
        from glow.effect import ExtenterSphere, EffectSynthetic
        exp, effect = EffectSynthetic.impose(
            exp, seed=0, extenter=ExtenterSphere(radius=1), effect_llr=0.3)

        config = _make_config(tmp_path)

        # compute voxel-wise stat (VBA-style: no tree, just voxels)
        from glow.analysis.mancova import get_wilks
        n_perm = 10
        num_vox = exp.y.shape[2]
        stat = np.full((n_perm + 1, num_vox), np.nan)
        for k in range(n_perm + 1):
            _exp = exp.permute(k) if k else exp
            stat[k, :] = AnalysisVoxel.get_stat_perm_multi(
                _exp, [get_wilks], children=None)[get_wilks]

        return exp, effect, config, stat, get_wilks

    def test_run_variant_vba(self, setup):
        exp, effect, config, stat, get_wilks = setup
        import time
        RunMancovaVba._emit_variant(
            config, effect, exp, get_wilks, 'VBA-wilks',
            stat, 0.05, 0.0, {}, time.time(), AnalysisVBA)

        # verify a result JSON was written
        result_files = list((config.folder / OUT).glob('*_result.json'))
        assert len(result_files) == 1

        with open(result_files[0]) as f:
            d = json.load(f)
        assert d['label'] == 'VBA-wilks'
        assert d['Analysis'] == 'AnalysisVBA'
        assert 'dice' in d

    def test_run_variant_cet(self, setup):
        exp, effect, config, stat, get_wilks = setup
        import time

        null_pool = stat[1:, :].ravel()
        cft_pval = 0.01
        cft = np.quantile(null_pool, 1 - cft_pval)

        RunMancovaVba._emit_variant(
            config, effect, exp, get_wilks, 'CET-wilks',
            stat, 0.05, 0.0, {}, time.time(), AnalysisCET,
            cft=cft, cft_pval=cft_pval, z_flag=False)

        result_files = list((config.folder / OUT).glob('*_result.json'))
        assert len(result_files) == 1

        with open(result_files[0]) as f:
            d = json.load(f)
        assert d['label'] == 'CET-wilks'
        assert d['Analysis'] == 'AnalysisCET'


# -----------------------------------------------------------------------
# RunAna shared voxel-stat walk
# -----------------------------------------------------------------------

class TestRunAnaShared:
    """Sharing the per-perm voxel stat walk across VBA/VBA-TFCE/CET must
    produce the same pval / effect arrays as running each analysis
    standalone."""

    @pytest.fixture()
    def exp(self):
        from glow.experiment.exper import Experiment
        from glow.effect import ExtenterSphere, EffectSynthetic
        exp = Experiment.from_gauss(num_img=20, shape=(6, 6), b=2, seed=0)
        exp, _ = EffectSynthetic.impose(
            exp, seed=0, extenter=ExtenterSphere(radius=1), effect_llr=0.3)
        return exp

    def test_dispatch_matches_standalone(self, exp):
        from glow.analysis.mancova import get_wilks
        n_perm = 8
        common = dict(n_perm_fwer=n_perm, alpha_fwer=0.05,
                      get_stat=get_wilks)
        members = [
            ('VBA',      AnalysisVBA, {**common}),
            ('VBA-z',    AnalysisVBA, {**common, 'z_flag': True}),
            ('VBA-TFCE', AnalysisVBA, {**common, 'tfce_flag': True}),
            ('CET',      AnalysisCET, {**common, 'cft_pval': 0.05}),
            ('CET-z',    AnalysisCET, {**common, 'cft_pval': 0.05,
                                       'z_flag': True}),
        ]

        stat_by_fn, _walk_time = _run_shared_voxel_walk(exp, members)
        # Only one stat fn → one entry; matrix shape (n_perm+1, num_vox).
        assert list(stat_by_fn.keys()) == [get_wilks]
        assert stat_by_fn[get_wilks].shape == (n_perm + 1, exp.y.shape[2])

        for label, Ana, kw in members:
            ana_shared = _dispatch_shared(
                Ana=Ana, exp=exp, ana_kw=kw, stat_by_fn=stat_by_fn)
            ana_baseline = Ana(exp=exp, **kw)
            np.testing.assert_array_equal(
                ana_shared.pval, ana_baseline.pval,
                err_msg=f'pval mismatch for {label}')

    def test_multi_stat_walk_runs_once(self, exp, monkeypatch):
        """If the dict spans multiple stat fns, get_stat_perm_multi must
        be called once per permutation (n_perm+1 times total), not per
        (stat, perm)."""
        from glow.analysis import AnalysisVoxel
        from glow.analysis.mancova import get_wilks, get_pillai

        calls = []
        orig = AnalysisVoxel.get_stat_perm_multi.__func__

        def spy(cls, exp, get_stat_list, children=None):
            calls.append(tuple(get_stat_list))
            return orig(cls, exp, get_stat_list, children=children)

        monkeypatch.setattr(AnalysisVoxel, 'get_stat_perm_multi',
                            classmethod(spy))

        n_perm = 5
        members = [
            ('VBA-wilks',  AnalysisVBA, {'n_perm_fwer': n_perm,
                                         'get_stat': get_wilks}),
            ('VBA-pillai', AnalysisVBA, {'n_perm_fwer': n_perm,
                                         'get_stat': get_pillai}),
            ('CET-wilks',  AnalysisCET, {'n_perm_fwer': n_perm,
                                         'get_stat': get_wilks,
                                         'cft_pval': 0.05}),
        ]
        _run_shared_voxel_walk(exp, members)

        # one call per permutation (incl. observed), each with both fns
        assert len(calls) == n_perm + 1
        assert all(set(c) == {get_wilks, get_pillai} for c in calls)

    def test_runana_groups_shareable(self, exp, tmp_path):
        """End-to-end: RunAna.run with a mixed dict produces one result
        row per label, with correct Analysis types."""
        from glow.analysis.mancova import get_wilks

        ana_dict = {
            'VBA':      (AnalysisVBA, {'n_perm_fwer': 8,
                                       'get_stat': get_wilks}),
            'VBA-TFCE': (AnalysisVBA, {'n_perm_fwer': 8,
                                       'get_stat': get_wilks,
                                       'tfce_flag': True}),
            'CET':      (AnalysisCET, {'n_perm_fwer': 8,
                                       'get_stat': get_wilks,
                                       'cft_pval': 0.05}),
        }
        runner = RunAna(ana_dict)

        # Build a stand-in config.  get_exp_eff returns (exp, effect);
        # we already have both via the fixture + a fresh effect.
        from glow.effect import ExtenterSphere, EffectSynthetic
        _, effect = EffectSynthetic.impose(
            exp, seed=1, extenter=ExtenterSphere(radius=1), effect_llr=0.3)

        fake_runner = SimpleNamespace(hash=lambda _c, _l: 'abc123')
        config = SimpleNamespace(
            folder=tmp_path,
            runner=fake_runner,
            error_save=False,
            get_exp_eff=lambda **_kw: (exp, effect),
        )

        runner.run(config)

        result_files = list((tmp_path / OUT).glob('*_result.json'))
        assert len(result_files) == 3
        labels = set()
        for path in result_files:
            with open(path) as f:
                d = json.load(f)
            labels.add(d['label'])
            assert d['Analysis'] in {'AnalysisVBA', 'AnalysisCET'}
        assert labels == {'VBA', 'VBA-TFCE', 'CET'}


# -----------------------------------------------------------------------
# _sanitize_adjusted_stat / DEFAULT_CET_CFT_PVAL
# -----------------------------------------------------------------------

class TestMagicConstants:
    def test_sanitize_adjusted_stat(self):
        from glow.analysis import _sanitize_adjusted_stat
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        result = _sanitize_adjusted_stat(arr)
        # nan/posinf -> 0 (treated as no evidence);
        # neginf -> nan (invalid, propagates to NaN p-value)
        expected = np.array([1.0, 0.0, 0.0, np.nan, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_default_cet_cft_pval(self):
        from glow.analysis import DEFAULT_CET_CFT_PVAL
        assert DEFAULT_CET_CFT_PVAL == 0.0001


# -----------------------------------------------------------------------
# short_uuid
# -----------------------------------------------------------------------

class TestShortUuid:
    def test_length(self):
        from glow.benchmark.file import short_uuid
        assert len(short_uuid()) == 8

    def test_unique(self):
        from glow.benchmark.file import short_uuid
        uuids = {short_uuid() for _ in range(100)}
        assert len(uuids) == 100
