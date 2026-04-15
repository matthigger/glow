"""Tests for glow.benchmark.run helpers (_write_result, _run_variant)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from glow.benchmark.file import OUT, ERROR
from glow.benchmark.run import _write_result, _run_variant, _score_and_emit
from glow.experiment.exper import Experiment
from glow.analysis import AnalysisVBA, AnalysisCET, Analysis


def _make_config(tmp_path, detail_save=False):
    """Build a minimal config-like object sufficient for _write_result."""
    config = SimpleNamespace(
        folder=tmp_path,
        detail_save=detail_save,
        _config_hash=lambda: 'abc123',
        _config_hash_for_label=lambda label: 'abc123',
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
        from glow.effect import ExtenterSphere
        exp, effect = exp.impose_effect(
            seed=0, extenter=ExtenterSphere(radius=1), effect_llr=0.3)

        config = _make_config(tmp_path)

        # compute voxel-wise stat (VBA-style: no tree, just voxels)
        from glow.analysis.mancova import get_wilks
        stat = Analysis.get_stat_perm_multi(
            exp, [get_wilks], n_perm=10, children=None)[get_wilks]

        return exp, effect, config, stat, get_wilks

    def test_run_variant_vba(self, setup):
        exp, effect, config, stat, get_wilks = setup
        import time
        _run_variant(
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

        _run_variant(
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
# _sanitize_adjusted_stat / DEFAULT_CET_CFT_PVAL
# -----------------------------------------------------------------------

class TestMagicConstants:
    def test_sanitize_adjusted_stat(self):
        from glow.analysis import _sanitize_adjusted_stat
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        result = _sanitize_adjusted_stat(arr)
        expected = np.array([1.0, 0.0, 0.0, -30.0, 2.0])
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
