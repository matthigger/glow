import pytest

from glow.analysis import AnalysisGLOW


# ---------------------------------------------------------------------------
# Fix 4: _estimate_perm_sec raises when runtime model missing
# ---------------------------------------------------------------------------

class TestEstimatePermSec:
    """Magic fallback removed -- missing model must raise FileNotFoundError."""

    def test_raises_when_model_missing(self, monkeypatch):
        from glow.experiment.exper import Experiment
        import glow.benchmark.runtime as rt_mod

        # stub load_runtime_model to simulate missing model file
        # (signature: load_runtime_model(analysis_type, platform))
        monkeypatch.setattr(rt_mod, 'load_runtime_model',
                            lambda *a, **kw: None)

        exp = Experiment.from_gauss(a=2, b=1, shape=(4, 4),
                                    num_img=20, seed=0)
        with pytest.raises(FileNotFoundError,
                           match='runtime model not found'):
            AnalysisGLOW._estimate_perm_sec(exp)

    def test_error_points_to_profile_cli(self, monkeypatch):
        from glow.experiment.exper import Experiment
        import glow.benchmark.runtime as rt_mod

        monkeypatch.setattr(rt_mod, 'load_runtime_model',
                            lambda *a, **kw: None)

        exp = Experiment.from_gauss(a=2, b=1, shape=(4, 4),
                                    num_img=20, seed=1)
        with pytest.raises(FileNotFoundError,
                           match=r'python -m glow\.benchmark\.runtime'):
            AnalysisGLOW._estimate_perm_sec(exp)