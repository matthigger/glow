from ._base import Analysis, DEFAULT_CET_CFT_PVAL, _sanitize_adjusted_stat
from ._vba import AnalysisVBA
from ._cet import AnalysisCET
from ._glow import AnalysisGLOW, get_best_model, _BEST_MODELS_PATH

__all__ = ['Analysis', 'AnalysisVBA', 'AnalysisCET', 'AnalysisGLOW',
           'get_best_model', 'DEFAULT_CET_CFT_PVAL', '_sanitize_adjusted_stat']

# Pickle compatibility: ensure instances pickled before the package split
# can still be unpickled. Old pickles store __module__ = 'glow.experiment.analysis'.
for _cls in (Analysis, AnalysisVBA, AnalysisCET, AnalysisGLOW):
    _cls.__module__ = 'glow.experiment.analysis'
