"""Effect-discovery analyses: voxel-based (VBA, CET) and GLOW.

Each analysis fits an Experiment, discovers significant regions, and
assigns FWER-controlled p-values. The shared machinery (FWER p-values
from a permutation null, connected-component effect discovery) lives in
the Analysis ABC; subpackage modules supply the per-region statistics
and the search strategy.
"""

from ._base import Analysis, AnalysisVoxel
from .vba import AnalysisVBA, AnalysisCET, DEFAULT_CET_CFT_PVAL
from ._glow import AnalysisGLOW
# imported after AnalysisGLOW: _fit_gpu reads _glow for the inner-seed
# block and the outer-perm reduction
from ._fit_gpu import GpuConfig

__all__ = ['Analysis', 'AnalysisVoxel', 'AnalysisVBA', 'AnalysisCET',
           'AnalysisGLOW', 'DEFAULT_CET_CFT_PVAL', 'GpuConfig']
