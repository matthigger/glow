"""Effect-discovery analyses: voxel-based (VBA, CET) and GLOW.

Each analysis fits an Experiment, discovers significant regions, and
assigns FWER-controlled p-values. Every arm ends at one max-stat
permutation test (MaxStatPerm, in fwer.py) and shares the stat walk,
standardization and connected-component discovery on the Analysis ABC;
subpackage modules supply the per-region statistics and the search
strategy.
"""

from .fwer import MaxStatPerm
from ._base import Analysis, AnalysisVoxel
from .vba import AnalysisVBA, AnalysisCET, DEFAULT_CET_CFT_PVAL
from ._glow import AnalysisGLOW
from ._fit_gpu import GpuConfig

__all__ = ['MaxStatPerm', 'Analysis', 'AnalysisVoxel', 'AnalysisVBA',
           'AnalysisCET', 'AnalysisGLOW', 'DEFAULT_CET_CFT_PVAL', 'GpuConfig']
