"""Effect-discovery analyses: voxel-based (VBA, CET) and GLOW.

Each analysis fits an Experiment, discovers significant regions, and
assigns FWER-controlled p-values. Every arm ends at one max-stat
permutation test (MaxStatPerm, in fwer.py) and shares the stat walk,
standardization and connected-component discovery on the Analysis ABC;
subpackage modules supply the per-region statistics and the search
strategy.

GLOW comes in two arms over one Ward tree: AnalysisGLOWSplit builds that
tree on a held-out fold of the images, AnalysisGLOW rebuilds it inside
every permutation. _glow.py's docstring sets out what each one buys.
"""

from .fwer import MaxStatPerm
from ._base import Analysis, AnalysisVoxel
from .vba import AnalysisVBA, AnalysisCET, DEFAULT_CET_CFT_PVAL
from ._glow import AnalysisGLOW, AnalysisGLOWBase
from ._glow_split import AnalysisGLOWSplit
from ._fit_gpu import GpuConfig

__all__ = ['MaxStatPerm', 'Analysis', 'AnalysisVoxel', 'AnalysisVBA',
           'AnalysisCET', 'AnalysisGLOW', 'AnalysisGLOWSplit',
           'AnalysisGLOWBase', 'DEFAULT_CET_CFT_PVAL', 'GpuConfig']
