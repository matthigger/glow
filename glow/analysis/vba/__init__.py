"""Voxel-based analysis: VBA, TFCE, and cluster-extent thresholding.

Per-voxel MANCOVA statistics with permutation-based FWER control. VBA
applies optional TFCE enhancement (Smith & Nichols 2009); CET thresholds
clusters against a permutation null of max cluster sizes.
"""

from ._vba import AnalysisVBA
from ._cet import AnalysisCET, DEFAULT_CET_CFT_PVAL

__all__ = ['AnalysisVBA', 'AnalysisCET', 'DEFAULT_CET_CFT_PVAL']
