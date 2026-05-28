"""glow: hierarchical, region-based effect discovery in imaging data.

General Linear models Optimized with Ward's method. Combines MANCOVA
statistics with permutation FWER control over a Ward tree of regions.
"""
import glow.analysis
import glow.effect
import glow.experiment
import glow.graph
import glow.mask
