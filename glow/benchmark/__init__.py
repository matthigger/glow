"""Benchmark engine: data sources, trial cache, and drivers.

Wires together the pieces that run a benchmark sweep: hashable,
memoising DataSource builders (data), a trial-iteration + result-IO
cache (trial_cache), local / AWS drivers, and on-disk result helpers
(file).
"""
from .file import *
from .trial_cache import TrialCache
