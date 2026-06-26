"""Benchmark engine (mid-overhaul).

The trial-iteration + caching layer (the hashable/frozen-dataclass
DataSource builders, the TrialCache, and the local/AWS drivers) is being
replaced by a joblib.Memory-backed approach on the benchmark-overhaul
branch -- those modules have been removed. What remains is the recorder
(provenance + timing capture) and the on-disk result helpers (file); the
new driver/cache layer will be wired back in here as it is rebuilt.
"""
from .file import *
from .recorder import Recorder
