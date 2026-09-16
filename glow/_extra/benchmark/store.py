"""The benchmark's shared disk cache and provenance records.

One cache and one recorder for every memoised call in the suite -- the cell
realizations and every leaf measurement -- so a leaf is stored beside the
cell it measures and both write into one provenance DAG (see .recorder).
They live here rather than beside any one caller because neither belongs to
one: .cell realizes cells into them and .run measures into them.
"""

import joblib

from .file import get_path_cache, get_path_records
from .recorder import Recorder

# Compressed, which is what makes a cell payload cheap to store: a mostly
# empty boolean mask volume is long runs, so zlib packs the analysis and
# support masks to a few KB each where raw they are the whole grid. The cost
# lands on the incompressible leaves instead (a float array pays ~100 ms a
# write), which is nothing beside the fit that produced it.
MEMORY = joblib.Memory(get_path_cache(), verbose=0, compress=3)

# captures each call's inputs / output / timing for provenance (see Recorder).
# Keyed by joblib's args hash, mirrored to the records dir beside the cache.
# Every edge is declared: a consumer is passed its parent's uid, so nothing
# here hashes an array to discover lineage.
RECORDER = Recorder(folder=get_path_records())
