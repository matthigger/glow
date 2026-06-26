"""Paper benchmarks: trial functions plus the trial-cache catalogue.

CACHE_BY_LABEL is intentionally NOT re-exported here. Building it
constructs every catalogue data source eagerly (DataSourceHCP reads the
HCP dataset in its __init__), so re-exporting it would force that data
load onto anything that imports a sibling submodule -- including the AWS
worker, which only needs .run to unpickle run_ana and has no HCP data on
the container. Import CACHE_BY_LABEL straight from .config where the
catalogue is actually needed.
"""
from .run import run_ana, run_mancova
