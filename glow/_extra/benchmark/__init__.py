"""Paper benchmark suite: build imaging Experiments, measure, and score them.

A catalogue (config) maps each figure to a grid of data / effect /
leaf-function kwargs the driver sweeps: data (data) builds a clean Experiment
(white-noise or HCP) and plants synthetic effect(s), then a leaf (run) fits an
Analysis and scores it (score). Every build and leaf call is disk-memoised and
captured by a shared Recorder (recorder), so a run joins one provenance DAG;
results recovers which of its leaves each figure's cache owns, make_csv
exports them as one tidy CSV per figure, and plot draws them. file holds the
on-disk paths, hcp the reference dataset.

Run it with python -m glow._extra.benchmark (see __main__).
"""
from .file import *
from .recorder import Recorder
