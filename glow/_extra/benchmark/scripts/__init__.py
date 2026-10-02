"""Command-line harnesses that read the benchmark's records, or probe them.

The sweeps themselves are driven from python -m glow._extra.benchmark; these
are the steps around them, each runnable as its own module:

  - figure_csv exports one tidy CSV per manuscript figure, straight off the
    records, and detection_stats prints the derived numbers the manuscript's
    detection section quotes beyond what those figures draw.
  - oracle_vs_k draws the one figure the records alone cannot, since it needs
    a fitted Ward tree rather than a scored leaf.
  - repro_probe and fp32_drift are the evidence behind rerun.md's account of
    what does and does not reproduce bit for bit: the first perturbs what a
    second machine changes, the second prices the device dtype.
  - paper_records exports the deposit; validate_records recomputes a seeded
    random sample of it within a time budget and diffs each leaf against its
    record.

Each takes --help, and none of them writes to the cache or the records (the
builders are called unwrapped), so an ad-hoc run cannot rewrite a benchmark
cell's provenance.

    python -m glow._extra.benchmark.scripts.figure_csv --out-dir figure_csv
"""
