"""Export one tidy CSV per manuscript figure, straight off the records.

The reproduction artifact for a reader who wants the numbers rather than the
pipeline: every data-backed figure and table in the paper, one CSV each,
carrying the rows that figure actually draws. See
glow/_extra/benchmark/rerun.md for the figure table these fill.

Each export reuses the plot layer's own tidy function and its own row
filtering (plot._select_glow_arm, plot._split_by_secondary), so a CSV is the
frame the figure was drawn from rather than a re-derivation of it. Change how
a figure selects its rows and these follow, because they call the same code.

The per-trial grain is deliberate: a figure shows a mean and a band, and the
CSV under it holds the trials those summarise, so a reader can recompute the
band, re-derive a metric the paper did not draw, or check a single seed. Every
frame carries seed, so nothing is anonymised into an aggregate.

Two figures are excluded and one needs help:

  - the four TikZ schematics and the two effect-illustration galleries have no
    tabular data (they are drawings and image panels).
  - oracle_vs_k reads a fitted Ward tree, which the records do not store; its
    own script caches the curves it computes to a CSV, and this module copies
    that sidecar in when it is present (see the rerun guide).

Written twice: to --out-dir, and to a figure_csv/ directory inside the
records tree, so the CSVs travel with the records they were derived from
rather than having to be remembered separately. The recorder globs *.json
non-recursively, so the subdirectory is invisible to it. --no-records-copy
skips the second write.

    python scripts/figure_csv.py --out-dir figure_csv
"""
import argparse
import pathlib
import shutil

import pandas as pd

from glow._extra.benchmark import plot, results
from glow._extra.benchmark.file import get_path_records
from glow._extra.benchmark.make_csv import config_results_df


# Sort keys, most significant first; whichever a frame has are used. The
# records are read with an unsorted glob, so a frame arrives in filesystem
# order and two runs of this module would otherwise write the same rows in
# different orders -- a CSV that cannot be diffed, and needless churn in
# whatever ships it.
_ORDER_COLS = ('label', 'source', 'pool', 'ana', 'stat_name', 'cell',
               'cluster_mode', 'x_name', 'b', 'num_img', 'effect_llr', 'llr',
               'x', 'k', 'frac_segment', 'seed')


def stable_order(df):
    """Return the frame in a deterministic row order.

    Args:
        df: any tidy frame.

    Returns:
        the frame sorted by whichever _ORDER_COLS it carries (then by every
        remaining column, so rows those keys tie are still ordered), with a
        fresh index. NaN sorts last, consistently.
    """
    if df.empty:
        return df
    keys = [c for c in _ORDER_COLS if c in df.columns]
    keys += [c for c in df.columns if c not in keys]
    return (df.sort_values(keys, kind='mergesort', na_position='last')
            .reset_index(drop=True))


def drawn_rows(df):
    """Drop the rows a figure holds but never draws: the unlabelled ones.

    A label is recovered from the recorded recipe at read time, so a record
    left behind by a recipe the catalogue has since retired (the struck
    GLM-Error arm) reads back with no label. The plot layer groups by label,
    so those rows are silently absent from every figure; keeping them in the
    CSV would misreport what the figure shows.

    Args:
        df: any tidy frame; one without a label column passes through.

    Returns:
        the frame less its unlabelled rows.
    """
    if df.empty or 'label' not in df.columns:
        return df
    return df[df['label'].notna()]

# Where oracle_vs_k.py leaves the curves it computed, checked in the order a
# reader is likely to have them: beside this checkout, then the manuscript's
# own image directory.
ORACLE_CSV_CANDIDATES = (
    pathlib.Path('oracle_vs_k.csv'),
    pathlib.Path('../../publications/submissions/2026_glow/image/'
                 'oracle_vs_k.csv'),
)


def fig_segment():
    """Return Fig. 7's frame: oracle Dice per (trial, Ward mode)."""
    return drawn_rows(plot.tidy_segment(config_results_df('segment')))


def fig_prune():
    """Return Fig. 8's frame: the Focus block of the pruning-rule sweep."""
    df = plot.tidy_prune(config_results_df('prune'))
    return drawn_rows(df[df['cluster_mode'] == 'Focus'])


def fig_null():
    """Return Fig. 9's frame: one min_pval per null trial, GLOW arm only.

    The calibration curve is the empirical CDF of min_pval against nominal
    alpha, so the p-value column is the whole figure. Narrowed to
    plot._CALIB_METHODS, the same cut _plot_calibration_faceted makes: the
    null cache was pointed at GLOW alone partway through the run, so the
    voxel-wise arms have leftover partial records the figure never draws.
    """
    raw = config_results_df('null')
    df = drawn_rows(plot._select_glow_arm(plot.tidy_run_ana(raw)))
    df = df[df['label'].isin(plot._CALIB_METHODS)]
    return df.dropna(subset=['min_pval'])


def fig_sweep_llr_b1():
    """Return Fig. 10's frame: the b=1 block of the effect-strength sweep.

    The cache sweeps effect_llr and varies b alongside it, so the figure holds
    b fixed; this takes the same b=1 split the plot layer draws.
    """
    raw = config_results_df('sweep_llr')
    df = plot._select_glow_arm(plot.tidy_run_ana(raw))
    if df.empty:
        return df
    x = plot._infer_x(df)
    for sub_label, sub in plot._split_by_secondary('sweep_llr', df, x):
        if sub_label.endswith('_b1'):
            return drawn_rows(sub)
    return drawn_rows(df)


def _fig_runtime(name: str):
    """Return one runtime cache's frame: leaf wall time vs its cost knob."""
    return drawn_rows(plot.tidy_runtime(name, config_results_df(name)))


def tab_stat_dice():
    """Return Tab. 2's frame: one row per (cell, stat variant) run_stat leaf.

    Per-cell, not the published means: the table is a mean Dice over the
    cells, so the counts behind it are what a reader needs to recompute it.
    Read via results.stat_cell_df, which reaches the run_stat leaves directly
    (their build ancestors are absent, so the forward DAG walk drops them).
    """
    return results.stat_cell_df()


# figure id -> (filename stem, what it backs, builder). Ordered as the
# manuscript numbers them, so the printed summary reads like its figure list.
EXPORT = {
    'fig07': ('fig07_segment', 'Fig. 7 segmentation quality', fig_segment),
    'fig08': ('fig08_prune_focus', 'Fig. 8 pruning rule (Focus)', fig_prune),
    'fig09': ('fig09_null_calibration', 'Fig. 9 FWER control', fig_null),
    'fig10': ('fig10_sweep_llr_b1', 'Fig. 10 detection vs LLR (b=1)',
              fig_sweep_llr_b1),
    'fig12': ('fig12_runtime_num_vox', 'Fig. 12 wall clock vs num_vox',
              lambda: _fig_runtime('runtime_num_vox')),
    'fig13a': ('fig13a_runtime_1perm_num_vox',
               'Fig. 13a per-perm cost vs num_vox',
               lambda: _fig_runtime('runtime_1perm_num_vox')),
    'fig13b': ('fig13b_runtime_1perm_n_perm_inner',
               'Fig. 13b per-perm cost vs n_perm_inner',
               lambda: _fig_runtime('runtime_1perm_n_perm_inner')),
    'fig13c': ('fig13c_runtime_1perm_b', 'Fig. 13c per-perm cost vs b',
               lambda: _fig_runtime('runtime_1perm_b')),
    'fig13d': ('fig13d_runtime_1perm_nimg',
               'Fig. 13d per-perm cost vs num_img',
               lambda: _fig_runtime('runtime_1perm_nimg')),
    'tab02': ('tab02_stat_dice', 'Tab. 2 MANCOVA-statistic bake-off',
              tab_stat_dice),
}


def copy_oracle_csv(out_dir: pathlib.Path):
    """Copy oracle_vs_k's cached curves in as Fig. 11, if they are on hand.

    Args:
        out_dir (pathlib.Path): destination directory.

    Returns:
        pandas.DataFrame | None: the copied frame, or None when no sidecar
            was found (the figure then needs oracle_vs_k.py).
    """
    here = pathlib.Path(__file__).resolve().parent
    for cand in ORACLE_CSV_CANDIDATES:
        path = cand if cand.is_absolute() else (here / cand).resolve()
        if path.is_file():
            dest = out_dir / 'fig11_oracle_vs_k.csv'
            shutil.copyfile(path, dest)
            return pd.read_csv(dest)
    return None


def main(argv=None) -> None:
    """Write every figure CSV and print the table the rerun guide reports."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-dir', type=pathlib.Path,
                        default=pathlib.Path('figure_csv'),
                        help='destination directory (created if missing)')
    parser.add_argument('--no-records-copy', action='store_true',
                        help='do not also write beside the records')
    args = parser.parse_args(argv)

    out_dir_list = [args.out_dir]
    if not args.no_records_copy:
        beside = get_path_records() / 'figure_csv'
        if beside.resolve() != args.out_dir.resolve():
            out_dir_list.append(beside)
    for out_dir in out_dir_list:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for stem, backs, build in EXPORT.values():
        df = stable_order(build())
        for out_dir in out_dir_list:
            df.to_csv(out_dir / f'{stem}.csv', index=False)
        summary.append((f'{stem}.csv', backs, len(df), len(df.columns)))

    oracle = None
    for out_dir in out_dir_list:
        oracle = copy_oracle_csv(out_dir)
    row = ('fig11_oracle_vs_k.csv', 'Fig. 11 region-budget oracle',
           len(oracle) if oracle is not None else 0,
           len(oracle.columns) if oracle is not None else 0)
    summary.insert(4, row)
    if oracle is None:
        print('NOTE: no oracle_vs_k.csv found; run scripts/oracle_vs_k.py')

    width = max(len(name) for name, *_ in summary)
    print(f'\n{"csv".ljust(width)}  {"rows":>6s} {"cols":>4s}  backs')
    for name, backs, n_row, n_col in summary:
        print(f'{name.ljust(width)}  {n_row:6d} {n_col:4d}  {backs}')
    for out_dir in out_dir_list:
        print(f'\nwritten to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
