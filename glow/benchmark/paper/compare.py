"""Interactive REPL: rank a config's trials by where GLOW most underperforms.

A companion to glow.benchmark.paper.plot. It scans the per-user results
directory for configs (labels) that have a results.csv,
asks which one to view and which metric to sort by, then lists the
config's worst GLOW cases (the N_SHOW trials with the largest gap).

"Worst" is per-trial regret: for each trial (one trial_hash, holding
seed / effect_llr / data-source / extenter fixed) it compares the chosen
GLOW variant's metric against the best of the non-GLOW methods (VBA,
VBA-TFCE, CET) on that same trial. Trials are sorted ascending by

    gap = glow_metric - max(non_glow_metric)

so the trial where GLOW falls furthest below the best alternative comes
first — the failure cases a researcher most wants to inspect.

The sortable metrics are dice, sens, ppv, and spec. All are derived from
the stored tp/fp/tn/fn counts at load (glow.mask.stats_from_counts).

Run it interactively:

    python -m glow.benchmark.paper.compare
"""

import sys

import pandas as pd

import glow.benchmark


# label -> the cluster_mode it stands for (config.py ANALYSIS_DICT)
GLOW_LABEL = {'focus': 'GLOW-Focus', 'error': 'GLOW-GLM'}

# how many trials (the worst GLOW cases) to print per query
N_SHOW = 10


def find_configs() -> list:
    """List (label, csv_path) for every config folder that has a results.csv.

    Skips the _latest scratch folder written by paper.plot.

    Returns:
        sorted list of (label, pathlib.Path) pairs, one per config whose
            results.csv exists on disk
    """
    base = glow.benchmark.get_path_result()
    out = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir() or sub.name == '_latest':
            continue
        csv = sub / 'results.csv'
        if csv.exists():
            out.append((sub.name, csv))
    return out


def choose(prompt: str, options: list) -> int:
    """Prompt for a 1-based choice among options; return its 0-based index.

    Re-prompts on bad input. Returns -1 if the user types q / quit / an
    empty line (the REPL reads that as "quit").

    Args:
        prompt (str): line printed above the numbered menu
        options (list): human-readable option strings

    Returns:
        the 0-based index of the chosen option, or -1 to quit
    """
    print(f'\n{prompt}')
    for i, opt in enumerate(options, start=1):
        print(f'  {i}. {opt}')
    while True:
        raw = input('  choice (q to quit): ').strip().lower()
        if raw in ('', 'q', 'quit'):
            return -1
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print('  ? please enter a number from the list (or q)')


def summarize(df, glow_label: str, metric_col: str):
    """Build a per-trial table of method scores plus the GLOW-vs-best gap.

    One output row per trial_hash. Method scores are pivoted into one
    column each; gap is the chosen GLOW variant's score minus the best
    non-GLOW score on the same trial. Rows are sorted ascending by gap,
    so the trial where GLOW falls furthest below the best alternative is
    first.

    Args:
        df: a config's results, indexed by trial_hash, with a label
            column and the metric column (one row per method per trial)
        glow_label (str): the chosen GLOW label, e.g. 'GLOW-Focus'
        metric_col (str): the results column to compare on (dice/sens/spec)

    Returns:
        a DataFrame indexed by trial_hash with the trial descriptors
            (seed, effect_llr, vox_effect where present), one column per
            method label, and best_other / gap columns, sorted by gap
    """
    df = df.copy()
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

    # one column per method label; rows are trials
    pivot = df.pivot_table(index=df.index, columns='label',
                           values=metric_col)

    # trial descriptors are constant within a trial_hash
    desc_cols = [c for c in ('seed', 'effect_llr', 'vox_effect')
                 if c in df.columns]
    desc = df.groupby(level=0)[desc_cols].first()

    non_glow = [c for c in pivot.columns if not str(c).startswith('GLOW')]
    out = desc.join(pivot)
    out['best_other'] = pivot[non_glow].max(axis=1)
    out['gap'] = pivot[glow_label] - out['best_other']

    # ascending: most-negative gap (GLOW worst vs. the field) first
    return out.sort_values('gap')


def print_summary(summary, glow_label: str, metric: str) -> None:
    """Print the sorted per-trial table and a one-line loss tally.

    Args:
        summary: the DataFrame from summarize()
        glow_label (str): the chosen GLOW label, for column ordering
        metric (str): user-facing metric name, for the header
    """
    method_cols = [c for c in summary.columns
                   if c not in ('seed', 'effect_llr', 'vox_effect',
                                'best_other', 'gap')]
    # lead with the GLOW variant, then the rest
    method_cols = ([glow_label] +
                   [c for c in method_cols if c != glow_label])
    desc_cols = [c for c in ('effect_llr', 'seed', 'vox_effect')
                 if c in summary.columns]
    cols = desc_cols + method_cols + ['best_other', 'gap']

    n = len(summary)
    n_loss = int((summary['gap'] < 0).sum())
    n_show = min(N_SHOW, n)
    print(f'\n{n} trials by {metric} gap ({glow_label} - best non-GLOW); '
          f'GLOW loses on {n_loss}/{n}.  Worst {n_show}:\n')
    with pd.option_context('display.max_rows', None,
                           'display.width', None,
                           'display.float_format', lambda v: f'{v:.4f}'):
        print(summary[cols].head(n_show).to_string())


def main() -> None:
    """Run the compare REPL: pick GLOW variant, then loop config + metric."""
    configs = find_configs()
    if not configs:
        print('no configs with a results.csv under '
              f'{glow.benchmark.get_path_result()}')
        return

    g = choose('Which GLOW method?', ['focus (GLOW-Focus)',
                                      'error (GLOW-GLM)'])
    if g < 0:
        return
    glow_label = GLOW_LABEL['focus' if g == 0 else 'error']

    while True:
        labels = [name for name, _ in configs]
        c = choose('Which config to view?', labels)
        if c < 0:
            return
        label, csv = configs[c]
        df = glow.benchmark.load_results_csv(csv)

        if glow_label not in set(df['label']):
            print(f'  {label} has no {glow_label} rows — skipping.')
            continue

        feats = ['dice', 'sens', 'ppv', 'spec']
        m = choose('Sort trials by which feature?', feats)
        if m < 0:
            return
        metric = feats[m]

        summary = summarize(df, glow_label, metric)
        print_summary(summary, glow_label, metric)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)
