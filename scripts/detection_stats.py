"""Print the derived numbers the manuscript's detection section quotes.

Everything the Detection Performance prose states beyond what the figures
draw, recomputed from the cached records so a reader can check any of it:

  - the seed-paired Dice difference against the best alternative, averaged
    below and above the crossover
  - the sensitivity gain and the precision it costs, and the PPV each method
    reaches at matched mean sensitivity
  - the split of GLOW's false volume between undersegmentation and wholly
    spurious regions, weighted per false voxel so no purity cut-off is needed
  - completeness read as the rate at which a trial returns one region, and
    the region purities that drive the structural pair apart
  - the split of the precision gap between the prune rule, the permutation
    test and the segmentation, each measured against its own oracle

    python scripts/detection_stats.py

Reads the sweep_llr and prune caches through the recorder, and
oracle_vs_k.py's cached curves for the candidate-pool block; no fit.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from glow._extra.benchmark.make_csv import config_results_df
from glow._extra.benchmark.plot import _hom_com, _pred_block
from glow.mask import stats_from_counts

# the manuscript's four arms, keyed by the substring identifying each recipe
ARM_DICT = {'GLOW': 'cluster_mode=Focus, prune_rule=greedy',
            'VBA': 'get_hotel_tr, n_perm_fwer=500, alpha_fwer=0.05, '
                   'tfce_flag=False',
            'VBA-TFCE': 'get_wilks, n_perm_fwer=500, alpha_fwer=0.05, '
                        'tfce_flag=True',
            'CET': 'AnalysisCET'}

# the crossover: GLOW leads at or below the first, trails at or above it
LLR_WEAK_MAX = 0.031
LLR_STRONG_MIN = 0.047

# the three effect strengths tabulated per source
LLR_REPORT = [0.0189287203344057, 0.0475467957738334, 0.2999999999999999]


def _n_feat(value):
    """Count imaging features in one hcp_feats cell, NaN off the HCP path.

    The recorder hands back the tuple it stored, while the same column read
    from a written CSV arrives as its repr, so both are accepted.
    """
    if isinstance(value, str):
        value = eval(value)
    try:
        return len(value)
    except TypeError:
        return np.nan


def _arm(value) -> str:
    """Name the arm one ana repr belongs to, or an empty string for none."""
    for name, key in ARM_DICT.items():
        if key in str(value):
            return name
    return ''


def load_sweep():
    """Build the tidy per-trial frame the section is scored on.

    Returns:
        df (pandas.DataFrame): one row per trial with {m, source, b, llr,
            seed, tp, fp, fn, dice, sens, ppv, com, hom, n_reg, one}
        n_r, o_r (np.array): (n_row, n_reg) per-region volume and overlap,
            row-aligned with df's source frame
    """
    raw = config_results_df('sweep_llr')
    n_r, o_r = _pred_block(raw, 'num_vox'), _pred_block(raw, 'target')
    hom, com = _hom_com(n_r, o_r)
    hcp = raw['data_factory_hcp.out.exp'].notna()
    counts = {k: raw[f'run_ana.out.score.target.{k}']
              for k in ('tp', 'fp', 'tn', 'fn')}
    df = pd.DataFrame({
        'm': raw['run_ana.in.ana'].map(_arm),
        'source': np.where(hcp, 'HCP', 'WGN'),
        'b': np.where(hcp, raw['data_factory_hcp.in.hcp_feats']
                      .map(_n_feat), raw['data_factory_wgn.in.b']),
        'llr': raw['effect_factory_single.in.effect_llr'],
        'seed': raw['data_factory_hcp.in.seed'].fillna(
            raw['data_factory_wgn.in.seed']),
        'n_reg': raw['run_ana.out.score.n_pred'],
        'hom': hom, 'com': com, **counts,
        **stats_from_counts(**counts)})
    df['one'] = df.n_reg == 1
    keep = (df.m != '') & (df.b == 1)
    return df[keep].reset_index(drop=True), n_r[keep.values], o_r[keep.values]


def report_dice(df) -> None:
    """Mean seed-paired Dice difference below and above the crossover."""
    print('\n== Dice against the best alternative on each seed ==')
    wide = df.pivot_table(index=['source', 'llr', 'seed'], columns='m',
                          values='dice')
    wide['best'] = wide[[c for c in wide.columns if c != 'GLOW']].max(axis=1)
    diff = (wide.GLOW - wide.best).groupby(level=['source', 'llr']).mean()
    for source in sorted(df.source.unique()):
        cell = diff.loc[source]
        weak = cell[cell.index <= LLR_WEAK_MAX].mean()
        strong = cell[cell.index >= LLR_STRONG_MIN].mean()
        print(f'  {source}: weak {weak:+.3f}   strong {strong:+.3f}')


def report_trade(df) -> None:
    """Sensitivity gained, precision lost, and PPV at matched sensitivity."""
    print('\n== the sensitivity / precision trade (HCP) ==')
    hcp = df[df.source == 'HCP']
    wide = hcp.pivot_table(index=['llr', 'seed'], columns='m',
                           values=['sens', 'ppv'])
    other = [c for c in wide['sens'].columns if c != 'GLOW']
    sens = (wide['sens'].GLOW - wide['sens'][other].max(axis=1))
    ppv = (wide['ppv'].GLOW - wide['ppv'][other].max(axis=1))
    print(f'  best sensitivity gain over a cell mean: '
          f'{sens.groupby(level="llr").mean().max():+.3f}')
    print(f'  cells where GLOW PPV trails: '
          f'{(ppv.groupby(level="llr").mean() < 0).sum()} of '
          f'{ppv.index.get_level_values("llr").nunique()}; '
          f'seed-paired PPV wins {(ppv > 0).sum()} of {ppv.notna().sum()}')
    cell = hcp.groupby(['m', 'llr'])[['sens', 'ppv']].mean().reset_index()
    grid = np.array([0.1, 0.4, 0.8, 0.95])
    print('  PPV at matched mean sensitivity ' +
          ' '.join(f'{g:g}' for g in grid))
    for m, sub in cell.groupby('m'):
        sub = sub.sort_values('sens')
        row = np.interp(grid, sub.sens, sub.ppv, left=np.nan, right=np.nan)
        print(f'    {m:9s}' + ' '.join(f'{v:8.3f}' for v in row))


def report_false_volume(df, n_r, o_r) -> None:
    """Split GLOW's false volume, weighting each region by its false voxels."""
    print('\n== where GLOW\'s false volume sits ==')
    for source in sorted(df.source.unique()):
        keep = ((df.m == 'GLOW') & (df.source == source)).to_numpy()
        n, o = n_r[keep], o_r[keep]
        n_eff = (df.tp + df.fn)[keep].iloc[0]
        weight = np.where(np.isnan(n), 0.0, n - o)
        cover = (weight * np.nan_to_num(o / n_eff)).sum() / weight.sum()
        spurious = int(np.nansum((o == 0) & (n > 0)))
        total = int(np.nansum(n > 0))
        print(f'  {source}: a false voxel\'s region holds {cover:.1%} of the '
              f'effect; {spurious} of {total} regions hold none')


def report_structure(df, n_r, o_r) -> None:
    """Completeness as a one-region rate, and the purity behind homogeneity."""
    print('\n== the structural pair ==')
    ok = df[df.com.notna()]
    for source in sorted(df.source.unique()):
        sub = ok[ok.source == source]
        agg = sub.groupby('m')[['com', 'hom', 'one']].mean()
        print(f'  {source} (completeness, homogeneity, P(one region)):')
        print('    ' + agg.round(3).to_string().replace('\n', '\n    '))
        cell = sub.groupby(['m', 'llr'])[['com', 'one']].mean()
        print(f'    corr(mean completeness, P(one region)) over cells: '
              f'{cell.com.corr(cell.one):.3f}')

    print('  share of a trial\'s regions lying wholly inside the plant, '
          'HCP mid-range:')
    mid = ((df.source == 'HCP') & df.llr.between(0.018, 0.08)
           & df.com.notna()).to_numpy()
    pure = np.where(np.isnan(n_r), np.nan, (o_r == n_r).astype(float))
    frac = pd.Series(np.nanmean(pure[mid], axis=1), index=df.index[mid])
    agg = pd.DataFrame({'frac_pure': frac, 'm': df.m[mid],
                        'n_reg': df.n_reg[mid]}).groupby('m').agg(
        frac_pure=('frac_pure', 'mean'), n_reg=('n_reg', 'median'))
    print('    ' + agg.round(3).to_string().replace('\n', '\n    '))

    glow = ok[(ok.m == 'GLOW') & (ok.source == 'HCP')]
    print(f'    within GLOW on HCP: homogeneity vs PPV rho '
          f'{spearmanr(glow.hom, glow.ppv).statistic:+.3f}, '
          f'completeness vs region count rho '
          f'{spearmanr(glow.com, glow.n_reg).statistic:+.3f}')


def report_prune() -> None:
    """Compare the greedy and oracle prune rules on the same fits."""
    print('\n== prune rules on the same significant regions (Focus, HCP) ==')
    raw = config_results_df('prune')
    counts = {k: raw[f'run_prune.out.score.{k}']
              for k in ('tp', 'fp', 'tn', 'fn')}
    df = pd.DataFrame({
        'rule': raw['run_prune.in.rule'],
        'mode': raw['run_prune.in.cluster_mode'],
        'source': np.where(raw['data_factory_hcp.out.exp'].notna(),
                           'HCP', 'WGN'),
        'llr': raw['effect_factory_single.in.effect_llr'],
        'n_sel': raw['run_prune.out.score.n_selected'],
        **stats_from_counts(**counts)})
    df = df[(df['mode'] == 'Focus') & (df.source == 'HCP')]
    for metric in ('ppv', 'n_sel'):
        piv = df.pivot_table(index='rule', columns='llr', values=metric)
        piv = piv[[c for c in piv.columns
                   if any(abs(c - t) < 1e-9 for t in LLR_REPORT)]]
        print(f'  {metric}:')
        print('    ' + piv.round(3).to_string().replace('\n', '\n    '))


def _greedy_trials(source: str):
    """Read greedy's per-trial operating point off the prune cache (Focus).

    Returns:
        pandas.DataFrame: {llr, seed, n_sel, ppv}, one row per trial in which
            greedy discovered something. PPV is undefined on a trial that
            discovered nothing, so those are dropped rather than scored zero.
    """
    raw = config_results_df('prune')
    raw = raw[(raw['run_prune.in.cluster_mode'].astype(str) == 'Focus')
              & (raw['run_prune.in.rule'] == 'greedy')]
    if source == 'HCP':
        raw = raw[raw['data_factory_hcp.out.exp'].notna()]
        seed = raw['data_factory_hcp.in.seed']
    else:
        raw = raw[raw['data_factory_wgn.out.exp'].notna()]
        seed = raw['data_factory_wgn.in.seed']
    tp, fp = raw['run_prune.out.score.tp'], raw['run_prune.out.score.fp']
    out = pd.DataFrame({'llr': raw['effect_factory_single.in.effect_llr'],
                        'seed': seed.astype(int),
                        'n_sel': raw['run_prune.out.score.n_selected'],
                        'ppv': tp / (tp + fp).replace(0, np.nan)})
    return out[out.ppv.notna()]


def report_oracle_pool(df, csv_path: pathlib.Path) -> None:
    """Split GLOW's precision gap between its prune rule, the test and the tree.

    Reads oracle_vs_k.py's per-trial curves and tabulates, at that run's
    largest region budget, the PPV and Dice of the best-Dice selection out of
    the FWER-significant regions and out of the whole Ward tree, beside
    greedy's own PPV and each voxel-wise arm's. Every quantity is averaged over the trials
    in which greedy discovered something, so the three rises partition the gap
    from greedy up to the best arm: the ranking's, the permutation test's, and
    the segmentation's.

    Args:
        df (pandas.DataFrame): load_sweep's per-trial frame
        csv_path (pathlib.Path): oracle_vs_k.py's cached per-trial curves
    """
    curve = pd.read_csv(csv_path)
    k = int(curve.k.max())
    curve = curve[curve.k == k]
    print(f'\n== PPV by candidate pool, budget k = {k} (Focus, the trials in '
          'which greedy discovered) ==')
    for source in ('HCP', 'WGN'):
        found = _greedy_trials(source)
        sweep = df[(df.source == source) & (df.b == 1)]
        col_dict = {}
        for llr in LLR_REPORT:
            near = found[np.isclose(found.llr, llr)]
            seed_set = set(near.seed)
            col = {'greedy': near.ppv.mean(), 'greedy n_sel': near.n_sel.mean(),
                   'n trial': float(len(near))}
            sub = curve[(curve.source == source) & np.isclose(curve.llr, llr)
                        & curve.seed.isin(seed_set)]
            for pool in ('sig', 'tree'):
                col[f'oracle {pool}'] = sub[sub.pool == pool].ppv.mean()
                col[f'oracle {pool} dice'] = sub[sub.pool == pool].dice.mean()
            arm = sweep[np.isclose(sweep.llr, llr) & sweep.seed.isin(seed_set)]
            for name in ('VBA', 'VBA-TFCE', 'CET'):
                col[name] = arm[arm.m == name].ppv.mean()
            col_dict[round(llr, 4)] = col
        tab = pd.DataFrame(col_dict)
        print(f'  {source}:')
        print('    ' + tab.round(3).to_string().replace('\n', '\n    '))

        best = tab.loc[['VBA', 'VBA-TFCE', 'CET']].max()
        share = pd.DataFrame(
            {'rule': tab.loc['oracle sig'] - tab.loc['greedy'],
             'test': tab.loc['oracle tree'] - tab.loc['oracle sig'],
             'segmentation': best - tab.loc['oracle tree']}).T
        print(f'    rises summing to best arm minus greedy '
              f'({(best - tab.loc["greedy"]).round(3).to_dict()}):')
        print('    ' + (share / (best - tab.loc['greedy'])).round(3)
              .to_string().replace('\n', '\n    '))


def main(argv=None) -> None:
    """Print every block."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--oracle-csv', type=pathlib.Path,
                        default=pathlib.Path('oracle_vs_k.csv'),
                        help="oracle_vs_k.py's cached per-trial curves")
    args = parser.parse_args(argv)
    df, n_r, o_r = load_sweep()
    report_dice(df)
    report_trade(df)
    report_false_volume(df, n_r, o_r)
    report_structure(df, n_r, o_r)
    report_prune()
    report_oracle_pool(df, args.oracle_csv)


if __name__ == '__main__':
    main()
