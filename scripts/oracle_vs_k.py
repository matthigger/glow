"""Chart the precision a Ward tree makes available at a given region budget.

The manuscript asks how much of GLOW's precision deficit is its greedy prune
and how much is the segmentation it prunes from. This answers the second half:
for each trial it finds the antichain of at most k tree nodes whose union best
matches the planted extent, choosing with knowledge of the ground truth, and
records the PPV that antichain attains. No pruning rule reading the same tree
can beat that curve at the same k, so it bounds what the segmentation offers;
GLOW's own operating point is drawn against it.

    python scripts/oracle_vs_k.py --out fig.pdf

Dice is a ratio of linear functions of (overlap, volume), so it is maximized
parametrically (Dinkelbach 1967): for a fixed lam the surrogate
2*tp - lam*vol is additive over disjoint nodes, a bottom-up DP carrying a
budget index solves every k at once, and sweeping lam recovers the Dice
optimum. Nodes holding no effect voxel are skipped, since selecting one only
subtracts lam*vol and so never appears in an optimum.

The run touches no fit: it reads the sweep_llr cache's records for the cells
to score, loads each cell's cached effect factory output, and rebuilds only
the Ward tree. Per-trial curves cache to a CSV beside the figure, so a second
run redraws without recomputing.
"""

import argparse
import pathlib
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from glow._extra.benchmark.file import get_path_cache
from glow._extra.benchmark.make_csv import config_results_df
from glow.analysis.cluster import ClusterMode, cluster

# the three effect strengths the manuscript reports, off the default LLR grid:
# the mid-range trough, the crossover, and the strongest effect tested.
LLR_LIST = [0.0189287203344057, 0.0475467957738334, 0.2999999999999999]

# lam is a Dice threshold, so the grid spans the unit interval; 24 points put
# the recovered optimum within a breakpoint of exact on every cell tested.
LAM_LIST = np.linspace(0.10, 0.995, 24)

GLOW_ARM = 'cluster_mode=Focus, prune_rule=greedy'

SOURCE_LIST = ['HCP', 'WGN']

# inches the figure is included at, the manuscript's 17.8cm text
# block, so it needs no font rescaling
FIG_W = 7.0


def _n_feat(value) -> int:
    """Count imaging features in one hcp_feats cell.

    The recorder hands back the tuple it stored, while the same column read
    from a written CSV arrives as its repr, so both are accepted.
    """
    if isinstance(value, str):
        value = eval(value)
    return len(value)


def _b1_cells(source: str):
    """Select one source's b = 1 rows out of the sweep_llr provenance frame."""
    df = config_results_df('sweep_llr')
    if source == 'HCP':
        df = df[df['data_factory_hcp.out.exp'].notna()]
        df = df[df['data_factory_hcp.in.hcp_feats'].map(_n_feat) == 1]
        seed = df['data_factory_hcp.in.seed']
    else:
        df = df[df['data_factory_wgn.out.exp'].notna()]
        df = df[df['data_factory_wgn.in.b'] == 1]
        seed = df['data_factory_wgn.in.seed']
    return df.assign(seed=seed.astype(int), source=source,
                     llr=df['effect_factory_single.in.effect_llr'])


def load_cells(llr_list, source: str):
    """Select one source's b=1 cells at each effect strength, one per seed.

    Args:
        llr_list (list): effect_llr values to keep, matched to within 1e-9.
        source (str): HCP or WGN.

    Returns:
        pandas.DataFrame: columns {llr, seed, eff_hash}, one row per cell.
    """
    df = _b1_cells(source)
    df = df[df.llr.map(lambda v: any(abs(v - t) < 1e-9 for t in llr_list))]
    df = df.rename(columns={'effect_factory_single.hash': 'eff_hash'})
    return df.groupby(['llr', 'seed']).first().reset_index()


def load_effect_cell(eff_hash: str):
    """Load one cached effect_factory_single output by its args hash.

    Returns:
        exp (Experiment): the perturbed experiment
        mask (np.array): (X, Y, Z) boolean, the planted support
    """
    path = (get_path_cache() / 'glow' / '_extra' / 'benchmark' / 'data'
            / 'effect_factory_single' / eff_hash / 'output.pkl')
    exp, mask_list = joblib.load(path)
    return exp, mask_list[0]


def node_counts(children, num_vox, plant):
    """Accumulate each forest node's volume and its overlap with the plant.

    Args:
        children (np.array): (num_internal, 2) child index pairs
        num_vox (int): leaf count; node num_vox + i is merge i
        plant (np.array): (num_vox,) boolean, True inside the planted extent

    Returns:
        size (np.array): (num_vox + num_internal,) int voxels per node
        over (np.array): (num_vox + num_internal,) int planted voxels per node
    """
    size = np.concatenate([np.ones(num_vox, dtype=np.int64),
                           np.zeros(len(children), dtype=np.int64)])
    over = np.concatenate([plant.astype(np.int64),
                           np.zeros(len(children), dtype=np.int64)])
    for i, (c0, c1) in enumerate(children):
        size[num_vox + i] = size[c0] + size[c1]
        over[num_vox + i] = over[c0] + over[c1]
    return size, over


def _combine(a, b, k_max):
    """Max-plus convolve two budget vectors, splitting the budget between them.

    Args:
        a, b (list): (k_max+1,) of (value, tp, vol), non-decreasing in value
        k_max (int): the largest budget to fill

    Returns:
        list: (k_max+1,) of (value, tp, vol)
    """
    out = [(0.0, 0, 0)]
    for j in range(1, k_max + 1):
        best = (0.0, 0, 0)
        for m in range(j + 1):
            x, y = a[m], b[j - m]
            if x[0] + y[0] > best[0]:
                best = (x[0] + y[0], x[1] + y[1], x[2] + y[2])
        out.append(best)
    return out


def dp_pass(children, num_vox, size, over, lam: float, k_max: int):
    """Maximize 2*tp - lam*vol over antichains of at most j nodes, j <= k_max.

    Only a node whose two children both hold effect voxels needs the budget
    split; elsewhere one side contributes nothing and the parent inherits its
    sibling's vector, which is what keeps the pass linear in the plant rather
    than in the tree.

    Args:
        children (np.array): (num_internal, 2) child index pairs
        num_vox (int): leaf count
        size, over (np.array): per-node volume and overlap (node_counts)
        lam (float): the Dice level the surrogate is tested against
        k_max (int): the largest region budget to solve

    Returns:
        list: (k_max+1,) of (tp, vol), the argmax at each budget
    """
    zero = [(0.0, 0, 0)] * (k_max + 1)
    leaf = [(0.0, 0, 0)] + [(2.0 - lam, 1, 1)] * k_max
    vec = {}
    for i, (c0, c1) in enumerate(children):
        v = num_vox + i
        if over[v] == 0:
            continue
        pair = []
        for c in (c0, c1):
            if c in vec:
                pair.append(vec.pop(c))
            elif c < num_vox and over[c] > 0:
                pair.append(leaf)
            else:
                pair.append(zero)
        a, b = pair
        if b is zero:
            out = list(a)
        elif a is zero:
            out = list(b)
        else:
            out = _combine(a, b, k_max)
        take = 2.0 * over[v] - lam * size[v]
        if take > 0:
            rec = (take, int(over[v]), int(size[v]))
            out = [out[0]] + [rec if take > o[0] else o for o in out[1:]]
        vec[v] = out

    # cluster returns a forest on a non-contiguous mask, so whatever vectors
    # survive unclaimed are the roots and their budgets still trade off.
    root = zero
    for other in vec.values():
        root = other if root is zero else _combine(root, other, k_max)
    return [(tp, vol) for _, tp, vol in root]


def oracle_curve(children, num_vox, plant, k_max: int, lam_list):
    """Score the best-Dice antichain of at most k nodes, for every k.

    Args:
        children (np.array): (num_internal, 2) child index pairs
        num_vox (int): leaf count
        plant (np.array): (num_vox,) boolean, the planted support
        k_max (int): the largest region budget to solve
        lam_list (np.array): (n_lam,) Dice levels to sweep

    Returns:
        list: one {k, dice, ppv, sens, n_vox} per budget with a non-empty
            selection
    """
    size, over = node_counts(children, num_vox, plant)
    n_eff = int(plant.sum())
    best = [(-1.0, 0, 0)] * (k_max + 1)
    for lam in lam_list:
        for j, (tp, vol) in enumerate(dp_pass(children, num_vox, size, over,
                                              lam, k_max)):
            if vol and 2.0 * tp / (vol + n_eff) > best[j][0]:
                best[j] = (2.0 * tp / (vol + n_eff), tp, vol)
    return [dict(k=j, dice=dice, ppv=tp / vol, sens=tp / n_eff, n_vox=vol)
            for j, (dice, tp, vol) in enumerate(best) if j and vol]


def compute(k_max: int):
    """Run every cell of both sources and return the per-trial curves.

    Returns:
        pandas.DataFrame: one row per (source, llr, seed, k)
    """
    row_list = []
    for source in SOURCE_LIST:
        for _, cell in load_cells(LLR_LIST, source).iterrows():
            sec = time.time()
            exp, mask = load_effect_cell(cell.eff_hash)
            plant = np.zeros(exp.y.shape[2], dtype=bool)
            plant[exp.mask_idx[mask]] = True
            children = cluster(exp, mode=ClusterMode.FOCUS)
            for rec in oracle_curve(children, exp.y.shape[2], plant, k_max,
                                    LAM_LIST):
                row_list.append(dict(source=source, llr=cell.llr,
                                     seed=cell.seed, **rec))
            print(f'{source} llr {cell.llr:.4f} seed {cell.seed:2d} '
                  f'[{time.time() - sec:.1f}s]', flush=True)
    return pd.DataFrame(row_list)


def greedy_points(source: str):
    """Read GLOW's own operating point per effect strength off the cache.

    Args:
        source (str): HCP or WGN.

    Returns:
        pandas.DataFrame: indexed by llr, columns {n_reg, ppv}, both means
            over the seeds of that cell
    """
    df = _b1_cells(source)
    df = df[df['run_ana.in.ana'].str.contains(GLOW_ARM, regex=False)]
    df = df[df.llr.map(lambda v: any(abs(v - t) < 1e-9 for t in LLR_LIST))]
    tp = df['run_ana.out.score.target.tp']
    fp = df['run_ana.out.score.target.fp']
    df = df.assign(ppv=np.where(tp + fp > 0, tp / (tp + fp), np.nan),
                   n_reg=df['run_ana.out.score.n_pred'])

    # PPV is undefined on a trial that discovered nothing, so the region count
    # is averaged over the same trials rather than over all of them; on WGN
    # most weak-effect trials find nothing and would drag the marker to zero.
    df = df[tp + fp > 0]
    return df.groupby('llr').agg(n_reg=('n_reg', 'mean'), ppv=('ppv', 'mean'))


def plot(df, out: pathlib.Path) -> None:
    """Draw mean PPV against region budget, one panel per image source.

    Two legends rather than one: colour carries the effect strength, and the
    line-against-marker distinction carries which curve is the bound and which
    is GLOW, so a reader does not have to infer the second from the caption.
    """
    sns.set_theme(style='whitegrid', context='paper')
    ramp = plt.get_cmap('viridis')(np.linspace(0, 0.88, df.llr.nunique()))
    source_list = [s for s in SOURCE_LIST if s in set(df.source)]

    # the full 17.8cm text block, as the paper's other multi-panel figures
    # take, so the figure is included 1:1 with no font rescaling
    fig, axes = plt.subplots(1, len(source_list), figsize=(FIG_W, 0.47 * FIG_W),
                             sharey=True, squeeze=False,
                             constrained_layout=True)
    handle_list = []
    for ax, source in zip(axes[0], source_list):
        sub = df[df.source == source]
        greedy = greedy_points(source)
        for llr, colour in zip(sorted(df.llr.unique()), ramp):
            curve = sub[sub.llr == llr].groupby('k').ppv.mean()
            ax.plot(curve.index, curve.values, color=colour, marker='.')
            near = greedy.index[np.argmin(np.abs(greedy.index - llr))]
            ax.plot(greedy.n_reg[near], greedy.ppv[near], marker='o', ms=8,
                    color=colour, mec='0.2', mew=0.7, ls='none', zorder=5)
        ax.set_title(source)
        ax.set_xlabel('regions output')
    for llr, colour in zip(sorted(df.llr.unique()), ramp):
        handle_list.append(Line2D([], [], color=colour, marker='.',
                                  label=f'{llr:.3g}'))
    mark_list = [Line2D([], [], color='0.35', marker='.',
                        label='Oracle, best $k$ regions'),
                 Line2D([], [], color='0.35', marker='o', ms=8, ls='none',
                        mec='0.2', mew=0.7, label='GLOW, greedy prune')]

    axes[0][0].set_ylabel('PPV')
    # floor well below the weakest curve, so the legends sit in empty space
    # rather than over the k = 1 points the surrounding argument turns on
    axes[0][0].set_ylim(0.28, 1.04)
    axes[0][-1].add_artist(axes[0][-1].legend(
        handles=handle_list, title='LLR / |r|', loc='lower right',
        frameon=True, handlelength=1.6, fontsize='small'))
    axes[0][0].legend(handles=mark_list, loc='lower right', frameon=True,
                      handlelength=1.6, fontsize='small')
    fig.savefig(out)
    print(f'saved: {out}')


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the oracle_vs_k CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=pathlib.Path,
                        default=pathlib.Path('oracle_vs_k.pdf'),
                        help='figure path; the cache CSV sits beside it')
    parser.add_argument('--k-max', type=int, default=10,
                        help='largest region budget to solve')
    parser.add_argument('--recompute', action='store_true',
                        help='ignore the cached CSV and score every cell')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Score the curves (or reuse the cached CSV) and draw the figure."""
    args = parse_args(argv)
    csv_path = args.out.with_suffix('.csv')
    if csv_path.exists() and not args.recompute:
        df = pd.read_csv(csv_path)
    else:
        df = compute(args.k_max)
        df.to_csv(csv_path, index=False)
        print(f'saved: {csv_path}')
    plot(df, args.out)


if __name__ == '__main__':
    main()
