"""Chart the precision available to a pruning rule, and to the tree it prunes.

The manuscript asks how much of GLOW's precision deficit is its greedy prune
and how much is the segmentation it prunes from. Two oracles answer it on one
budget axis. Both take the antichain of at most k regions whose union best
matches the planted extent, choosing with knowledge of the ground truth, and
differ only in what they may choose from:

  - sig: the FWER-significant regions, the pool greedy itself ranks. No rule
    reading this fit can beat it at the same region count, so the rise from
    GLOW's own point to this curve is what the ranking leaves behind.
  - tree: every node of the Ward tree. The rise from sig to tree is what the
    permutation test withheld; what still separates tree from a voxel-wise
    method's precision is the segmentation's own.

    python scripts/oracle_vs_k.py --out fig.pdf

Dice is a ratio of linear functions of (overlap, volume), so it is maximized
parametrically (Dinkelbach 1967): at a fixed lam the surrogate 2*tp - lam*vol
is additive over disjoint nodes, and a bottom-up DP carrying a budget index
solves every k at once. Feeding each budget's attained Dice back as the next
lam converges to its exact optimum. prune.prune_oracle runs the same reduction
over the sig pool without a budget, so its recorded Dice is what the sig curve
converges to, and the two agree cell by cell.

The run touches no fit: it reads the prune cache's records and takes each
cell's Ward tree and significant set off the memoised glow_fit_for_prune
triple that cache already built. Per-trial curves cache to a CSV beside the
figure, so a second run redraws without recomputing.
"""

import argparse
import functools
import pathlib
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from glow._extra.benchmark.config import (ALPHA_FWER, N_PERM_FWER,
                                          N_PERM_INNER)
from glow._extra.benchmark.file import get_path_cache
from glow._extra.benchmark.make_csv import config_results_df
from glow._extra.benchmark.run import glow_fit_for_prune
from glow.analysis.cluster import ClusterMode

# the three effect strengths the manuscript reports, off the default LLR grid:
# the mid-range trough, the crossover, and the strongest effect tested.
LLR_LIST = [0.0189287203344057, 0.0475467957738334, 0.2999999999999999]

# lam is a Dice threshold, so the seed grid spans the unit interval. It only
# has to put each budget in reach of its optimum; the Dinkelbach iteration
# oracle_curve runs afterwards walks the rest of the way.
LAM_LIST = np.linspace(0.05, 0.95, 10)

# prune.prune_oracle's tolerances, so both routes to the same optimum stop at
# the same place: a Dice gain this small is a fixed point, and the cap only
# guards a cycle.
_DICE_TOL = 1e-9
_DICE_MAX_ITER = 32

# the two candidate pools, in draw order (see the module docstring)
POOL_LIST = ['sig', 'tree']

SOURCE_LIST = ['HCP', 'WGN']

# inches the figure is included at, the manuscript's 17.8cm text
# block, so it needs no font rescaling
FIG_W = 7.0


@functools.lru_cache(maxsize=1)
def _records():
    """Read the prune cache's provenance frame (once per process)."""
    return config_results_df('prune')


def _n_feat(value) -> int:
    """Count imaging features in one hcp_feats cell.

    The recorder hands back the tuple it stored, while the same column read
    from a written CSV arrives as its repr, so both are accepted.
    """
    if isinstance(value, str):
        value = eval(value)
    return len(value)


def _pool_df(source: str):
    """Select one source's Focus-mode rows out of the prune frame.

    Focus is the Ward mode the manuscript's GLOW arm runs, and the mode whose
    shared fit the significant sets below come from.
    """
    df = _records()
    df = df[df['run_prune.in.cluster_mode'].astype(str) == 'Focus']
    if source == 'HCP':
        df = df[df['data_factory_hcp.out.exp'].notna()]
        df = df[df['data_factory_hcp.in.hcp_feats'].map(_n_feat) == 1]
        seed = df['data_factory_hcp.in.seed']
    else:
        df = df[df['data_factory_wgn.out.exp'].notna()]
        df = df[df['data_factory_wgn.in.b'] == 1]
        seed = df['data_factory_wgn.in.seed']
    df = df[df['effect_factory_single.in.effect_llr'].map(
        lambda v: any(abs(v - t) < 1e-9 for t in LLR_LIST))]
    return df.assign(seed=seed.astype(int), source=source,
                     llr=df['effect_factory_single.in.effect_llr'])


def load_cells(source: str):
    """Select one source's cells at each effect strength, one per seed.

    Returns:
        pandas.DataFrame: columns {llr, seed, uid, eff_hash}, one row per
            cell; uid keys the shared GLOW fit, eff_hash the planted extent.
    """
    df = _pool_df(source).rename(
        columns={'effect_factory_single.hash': 'eff_hash',
                 'run_prune.in.parent_uid': 'uid'})
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


def load_fit(uid: str):
    """Read one cell's memoised GLOW fit, without refitting it.

    The tree here is the fit's own, built on the scaled experiment
    Analysis.fit derives; re-clustering the raw experiment gives a tree that
    differs in a few percent of its merges and that no significant-region
    index refers to.

    Args:
        uid (str): the cell's parent_uid, which keys the shared fit.

    Returns:
        children (np.array): (num_reg - num_vox, 2) Ward child-index pairs
        sig_reg_list (list): int FWER-significant region indices

    Raises:
        RuntimeError: the fit is not cached, so reading it would refit.
    """
    kwargs = dict(parent_uid=uid, n_perm_fwer=N_PERM_FWER,
                  n_perm_inner=N_PERM_INNER, alpha_fwer=ALPHA_FWER,
                  cluster_mode=ClusterMode.FOCUS)
    if not glow_fit_for_prune.check_call_in_cache(None, **kwargs):
        raise RuntimeError(f'no cached GLOW fit for parent_uid {uid}; build '
                           'the prune cache before scoring against it')
    children, _, sig_reg_list = glow_fit_for_prune(None, **kwargs)
    return children, sig_reg_list


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


def dp_pass(children, num_vox, size, over, lam: float, k_max: int, sig=None):
    """Maximize 2*tp - lam*vol over antichains of at most j nodes, j <= k_max.

    Only a node whose two children both hold effect voxels needs the budget
    split; elsewhere one side contributes nothing and the parent inherits its
    sibling's vector, which is what keeps the pass linear in the plant rather
    than in the tree. A node that no selection can profit from is collapsed
    back onto the shared zero vector, so gating on sig shrinks the work
    instead of leaving empty subtrees to convolve.

    Args:
        children (np.array): (num_internal, 2) child index pairs
        num_vox (int): leaf count
        size, over (np.array): per-node volume and overlap (node_counts)
        lam (float): the Dice level the surrogate is tested against
        k_max (int): the largest region budget to solve
        sig (np.array | None): (num_vox + num_internal,) boolean, True where a
            node may be selected. None (default) allows every node; a gated
            node is still traversed, so its descendants stay selectable.

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
            elif (c < num_vox and over[c] > 0
                  and (sig is None or sig[c])):
                pair.append(leaf)
            else:
                pair.append(zero)
        a, b = pair
        if b is zero:
            out = a
        elif a is zero:
            out = b
        else:
            out = _combine(a, b, k_max)
        take = 2.0 * over[v] - lam * size[v]
        if take > 0 and (sig is None or sig[v]):
            rec = (take, int(over[v]), int(size[v]))
            out = [out[0]] + [rec if take > o[0] else o for o in out[1:]]

        # value is non-decreasing in budget, so an all-zero tail means no
        # selection in this subtree pays; hand the parent the shared zero
        # vector rather than a copy, and it skips the convolution entirely.
        vec[v] = zero if out[k_max][0] == 0.0 else out

    # cluster returns a forest on a non-contiguous mask, so whatever vectors
    # survive unclaimed are the roots and their budgets still trade off.
    root = zero
    for other in vec.values():
        if other is not zero:
            root = other if root is zero else _combine(root, other, k_max)
    return [(tp, vol) for _, tp, vol in root]


def oracle_curve(children, num_vox, plant, k_max: int, lam_list, sig=None):
    """Score the best-Dice antichain of at most k nodes, for every k.

    The seed grid is swept first, then each budget's incumbent Dice is fed
    back as the next lam (Dinkelbach 1967). A surrogate pass at lam either
    returns a selection beating lam, which becomes the next incumbent, or
    proves lam optimal, so the iteration reaches the exact optimum from any
    feasible start and the grid only has to supply one.

    Args:
        children (np.array): (num_internal, 2) child index pairs
        num_vox (int): leaf count
        plant (np.array): (num_vox,) boolean, the planted support
        k_max (int): the largest region budget to solve
        lam_list (np.array): (n_lam,) Dice levels to seed the iteration
        sig (np.array | None): selectable-node gate, see dp_pass

    Returns:
        list: one {k, dice, ppv, sens, n_vox} per budget with a non-empty
            selection
    """
    size, over = node_counts(children, num_vox, plant)
    n_eff = int(plant.sum())
    best = [(-1.0, 0, 0)] * (k_max + 1)

    def _absorb(lam: float) -> bool:
        """Keep whatever one surrogate pass improves; True if anything did."""
        gain = False
        for j, (tp, vol) in enumerate(dp_pass(children, num_vox, size, over,
                                              lam, k_max, sig=sig)):
            dice = 2.0 * tp / (vol + n_eff) if vol else 0.0
            if vol and dice > best[j][0] + _DICE_TOL:
                best[j] = (dice, tp, vol)
                gain = True
        return gain

    for lam in lam_list:
        _absorb(lam)
    for _ in range(_DICE_MAX_ITER):
        lam_set = {best[j][0] for j in range(1, k_max + 1) if best[j][0] > 0}
        # a list, not a generator: every budget's lam has to run, and any() on
        # a generator would stop at the first that improved something
        if True not in [_absorb(lam) for lam in lam_set]:
            break
    return [dict(k=j, dice=dice, ppv=tp / vol, sens=tp / n_eff, n_vox=vol)
            for j, (dice, tp, vol) in enumerate(best) if j and vol]


def cell_curves(cell, k_max: int):
    """Score both candidate pools on one cell.

    Args:
        cell (pandas.Series): a load_cells row (uid, eff_hash)
        k_max (int): the largest region budget to solve

    Returns:
        list[dict]: one {pool, k, dice, ppv, sens, n_vox} per pool and budget.
            A pool with no region reaching the plant scores 0 at every budget,
            with n_vox NaN (see the body).
    """
    exp, mask = load_effect_cell(cell.eff_hash)
    num_vox = exp.y.shape[2]
    children, sig_reg_list = load_fit(cell.uid)

    # a planted voxel outside the analysed mask indexes as -1, which would
    # otherwise plant the last voxel of the array
    idx = exp.mask_idx[mask]
    plant = np.zeros(num_vox, dtype=bool)
    plant[idx[idx > -1]] = True

    sig = np.zeros(num_vox + len(children), dtype=bool)
    sig[sig_reg_list] = True

    row_list = []
    for pool in POOL_LIST:
        gate = sig if pool == 'sig' else None
        rec_list = oracle_curve(children, num_vox, plant, k_max, LAM_LIST,
                                sig=gate)

        # a pool holding regions but none that reach the plant ties every
        # selection at Dice 0. The tie leaves the volume undetermined, but not
        # the precision: whatever a rule picks here scores 0, so the trial is
        # recorded at 0 rather than dropped, which would otherwise score the
        # oracle over a different trial set than the rule it bounds.
        if not rec_list and gate is not None and gate.any():
            rec_list = [dict(k=j, dice=0.0, ppv=0.0, sens=0.0, n_vox=np.nan)
                        for j in range(1, k_max + 1)]
        for rec in rec_list:
            row_list.append(dict(pool=pool, **rec))
    return row_list


def compute(k_max: int):
    """Run every cell of both sources and return the per-trial curves.

    Returns:
        pandas.DataFrame: one row per (source, llr, seed, pool, k)
    """
    row_list = []
    for source in SOURCE_LIST:
        for _, cell in load_cells(source).iterrows():
            sec = time.time()
            for rec in cell_curves(cell, k_max):
                row_list.append(dict(source=source, llr=cell.llr,
                                     seed=cell.seed, **rec))
            print(f'{source} llr {cell.llr:.4f} seed {cell.seed:2d} '
                  f'[{time.time() - sec:.1f}s]', flush=True)
    return pd.DataFrame(row_list)


def greedy_trials(source: str):
    """Read GLOW's own per-trial operating point off the prune cache.

    Args:
        source (str): HCP or WGN.

    Returns:
        pandas.DataFrame: columns {llr, seed, n_reg, ppv}, one row per trial
            in which greedy discovered something. PPV is undefined on a
            trial that discovered nothing, so those trials are dropped
            rather than scored as zero; plot restricts the oracle curves to
            the same trials, which is what makes the vertical gaps between
            the three quantities a decomposition of one number.
    """
    df = _pool_df(source)
    df = df[df['run_prune.in.rule'] == 'greedy']
    tp = df['run_prune.out.score.tp']
    fp = df['run_prune.out.score.fp']
    df = df.assign(ppv=tp / (tp + fp).replace(0, np.nan),
                   n_reg=df['run_prune.out.score.n_selected'])
    return df.loc[tp + fp > 0, ['llr', 'seed', 'n_reg', 'ppv']]


def plot(df, out: pathlib.Path) -> None:
    """Draw mean PPV against region budget, one panel per image source.

    Three legends' worth of information in two: colour carries the effect
    strength, and line style carries which pool an oracle drew from (with
    GLOW's own marker beside them), so a reader does not have to infer either
    from the caption.
    """
    sns.set_theme(style='whitegrid', context='paper')
    ramp = plt.get_cmap('viridis')(np.linspace(0, 0.88, df.llr.nunique()))
    source_list = [s for s in SOURCE_LIST if s in set(df.source)]
    style = dict(sig=dict(ls='-', marker='.'),
                 tree=dict(ls='--', marker='', lw=1.1))

    # the full 17.8cm text block, as the paper's other multi-panel figures
    # take, so the figure is included 1:1 with no font rescaling
    fig, axes = plt.subplots(1, len(source_list), figsize=(FIG_W, 0.47 * FIG_W),
                             sharey=True, squeeze=False,
                             constrained_layout=True)
    handle_list = []
    for ax, source in zip(axes[0], source_list):
        sub = df[df.source == source]
        greedy = greedy_trials(source)
        for llr, colour in zip(sorted(df.llr.unique()), ramp):
            near = min(greedy.llr.unique(), key=lambda v: abs(v - llr))
            found = greedy[greedy.llr == near]
            cell = sub[(sub.llr == llr) & sub.seed.isin(set(found.seed))]
            for pool in POOL_LIST:
                curve = cell[cell.pool == pool].groupby('k').ppv.mean()
                ax.plot(curve.index, curve.values, color=colour,
                        **style[pool])
            ax.plot(found.n_reg.mean(), found.ppv.mean(), marker='o', ms=8,
                    color=colour, mec='0.2', mew=0.7, ls='none', zorder=5)
        ax.set_title(source)
        ax.set_xlabel('regions output')
    for llr, colour in zip(sorted(df.llr.unique()), ramp):
        handle_list.append(Line2D([], [], color=colour, marker='.',
                                  label=f'{llr:.3g}'))
    mark_list = [Line2D([], [], color='0.35', label='Oracle, significant',
                        **style['sig']),
                 Line2D([], [], color='0.35', label='Oracle, whole tree',
                        **style['tree']),
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
                      handlelength=2.2, fontsize='small')
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
