"""Re-run a single benchmark experiment and launch the viewer.

Reads results from a completed benchmark, identifies where GLOW
underperforms VBA the most, re-runs that experiment locally with parallel
permutations, saves the artifacts, and launches the interactive viewer.

Interactive — prompts the user to pick a config then a case to explore.

Usage::

    python glow/benchmark/paper_redo_view.py
"""

import gzip
import json

import cloudpickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import glow
from glow.benchmark.file import get_path_result, load_update_all
from glow.benchmark.paper_config import (
    ANALYSES, CONFIG_BY_LABEL, make_config, ana_kwargs_dict_vba,
)
from glow.experiment.mancova import decompose, llr_from_ysum_yout
from glow.experiment.prune import _compute_perm_deltas
from glow.graph import get_f1_sens_spec, iter_topo


# ---------------------------------------------------------------------------
# Phase 1: find worst cases
# ---------------------------------------------------------------------------

def _load_and_rank(config_label, top_n=20):
    """Load results and rank experiments by GLOW underperformance vs VBA-TFCE.

    Returns
    -------
    ranking : pd.DataFrame
        Sorted by gap descending (top_n rows).  Columns include seed,
        effect_llr, per-method f1/sens/spec, and gap.
    """
    df, folder, _ = load_update_all(config_label, verbose=False)
    if df.empty:
        raise SystemExit(
            f'No results found for "{config_label}".  '
            f'Run the benchmark first (python glow/benchmark/paper.py).')

    labels = set(df['label'].unique())
    if 'GLOW' not in labels:
        raise SystemExit(f'No GLOW results in "{config_label}".')
    if 'VBA-TFCE' not in labels:
        raise SystemExit(f'No VBA-TFCE results in "{config_label}".')

    # keep only GLOW and VBA-TFCE
    df = df[df['label'].isin(['GLOW', 'VBA-TFCE'])].copy()

    # pivot each metric separately, then merge
    parts = []
    for metric in ('f1', 'sens', 'spec'):
        piv = df.pivot_table(index=['seed', 'effect_llr'], columns='label',
                             values=metric, aggfunc='first')
        piv.columns = [f'{col}_{metric}' for col in piv.columns]
        parts.append(piv)

    merged = parts[0].join(parts[1:]).reset_index()
    merged = merged.dropna(subset=['GLOW_f1'])

    # gap = VBA-TFCE f1 − GLOW f1
    merged['gap'] = merged['VBA-TFCE_f1'] - merged['GLOW_f1']

    ranking = (merged
               .sort_values('gap', ascending=False)
               .head(top_n)
               .reset_index(drop=True))
    return ranking


def _print_ranking(ranking):
    """Pretty-print the ranking table with a 0-based index column."""
    # columns: GLOW stats, then VBA-TFCE stats, then gap
    stat_cols = []
    for method in ('GLOW', 'VBA-TFCE'):
        for metric in ('f1', 'sens', 'spec'):
            col = f'{method}_{metric}'
            if col in ranking.columns:
                stat_cols.append(col)
    stat_cols.append('gap')

    print(f'\n  Top {len(ranking)} worst cases (VBA-TFCE f1 − GLOW f1):')
    sep = '  ' + '-' * (8 + 12 + len(stat_cols) * 12)
    print(sep)
    header = '  {:>3s}  {:>4s}  {:>10s}'.format('#', 'seed', 'effect_llr')
    for c in stat_cols:
        header += f'  {c:>10s}'
    print(header)
    print(sep)

    for idx, (_, row) in enumerate(ranking.iterrows()):
        line = f'  {idx:3d}  {int(row["seed"]):4d}  {row["effect_llr"]:10.4f}'
        for c in stat_cols:
            line += f'  {row[c]:10.4f}'
        print(line)
    print()


# ---------------------------------------------------------------------------
# Phase 2: cache, re-run, and view
# ---------------------------------------------------------------------------

_CACHE_DIR = 'redo_view'
_MANIFEST = 'manifest.json'


def _cache_dir():
    return get_path_result() / _CACHE_DIR


def _load_manifest():
    """Load the cached manifest (lightweight, no pickles).

    Returns the manifest dict or None.
    """
    manifest_path = _cache_dir() / _MANIFEST
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_cache():
    """Load the cached analysis and effect from disk.

    Returns (ana, effect_mask) or (None, None).
    """
    d = _cache_dir()
    try:
        with gzip.open(d / 'analysis.p.gz', 'rb') as f:
            ana = pickle.load(f)
        with gzip.open(d / 'effect.p.gz', 'rb') as f:
            effect = pickle.load(f)
        return ana, effect.mask
    except Exception:
        return None, None


def _save_cache(config_label, seed, effect_llr, exp_eff, effect, ana):
    """Save artifacts and a manifest recording the experiment identity."""
    out = _cache_dir()
    out.mkdir(exist_ok=True, parents=True)

    manifest = {'config': config_label, 'seed': seed, 'effect_llr': effect_llr}
    with open(out / _MANIFEST, 'w') as f:
        json.dump(manifest, f)

    for name, obj in [('experiment', exp_eff),
                      ('effect', effect),
                      ('analysis', ana)]:
        path = out / f'{name}.p.gz'
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f'  Saved {path}')


def _generate_delta_histograms(ana, mask_target, n_top=20, n_perm=200):
    """Generate per-page PDF of permutation-delta histograms.

    For each of the *n_top* highest-F1 significant internal nodes, runs
    *n_perm* permutations and plots the delta distribution with mu and
    the mu + (k-1)*lam_geom threshold.

    Saves delta_histograms.pdf next to the redo-view cache.
    """
    exp = ana.exp
    children = ana.children
    num_vox = exp.y.shape[2]
    dp_info = getattr(ana, 'dp_info', {})
    sig_reg_list = dp_info.get('sig_reg_list', [])
    sc_children = dp_info.get('subgraph_children', {})
    compared_to = dp_info.get('prune_compared_to', {})

    if not sig_reg_list:
        print('  No significant regions — skipping histograms.')
        return

    # F1 scores for ranking
    f1, _, _ = get_f1_sens_spec(
        mask=mask_target, mask_idx=exp.mask_idx, children=children)

    # find internal SCGraph nodes among significant regions, ranked by F1
    internal = [n for n in sig_reg_list if sc_children.get(n, [])]
    internal.sort(key=lambda n: -f1[n])
    nodes = internal[:n_top]

    if not nodes:
        print('  No internal significant nodes — skipping histograms.')
        return

    sizes = ana.size.astype(float)
    stat = np.nan_to_num(ana.stat.astype(float) if ana.stat.ndim == 1
                         else ana.stat[0].astype(float),
                         nan=0.0, posinf=0.0, neginf=0.0)
    q_tup = decompose(x=exp.x, contrast=exp.contrast)

    exp_eff = dp_info.get('exp_eff', 3)
    lam_geom = np.log(1 + 1 / exp_eff)

    # leaf cache
    _leaf_cache = {}

    def _get_leaves(node):
        if node not in _leaf_cache:
            _leaf_cache[node] = np.array(sorted(
                iter_topo(children=children, num_leaf=num_vox,
                          node_start=node, only_leaf=True)), dtype=int)
        return _leaf_cache[node]

    out_path = _cache_dir() / 'delta_histograms.pdf'
    out_path.parent.mkdir(exist_ok=True, parents=True)

    with PdfPages(out_path) as pdf:
        for i, node in enumerate(nodes):
            child_ac = compared_to.get(node, list(sc_children.get(node, [])))
            if not child_ac:
                continue

            parent_voxels = _get_leaves(node)
            ac_sizes = [int(sizes[r]) for r in child_ac]
            leftover = int(sizes[node]) - sum(ac_sizes)
            split_val = sum(float(stat[r]) for r in child_ac)
            if leftover > 0:
                ac_voxels = set()
                for r in child_ac:
                    ac_voxels.update(_get_leaves(r).tolist())
                leftover_vox = np.array(
                    [v for v in parent_voxels if v not in ac_voxels], dtype=int)
                y_left = exp.y[:, :, leftover_vox]
                ysum_l = y_left.sum(axis=2)
                yout_l = np.einsum('bin,cin->bc', y_left, y_left)
                lo_llr = llr_from_ysum_yout(ysum_l, yout_l, leftover, q_tup)
                if np.isfinite(lo_llr):
                    split_val += lo_llr
            observed_delta = split_val - float(stat[node])

            deltas = _compute_perm_deltas(
                parent_voxels=parent_voxels,
                ac_sizes=ac_sizes,
                leftover=leftover,
                y=exp.y,
                q_tup=q_tup,
                parent_stat=float(stat[node]),
                n_perm=n_perm,
                seed=node,
            )

            mu = float(np.mean(deltas))
            k = len(child_ac)
            penalty = (k - 1) * lam_geom
            threshold = mu + penalty

            keep = (float(stat[node]) + threshold) >= split_val
            decision = 'KEEP' if keep else 'SPLIT'

            fig, ax = plt.subplots(figsize=(8, 5))

            sigma = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.01
            ax.hist(deltas, bins=30, density=True, alpha=0.6,
                    color='steelblue', edgecolor='white',
                    label=f'perm deltas (n={n_perm})')

            ax.axvline(mu, color='blue', ls=':', lw=1.5,
                       label=f'mu={mu:.3f}')
            ax.axvline(threshold, color='red', ls='--', lw=2,
                       label=f'mu+{k-1}*lam={threshold:.3f} '
                             f'(lam_geom={lam_geom:.3f})')
            ax.axvline(observed_delta, color='green', ls='-', lw=2,
                       label=f'observed delta={observed_delta:.3f}')

            ax.set_xlabel('delta = sum(all group LLRs) - LLR_parent')
            ax.set_ylabel('density')
            ax.set_title(
                f'Node {node} vs {child_ac}   [{decision}]\n'
                f'LLR_parent={stat[node]:.2f}  split_val={split_val:.2f}  '
                f'F1={f1[node]:.3f}  '
                f'(sz={int(sizes[node])}, left={leftover}, k={k})')
            ax.legend(fontsize=9)
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    print(f'  Saved delta histograms ({len(nodes)} pages) → {out_path}')


def _rerun_and_view(config_label, seed, effect_llr, use_cache=False,
                    port=8050):
    """Re-run a single experiment, save artifacts, and launch viewer."""
    # try cached result
    if use_cache:
        ana, effect_mask = _load_cache()
        if ana is not None:
            print(f'  Loading cached result ...')
            _generate_delta_histograms(ana, effect_mask)
            from glow.viewer import launch
            launch(ana, mask_target=effect_mask, port=port)
            return
        print('  Cache load failed, re-running ...')

    # determine source from config label
    if 'wgn' in config_label:
        source = 'wgn'
    elif 'hcp' in config_label:
        source = 'hcp'
    else:
        raise SystemExit(
            f'Cannot infer source from "{config_label}".  '
            f'Expected label containing "wgn" or "hcp".')

    # build a Config with exact paper params (DRY)
    config = make_config(
        label=f'_redo_{config_label}',
        source=source,
        run_fnc=None,
        ana_kwargs_dict=ana_kwargs_dict_vba,
    )

    print(f'  Re-running: seed={seed}, effect_llr={effect_llr:.4f}, '
          f'source={source}')

    # build experiment + effect
    print('  Building experiment ...')
    exp_eff, effect = config.get_exp_eff(seed=seed, effect_llr=effect_llr)
    print(f'    y.shape={exp_eff.y.shape}, '
          f'effect: {int(effect.mask.sum())} voxels')

    # run GLOW analysis with parallel permutations
    glow_kwargs = dict(ANALYSES['GLOW'])
    glow_kwargs['n_jobs_perm'] = -1  # use all cores
    print(f'  Running AnalysisGLOW (n_perm_fwer={glow_kwargs["n_perm_fwer"]}, '
          f'parallel) ...')
    ana = glow.experiment.AnalysisGLOW(exp_eff, verbose=True, **glow_kwargs)
    print(f'  Found {len(ana.effect_list)} effects')

    # compute overall score
    mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
    for eff in ana.effect_list:
        mask_pred |= eff.mask
    f1, sens, spec = glow.mask.get_score(
        mask_pred=mask_pred,
        mask_target=effect.mask,
        mask_active=exp_eff.mask_idx > -1)
    print(f'  Score: f1={f1:.4f}, sens={sens:.4f}, spec={spec:.4f}')

    # save (overwrites any previous cache)
    _save_cache(config_label, seed, effect_llr, exp_eff, effect, ana)

    _generate_delta_histograms(ana, effect.mask)

    # launch viewer
    from glow.viewer import launch
    launch(ana, mask_target=effect.mask, port=port)


# ---------------------------------------------------------------------------
# interactive prompts
# ---------------------------------------------------------------------------

def _prompt_choice(prompt, n_options):
    """Ask the user for an integer in [0, n_options).  Returns the int."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            continue
        try:
            idx = int(raw)
        except ValueError:
            print(f'  Please enter a number between 0 and {n_options - 1}.')
            continue
        if 0 <= idx < n_options:
            return idx
        print(f'  Out of range.  Please enter 0–{n_options - 1}.')


def main():
    # -- step 1: choose a config ------------------------------------------
    # only show configs whose results actually exist (run_ana-based, i.e.
    # those that produce per-experiment f1 scores)
    available = []
    for label, cfg in sorted(CONFIG_BY_LABEL.items()):
        if cfg.ana_kwargs_dict is not None:
            available.append(label)

    if not available:
        raise SystemExit('No benchmark configs found in paper_config.py.')

    print('\n  Available benchmark configs:')
    print('  ' + '-' * 40)
    for i, label in enumerate(available):
        print(f'  {i:3d}  {label}')
    print()

    choice = _prompt_choice('  Select a config [#]: ', len(available))
    config_label = available[choice]
    print(f'  → {config_label}\n')

    # -- step 2: load & rank -----------------------------------------------
    ranking = _load_and_rank(config_label, top_n=20)
    _print_ranking(ranking)

    # show cache hint if the last run matches a row in the ranking
    manifest = _load_manifest()
    cached_idx = None
    if (manifest is not None
            and manifest.get('config') == config_label):
        c_seed = manifest.get('seed')
        c_htr = manifest.get('effect_llr')
        for idx, (_, row) in enumerate(ranking.iterrows()):
            if int(row['seed']) == c_seed and row['effect_llr'] == c_htr:
                cached_idx = idx
                break
        if cached_idx is not None:
            print(f'  (last run cached: #{cached_idx}, '
                  f'seed={c_seed}, effect_llr={c_htr})')

    case = _prompt_choice('  Select a case to explore [#]: ',
                          len(ranking))
    row = ranking.iloc[case]
    seed = int(row['seed'])
    effect_llr = float(row['effect_llr'])
    print(f'  → seed={seed}, effect_llr={effect_llr:.4f}\n')

    # -- step 3: re-run & launch viewer ------------------------------------
    use_cache = (cached_idx is not None and case == cached_idx)
    _rerun_and_view(config_label, seed, effect_llr, use_cache=use_cache)


if __name__ == '__main__':
    main()
