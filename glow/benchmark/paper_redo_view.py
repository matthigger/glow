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
import numpy as np

import glow
from glow.benchmark.file import get_path_result, load_update_all
from glow.benchmark.paper_config import (
    ANALYSES, CONFIG_BY_LABEL, make_config, ana_kwargs_dict_vba,
)


# ---------------------------------------------------------------------------
# Phase 1: find worst cases
# ---------------------------------------------------------------------------

def _sort_options(glow_label, baseline_label):
    """Build sort-option dict keyed on metric for a given pair of methods."""
    return {
        'dice':  (f'{baseline_label}_dice',  f'{glow_label}_dice',
                  f'Dice difference ({glow_label} − {baseline_label})'),
        'sens': (f'{baseline_label}_sens', f'{glow_label}_sens',
                 f'Sensitivity difference ({glow_label} − {baseline_label})'),
        'spec': (f'{baseline_label}_spec', f'{glow_label}_spec',
                 f'Specificity difference ({glow_label} − {baseline_label})'),
    }


def _load_and_rank(config_label, glow_label='GLOW',
                   baseline_label='VBA-TFCE', sort_metric='dice', top_n=20):
    """Load results and rank experiments by gap between two methods.

    Parameters
    ----------
    config_label : str
    glow_label : str
        Primary method to explore in the viewer.
    baseline_label : str
        Comparison method (gap = glow - baseline; most negative = worst).
    sort_metric : str
        One of 'dice', 'sens', 'spec'.
    top_n : int

    Returns
    -------
    ranking : pd.DataFrame
        Sorted by gap ascending (most negative = glow_label worst).
    """
    df, folder, _ = load_update_all(config_label, verbose=False)
    if df.empty:
        raise SystemExit(
            f'No results found for "{config_label}".  '
            f'Run the benchmark first (python glow/benchmark/paper.py).')

    labels = set(df['label'].unique())
    if glow_label not in labels:
        raise SystemExit(f'No {glow_label} results in "{config_label}".')
    if baseline_label not in labels:
        raise SystemExit(f'No {baseline_label} results in "{config_label}".')

    df = df[df['label'].isin([glow_label, baseline_label])].copy()

    parts = []
    for metric in ('dice', 'sens', 'spec'):
        piv = df.pivot_table(index=['seed', 'effect_llr'], columns='label',
                             values=metric, aggfunc='first')
        piv.columns = [f'{col}_{metric}' for col in piv.columns]
        parts.append(piv)

    merged = parts[0].join(parts[1:]).reset_index()
    merged = merged.dropna(subset=[f'{glow_label}_dice'])

    opts = _sort_options(glow_label, baseline_label)
    base_col, glow_col, _ = opts[sort_metric]
    merged['gap'] = merged[glow_col] - merged[base_col]

    ranking = (merged
               .sort_values('gap', ascending=True)
               .head(top_n)
               .reset_index(drop=True))
    return ranking


def _print_ranking(ranking, glow_label='GLOW', baseline_label='VBA-TFCE',
                   sort_label=''):
    """Pretty-print the ranking table with a 0-based index column."""
    stat_cols = []
    for method in (glow_label, baseline_label):
        for metric in ('dice', 'sens', 'spec'):
            col = f'{method}_{metric}'
            if col in ranking.columns:
                stat_cols.append(col)
    stat_cols.append('gap')

    title = sort_label or f'{glow_label} − {baseline_label}'
    print(f'\n  Top {len(ranking)} worst cases ({title}):')
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
    except (json.JSONDecodeError, OSError):
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
    except (OSError, gzip.BadGzipFile, pickle.UnpicklingError,
            ModuleNotFoundError, AttributeError, EOFError):
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


def _rerun_and_view(config_label, seed, effect_llr, glow_label='GLOW',
                    use_cache=False, port=8050):
    """Re-run a single experiment, save artifacts, and launch viewer."""
    # try cached result
    if use_cache:
        ana, effect_mask = _load_cache()
        if ana is not None:
            print(f'  Loading cached result ...')
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
    from glow.benchmark.runner import RunAna
    config = make_config(
        label=f'_redo_{config_label}',
        source=source,
        runner=RunAna(ana_kwargs_dict_vba),
    )

    print(f'  Re-running ({glow_label}): seed={seed}, '
          f'effect_llr={effect_llr:.4f}, source={source}')

    # build experiment + effect
    print('  Building experiment ...')
    exp_eff, effect = config.get_exp_eff(seed=seed, effect_llr=effect_llr)
    print(f'    y.shape={exp_eff.y.shape}, '
          f'effect: {int(effect.mask.sum())} voxels')

    # run analysis with the selected GLOW variant
    glow_kwargs = dict(ANALYSES[glow_label])
    glow_kwargs['n_jobs_perm'] = -1
    print(f'  Running AnalysisGLOW[{glow_label}] '
          f'(n_perm_fwer={glow_kwargs["n_perm_fwer"]}, parallel) ...')
    ana = glow.analysis.AnalysisGLOW(exp_eff, verbose=True, **glow_kwargs)
    print(f'  Found {len(ana.effect_list)} effects')

    # compute overall score
    mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
    for eff in ana.effect_list:
        mask_pred |= eff.mask
    dice, sens, spec = glow.mask.get_score(
        mask_pred=mask_pred,
        mask_target=effect.mask,
        mask_active=exp_eff.mask_idx > -1)
    print(f'  Score: dice={dice:.4f}, sens={sens:.4f}, spec={spec:.4f}')

    # save (overwrites any previous cache)
    _save_cache(config_label, seed, effect_llr, exp_eff, effect, ana)

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


def _get_method_labels(config_label):
    """Return all labels the config's runner emits."""
    from glow.benchmark.runner import RunAna
    cfg = CONFIG_BY_LABEL.get(config_label)
    if cfg is None or not isinstance(cfg.runner, RunAna):
        return []
    return list(cfg.runner.ana_kwargs_dict.keys())


def main():
    # -- step 1: choose a config ------------------------------------------
    from glow.benchmark.runner import RunAna
    available = []
    for label, cfg in sorted(CONFIG_BY_LABEL.items()):
        if isinstance(cfg.runner, RunAna):
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

    # -- step 1b: choose primary method (re-run in viewer) -----------------
    all_methods = _get_method_labels(config_label)
    glow_methods = [m for m in all_methods if m.startswith('GLOW')]
    if len(glow_methods) == 0:
        raise SystemExit(f'No GLOW variants found in "{config_label}".')

    print('  The primary method is the one that will be re-run and')
    print('  explored in the viewer.  Choose a GLOW variant:')
    print()
    for i, v in enumerate(glow_methods):
        print(f'  {i:3d}  {v}')
    print()
    if len(glow_methods) == 1:
        glow_label = glow_methods[0]
        print(f'  → {glow_label} (only one available)\n')
    else:
        gv_choice = _prompt_choice('  Select primary method [#]: ',
                                   len(glow_methods))
        glow_label = glow_methods[gv_choice]
        print(f'  → {glow_label}\n')

    # -- step 1c: choose baseline method (for ranking) ---------------------
    baseline_options = [m for m in all_methods if m != glow_label]
    if len(baseline_options) == 0:
        raise SystemExit('Only one method in this config — nothing to '
                         'compare against.')

    print('  The baseline method is used for sorting: experiments are')
    print(f'  ranked by where {glow_label} underperforms the baseline')
    print('  the most, so you can explore the hardest cases.')
    print()
    for i, v in enumerate(baseline_options):
        print(f'  {i:3d}  {v}')
    print()
    if len(baseline_options) == 1:
        baseline_label = baseline_options[0]
        print(f'  → {baseline_label} (only one available)\n')
    else:
        bl_choice = _prompt_choice('  Select baseline method [#]: ',
                                   len(baseline_options))
        baseline_label = baseline_options[bl_choice]
        print(f'  → {baseline_label}\n')

    # -- step 1d: choose sort metric ----------------------------------------
    opts = _sort_options(glow_label, baseline_label)
    sort_keys = list(opts.keys())
    print('  Sort ranking by:')
    for i, key in enumerate(sort_keys):
        _, _, desc = opts[key]
        print(f'  {i:3d}  {desc}')
    print()
    sort_choice = _prompt_choice('  Sort metric [#]: ', len(sort_keys))
    sort_metric = sort_keys[sort_choice]
    _, _, sort_label = opts[sort_metric]
    print(f'  → {sort_label}\n')

    # -- step 2: load & rank -----------------------------------------------
    ranking = _load_and_rank(config_label, glow_label=glow_label,
                             baseline_label=baseline_label,
                             sort_metric=sort_metric, top_n=20)
    _print_ranking(ranking, glow_label=glow_label,
                   baseline_label=baseline_label, sort_label=sort_label)

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
    _rerun_and_view(config_label, seed, effect_llr,
                    glow_label=glow_label, use_cache=use_cache)


if __name__ == '__main__':
    main()
