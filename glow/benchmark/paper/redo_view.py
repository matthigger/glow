"""Interactive REPL: rebuild one benchmark trial and open it in glow:viewer.

A companion to glow.benchmark.paper.compare. Where compare *ranks* a
config's trials by where GLOW underperforms, redo_view *reproduces* a
single trial end to end: it recovers the exact experiment behind one
results.csv row, re-fits the chosen GLOW variant, and launches the
viewer on it.

A results.csv row cannot be inverted on its own -- its ds and extenter
columns are one-way value_id hashes (glow.util.value_id). What makes a
row reproducible is its folder: the folder name is a cache label in
config.CACHE_BY_LABEL, which still holds the real TrialCache (the actual
ds / extenter objects and the seed x effect_llr grid). The row's
trial_hash (the csv index) then pins the trial -- re-derive the cache's
iter_trial() and keep the one whose stable_hash matches. That recovers
the real ds / extenter / seed / effect_llr, from which run._plant
rebuilds exp_eff + the planted mask_target deterministically.

The REPL asks for three things:

  1. the GLOW variant to fit -- focus (GLOW-Focus) or error (GLOW-GLM);
  2. the config (results folder) to draw from, e.g. vba_hcp_famd;
  3. the trial, either by typing its trial_hash or by reusing compare's
     gap ranking to pick from the worst GLOW cases.

It then fits AnalysisGLOW at the config's paper settings and launches
the viewer. The fit is the full permutation analysis (the row's recorded
time_sec is your time estimate), so it is cached to disk the moment it
completes -- before the viewer is launched -- keyed by label + GLOW
variant + trial_hash (see CACHE_DIR). Redoing the same trial again, or
re-running after the viewer fails to launch (e.g. a busy port), reloads
the fit in seconds. Launching the viewer takes over the terminal --
Ctrl+C stops the viewer and exits the process, so it's one redo per run.

Run it interactively:

    python -m glow.benchmark.paper.redo_view

The cached pickle is a plain AnalysisGLOW, so it also opens directly in
the standalone viewer:

    python -m glow.viewer <CACHE_DIR>/<label>__<variant>__<hash>.p.gz \\
        --mask <CACHE_DIR>/<label>__<variant>__<hash>.mask.npy
"""

import gzip
import pathlib
import pickle
import sys

import numpy as np
from platformdirs import user_cache_dir

import glow.benchmark
from glow.analysis import AnalysisGLOW
from glow.util import stable_hash

from . import compare
from .config import ANALYSIS_DICT, CACHE_BY_LABEL
from .run import _plant


# Fitted-analysis cache. Re-computing a trial is the expensive step
# (minutes), so the first fit is pickled here -- keyed by label + GLOW
# variant + trial_hash -- and reloaded on the next redo. It lives in the
# user cache dir (regenerable, tens of MB, deliberately not the
# Dropbox-synced results dir); delete the folder to force re-fits.
CACHE_DIR = pathlib.Path(user_cache_dir('glow', 'glow_author')) / 'redo_view'


def _cache_paths(label: str, glow_label: str, trial_hash: str):
    """Return the (analysis, mask) cache paths for one fitted trial.

    The analysis path is a gzipped pickle of the AnalysisGLOW (so it is
    also directly loadable by `python -m glow.viewer`); the mask path is
    the planted target mask as .npy. Both are keyed by label + GLOW
    variant + trial_hash so focus and GLM-error fits of the same trial
    don't collide.

    Args:
        label (str): results folder name == CACHE_BY_LABEL key.
        glow_label (str): GLOW variant, 'GLOW-Focus' or 'GLOW-GLM'.
        trial_hash (str): the row's trial_hash.

    Returns:
        ana_path (pathlib.Path): the AnalysisGLOW .p.gz path.
        mask_path (pathlib.Path): the target-mask .npy path.
    """
    key = f'{label}__{glow_label}__{trial_hash}'.replace('/', '_')
    return CACHE_DIR / f'{key}.p.gz', CACHE_DIR / f'{key}.mask.npy'


def _load_cached(ana_path, mask_path):
    """Load a cached AnalysisGLOW + target mask from disk.

    Args:
        ana_path (pathlib.Path): gzipped-pickle AnalysisGLOW path.
        mask_path (pathlib.Path): target-mask .npy path (may be absent).

    Returns:
        ana (AnalysisGLOW): the unpickled analysis.
        mask_target (np.array | None): (X, Y, Z) bool mask, or None when
            no mask file is present.
    """
    # the cache holds only pickles this module wrote itself, under the
    # local user cache dir -- same trust model as viewer._load_analysis
    with gzip.open(ana_path, 'rb') as f:
        ana = pickle.load(f)
    mask_target = np.load(mask_path) if mask_path.exists() else None
    return ana, mask_target


def _save_cached(ana, mask_target, ana_path, mask_path) -> None:
    """Pickle a fitted AnalysisGLOW + target mask to the cache.

    Args:
        ana (AnalysisGLOW): the fitted analysis to cache.
        mask_target (np.array): (X, Y, Z) bool planted-effect support.
        ana_path (pathlib.Path): destination for the gzipped pickle.
        mask_path (pathlib.Path): destination for the mask .npy.
    """
    ana_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ana_path, 'wb') as f:
        pickle.dump(ana, f)
    np.save(mask_path, mask_target)


def recover_trial(label: str, trial_hash: str) -> dict:
    """Recover the exact trial-kwargs dict behind one results.csv row.

    Re-derives the label's TrialCache iteration and returns the trial
    whose stable_hash matches trial_hash, so the opaque ds / extenter
    hash columns become the real objects again.

    Args:
        label (str): results folder name == CACHE_BY_LABEL key.
        trial_hash (str): the row's trial_hash (results.csv index).

    Returns:
        the trial dict {ds, extenter, seed, effect_llr} for that row.

    Raises:
        KeyError: label is not a known cache.
        LookupError: no trial in the cache matches trial_hash.
    """
    cache, _ = CACHE_BY_LABEL[label]
    for trial in cache.iter_trial():
        if stable_hash(trial) == trial_hash:
            return trial
    raise LookupError(
        f'no trial in cache {label!r} matches trial_hash {trial_hash!r}')


def _recorded_time(df, trial_hash: str, glow_label: str):
    """Return the row's recorded time_sec for this trial + GLOW variant.

    A rough time estimate for the re-fit (the analysis is deterministic
    in cost, not just in output). None when the variant wasn't scored.

    Args:
        df: a config's results, indexed by trial_hash, with label and
            time_sec columns.
        trial_hash (str): the trial to look up.
        glow_label (str): the GLOW label, e.g. 'GLOW-Focus'.

    Returns:
        the recorded wall time in seconds, or None if unavailable.
    """
    if trial_hash not in df.index:
        return None
    sub = df.loc[[trial_hash]]
    sub = sub[sub['label'] == glow_label]
    if sub.empty or 'time_sec' not in sub.columns:
        return None
    return float(sub['time_sec'].iloc[0])


def _enter_hash(df):
    """Prompt for a trial_hash present in df (exact or unique prefix).

    Args:
        df: a config's results, indexed by trial_hash.

    Returns:
        the matched trial_hash, or None to go back / quit.
    """
    hashes = [str(h) for h in df.index.unique()]
    while True:
        raw = input('  trial_hash (q to go back): ').strip().lower()
        if raw in ('', 'q', 'quit'):
            return None
        if raw in hashes:
            return raw
        pref = [h for h in hashes if h.startswith(raw)]
        if len(pref) == 1:
            print(f'  -> {pref[0]}')
            return pref[0]
        if len(pref) > 1:
            print(f'  ? ambiguous prefix matches {len(pref)} trials')
        else:
            print('  ? no row with that trial_hash in this config')


def _pick_from_ranking(df, glow_label: str):
    """Rank trials by compare's GLOW-vs-best gap, then pick one to redo.

    Reuses compare.summarize / compare.print_summary to show the worst
    GLOW cases, then offers the top N as a menu.

    Args:
        df: a config's results, indexed by trial_hash.
        glow_label (str): the chosen GLOW label, e.g. 'GLOW-Focus'.

    Returns:
        the chosen trial_hash, or None to go back / quit.
    """
    feats = ['dice', 'sens', 'ppv', 'spec']
    m = compare.choose('Sort trials by which feature?', feats)
    if m < 0:
        return None
    metric = feats[m]

    summary = compare.summarize(df, glow_label, metric)
    summary = summary.dropna(subset=['gap'])
    compare.print_summary(summary, glow_label, metric)

    top = summary.head(compare.N_SHOW)
    options = []
    for th, row in top.iterrows():
        desc = []
        if 'effect_llr' in top.columns:
            desc.append(f"llr={row['effect_llr']:.3g}")
        if 'seed' in top.columns:
            desc.append(f"seed={int(row['seed'])}")
        if 'vox_effect' in top.columns:
            desc.append(f"vox={int(row['vox_effect'])}")
        options.append(f"{th}  gap={row['gap']:+.4f}  " + ' '.join(desc))

    i = compare.choose('Redo which trial?', options)
    if i < 0:
        return None
    return str(top.index[i])


def choose_trial_hash(df, glow_label: str):
    """Choose a trial_hash by direct entry or by compare's gap ranking.

    Args:
        df: a config's results, indexed by trial_hash.
        glow_label (str): the chosen GLOW label, e.g. 'GLOW-Focus'.

    Returns:
        the chosen trial_hash, or None to go back / quit.
    """
    mode = compare.choose(
        'How would you like to pick the trial?',
        ['enter a trial_hash',
         'rank by metric gap (worst GLOW cases)'])
    if mode < 0:
        return None
    if mode == 0:
        return _enter_hash(df)

    # the gap ranking compares GLOW against the non-GLOW field, so it
    # needs the chosen GLOW variant's rows in this config's csv
    if glow_label not in set(df['label']):
        print(f'  this config has no {glow_label} rows -- cannot rank by'
              f' gap (enter a trial_hash instead).')
        return None
    return _pick_from_ranking(df, glow_label)


def reproduce(label: str, glow_label: str, trial_hash: str,
              verbose: bool = True, use_cache: bool = True):
    """Rebuild one trial's planted experiment and fit the chosen GLOW.

    The fitted AnalysisGLOW + mask are cached under CACHE_DIR (keyed by
    label + glow_label + trial_hash) the moment the fit completes, before
    this function returns. So a re-redo of the same trial -- or a re-run
    after the viewer fails to launch (e.g. a busy port) -- reloads the
    minutes-long fit in seconds instead of recomputing it.

    Args:
        label (str): results folder name == CACHE_BY_LABEL key.
        glow_label (str): GLOW variant to fit, 'GLOW-Focus' or 'GLOW-GLM'
            (its ANALYSIS_DICT recipe sets cluster_mode and perm counts).
        trial_hash (str): the row's trial_hash, pinning the trial.
        verbose (bool): print progress and the GLOW fit log.
        use_cache (bool): reuse a cached fit when present (and write one
            after fitting); False forces a fresh fit and overwrite.

    Returns:
        ana (AnalysisGLOW): the fitted analysis to pass to viewer.launch.
        mask_target (np.array): (X, Y, Z) bool planted-effect support, the
            launch mask_target argument.
    """
    ana_path, mask_path = _cache_paths(label, glow_label, trial_hash)
    if use_cache and ana_path.exists():
        if verbose:
            print(f'\n  redo {label} / {trial_hash}  ->  {glow_label}')
            print(f'  loading cached fit <- {ana_path}')
        return _load_cached(ana_path, mask_path)

    trial = recover_trial(label, trial_hash)

    cache, _ = CACHE_BY_LABEL[label]
    df = glow.benchmark.load_results_csv(cache.folder / 'results.csv')
    t_rec = _recorded_time(df, trial_hash, glow_label)

    if verbose:
        print(f'\n  redo {label} / {trial_hash}  ->  {glow_label}')
        print(f"    seed={trial['seed']}  effect_llr={trial['effect_llr']:.4g}")
        if t_rec is not None:
            print(f'    (recorded fit took ~{t_rec:.0f}s; re-fit is similar)')
        print('  rebuilding planted experiment ...')

    _exp, exp_eff, mask_target = _plant(
        trial['ds'], trial['extenter'], trial['effect_llr'], trial['seed'])

    kw = dict(ANALYSIS_DICT[glow_label][1])
    if verbose:
        print(f'  fitting AnalysisGLOW({kw}) on y={exp_eff.y.shape} ...')
    ana = AnalysisGLOW(exp=exp_eff, **kw).fit(verbose=verbose)

    _save_cached(ana, mask_target, ana_path, mask_path)
    if verbose:
        print(f'  cached fit -> {ana_path}')
    return ana, mask_target


def main() -> None:
    """Run the redo_view REPL: pick GLOW variant, then loop config + trial."""
    configs = compare.find_configs()
    if not configs:
        print('no configs with a results.csv under '
              f'{glow.benchmark.get_path_result()}')
        return

    g = compare.choose('Which GLOW method to fit?',
                       ['focus (GLOW-Focus)', 'error (GLOW-GLM)'])
    if g < 0:
        return
    glow_label = compare.GLOW_LABEL['focus' if g == 0 else 'error']

    while True:
        labels = [name for name, _ in configs]
        c = compare.choose('Which config to draw a trial from?', labels)
        if c < 0:
            return
        label, csv = configs[c]
        df = glow.benchmark.load_results_csv(csv)

        trial_hash = choose_trial_hash(df, glow_label)
        if trial_hash is None:
            continue

        ana, mask_target = reproduce(label, glow_label, trial_hash)

        # launch blocks and os._exit's on Ctrl+C, so this is the terminal
        # action -- the loop above only re-runs if the user backed out
        # before committing to a fit.
        from glow.viewer import launch
        launch(ana, mask_target=mask_target)
        return


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)
