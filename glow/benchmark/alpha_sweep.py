"""Post-hoc alpha_FWER sweep over saved analyses.

Reads ``(ana, effect)`` detail pickles from a completed benchmark
config, re-applies a grid of alpha_fwer values via
``ana.rethreshold()``, scores each result against the target effect
with ``score_ana()``, and emits new result JSONs to a sibling sweep
folder.  No permutation work is re-run.

Usage::

    python -m glow.benchmark.alpha_sweep <config_label> \
        [--alphas 0.001 0.005 0.01 0.025 0.05 0.1 0.2] \
        [--cft-pvals 0.0001 0.001 0.01] \
        [--labels GLOW-Focus GLOW-GLM]

For ``AnalysisCET`` rows, ``--cft-pvals`` (if given) sweeps the
cluster-forming threshold pval as a second axis.  Non-CET rows ignore
that flag.

Output schema adds: ``source_uuid``, ``alpha_fwer``, ``cft_pval``
(populated for CET only).  Loadable via
``load_update_all(f'{config_label}_alpha_sweep')``.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

import cloudpickle as pickle
import numpy as np
from tqdm import tqdm

from glow.benchmark.file import OUT, get_path_result, load_update_all
from glow.benchmark.paper_config import CONFIG_BY_LABEL
from glow.benchmark.runner import score_ana


DEFAULT_ALPHAS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2)
SWEEP_SUFFIX = '_alpha_sweep'

# labels eligible by default: any GLOW/VBA/VBA-TFCE/CET emitting analysis
_DEFAULT_LABEL_PREFIXES = ('GLOW', 'VBA', 'CET')


def _sweep_folder(config_label):
    return get_path_result() / f'{config_label}{SWEEP_SUFFIX}'


def _load_source_df(config_label):
    """Consolidate JSON results and return the dataframe + folder.

    ``load_update_all`` deletes the JSON files after building
    ``results.csv``, so the dataframe is the authoritative mapping
    from ``uuid`` -> (label, seed, effect_llr, iter_kw...).
    """
    df, folder, _ = load_update_all(config_label, verbose=False)
    if df.empty:
        raise SystemExit(
            f'no results found for "{config_label}" — run the benchmark '
            f'first.')
    if 'uuid' not in df.columns:
        raise SystemExit(
            f'results for "{config_label}" lack a uuid column; pre-uuid '
            f'benchmark output is not sweep-compatible.')
    return df, folder


def _load_existing_sweep_keys(out_folder):
    """Return set of (source_uuid, alpha_fwer, cft_pval) already emitted.

    Loads both the consolidated CSV and any not-yet-aggregated JSON files
    so a partial sweep can be resumed cleanly.
    """
    keys = set()
    csv_path = out_folder / 'results.csv'
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            cft = row.get('cft_pval')
            if cft is None or (isinstance(cft, float) and math.isnan(cft)):
                cft = None
            keys.add((row['source_uuid'], float(row['alpha_fwer']), cft))
    out_json_folder = out_folder / OUT
    if out_json_folder.exists():
        for f in out_json_folder.glob('*_result.json'):
            with open(f) as fh:
                d = json.load(fh)
            cft = d.get('cft_pval')
            keys.add((d['source_uuid'], float(d['alpha_fwer']), cft))
    return keys


def _write_row(out_folder, row):
    """Write one sweep result JSON to ``<out_folder>/out/``."""
    from glow.benchmark.file import short_uuid
    out_dir = out_folder / OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    uuid_str = short_uuid()
    row = {**row, 'uuid': uuid_str}
    path = out_dir / f'{uuid_str}_result.json'
    with open(path, 'w') as f:
        json.dump(row, f, sort_keys=True, indent=4)


# Columns from the source row that should NOT be inherited — these are
# either score outputs (we recompute), timing (irrelevant), or already
# tied to the original alpha (config_hash).
_DROP_COLS = {'dice', 'sens', 'spec', 'pct_max_dice', 'min_pval',
              'vox_total', 'vox_effect', 'time_sec', 'config_hash',
              'uuid'}


def _inherit(row):
    """Return a dict of fields to carry over from the source result row."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in row.items() if k not in _DROP_COLS}


def _sweep_one(ana, effect, source_row, alphas, cft_pvals, out_folder,
               done_keys):
    """Sweep a single (ana, effect) pickle over alphas (and cft_pvals).

    Returns the number of rows emitted.
    """
    uuid = source_row['uuid']
    is_cet = type(ana).__name__ == 'AnalysisCET'
    cft_grid = cft_pvals if (is_cet and cft_pvals) else [None]
    inherited = _inherit(source_row)
    n_emitted = 0

    for cft in cft_grid:
        # the cft_pval that will appear in emitted rows for this slice;
        # for CET without an override, that's the analysis's stored
        # cft_pval (used as the dedup key too, so resume is consistent
        # with what landed on disk).
        row_cft = (float(ana.cft_pval) if (is_cet and cft is None) else cft)
        for alpha in alphas:
            key = (uuid, float(alpha), row_cft)
            if key in done_keys:
                continue
            if cft is None:
                ana.rethreshold(alpha)
            else:
                ana.rethreshold(alpha, cft_pval=cft)
            scores = score_ana(ana, effect)
            row = {
                **inherited,
                **scores,
                'source_uuid': uuid,
                'alpha_fwer': float(alpha),
                'cft_pval': (float(ana.cft_pval) if is_cet else None),
            }
            _write_row(out_folder, row)
            done_keys.add(key)
            n_emitted += 1
    return n_emitted


def run_sweep(config_label, alphas=DEFAULT_ALPHAS, cft_pvals=None,
              labels=None, verbose=True):
    """Run an alpha (+ optional CET cft_pval) sweep for one benchmark config.

    Args:
        config_label: matches ``CONFIG_BY_LABEL``.
        alphas: iterable of alpha_fwer values to apply.
        cft_pvals: iterable of cft_pval values, or ``None`` to leave the
            CET cluster-forming threshold unchanged.
        labels: filter of source labels to sweep; default is every label
            whose name starts with GLOW / VBA / CET.
        verbose: print progress.
    """
    cfg = CONFIG_BY_LABEL.get(config_label)
    if cfg is None:
        raise SystemExit(f'unknown config label: {config_label}')

    src_df, src_folder = _load_source_df(config_label)
    out_folder = _sweep_folder(config_label)
    out_folder.mkdir(parents=True, exist_ok=True)
    done_keys = _load_existing_sweep_keys(out_folder)

    if labels is None:
        labels = {lbl for lbl in src_df['label'].unique()
                  if any(lbl.startswith(p) for p in _DEFAULT_LABEL_PREFIXES)}
    else:
        labels = set(labels)

    rows_by_uuid = {r['uuid']: r for _, r in src_df.iterrows()
                    if r['label'] in labels}

    pickles = []
    for path in sorted((src_folder / OUT).glob('*_detail.p.gz')):
        uuid = path.name.split('_detail.p.gz')[0]
        if uuid in rows_by_uuid:
            pickles.append((path, rows_by_uuid[uuid]))

    if verbose:
        n_alpha = len(list(alphas))
        n_cft = len(list(cft_pvals)) if cft_pvals else 1
        print(f'sweep: {config_label} -> {out_folder}')
        print(f'  source rows matched: {len(pickles)} '
              f'(labels: {sorted(labels)})')
        print(f'  grid: {n_alpha} alphas x {n_cft} cft_pvals '
              f'= {n_alpha * n_cft} points/row')
        print(f'  already done: {len(done_keys)} keys')

    total_emitted = 0
    iterable = tqdm(pickles, disable=not verbose, desc='sweep')
    for path, row in iterable:
        with gzip.open(path, 'rb') as f:
            ana, effect = pickle.load(f)
        # discover_mask -> EffectEstimate.from_exp_mask reads exp.y,
        # which is dropped from slim pickles.  Rebuild via the recipe.
        ana.exp.rehydrate()
        total_emitted += _sweep_one(
            ana=ana, effect=effect, source_row=row,
            alphas=alphas, cft_pvals=cft_pvals,
            out_folder=out_folder, done_keys=done_keys)

    if verbose:
        print(f'  emitted {total_emitted} new rows')

    # consolidate sweep JSONs into the sweep folder's results.csv
    load_update_all(f'{config_label}{SWEEP_SUFFIX}', verbose=verbose)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('config_label', help='benchmark config to sweep')
    p.add_argument('--alphas', nargs='+', type=float, default=list(DEFAULT_ALPHAS),
                   help='alpha_fwer grid (default: %(default)s)')
    p.add_argument('--cft-pvals', nargs='+', type=float, default=None,
                   help='CET cft_pval grid (default: keep stored value)')
    p.add_argument('--labels', nargs='+', default=None,
                   help='source labels to sweep (default: all GLOW/VBA/CET)')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_sweep(args.config_label, alphas=args.alphas,
              cft_pvals=args.cft_pvals, labels=args.labels,
              verbose=not args.quiet)
