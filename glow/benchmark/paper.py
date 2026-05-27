"""CLI entry point for paper benchmarks.

Resolves config labels from ``paper_config.CACHE_BY_LABEL`` and runs
each one through ``driver_local``.  One trial = one (data source,
extenter, effect_llr, seed) tuple; for every trial, every analysis in
the cache's ``ana_kwargs_dict`` is fit and scored, emitting one row
per analysis into the cache's csv.

Usage:
    python -m glow.benchmark.paper                       # everything
    python -m glow.benchmark.paper null_*                # glob
    python -m glow.benchmark.paper -j 4 vba_hcp_famd     # parallel
"""
import argparse
import time
import traceback
from fnmatch import fnmatch

import numpy as np
import pandas as pd

import glow
from glow.effect import EffectSynthetic

from .driver import driver_local
from .paper_config import CACHE_BY_LABEL


def run_trial(*, ds, extenter, effect_llr, seed, ana_kwargs_dict):
    """Fit every analysis in ``ana_kwargs_dict`` on one synthetic trial.

    Builds ``ds.exp`` (memoised on the DataSource), plants a synthetic
    effect at the requested LLR with the given extenter / seed, fits
    each analysis, and scores the discovered mask against the planted
    one.  Returns a DataFrame with one row per analysis label.
    """
    exp = ds.exp
    synth = EffectSynthetic(extenter=extenter, effect_llr=effect_llr,
                            seed=seed)
    exp_eff = synth.fit(exp)

    mask_active = exp.mask_idx > -1
    mask_target = synth.mask_

    rows = []
    for label, (Ana, kw) in ana_kwargs_dict.items():
        row = {
            'analysis_label': label,
            'analysis_cls': Ana.__name__,
            'vox_total': int(exp.y.shape[2]),
            'vox_effect': int(mask_target.sum()),
        }
        t0 = time.time()
        try:
            ana = Ana(exp=exp_eff, **kw).fit()
        except Exception:
            row['time_sec'] = time.time() - t0
            row['error'] = traceback.format_exc()
            rows.append(row)
            continue
        row['time_sec'] = time.time() - t0

        mask_pred = np.zeros(exp.mask_idx.shape, dtype=bool)
        for eff in (ana.effect_list or ()):
            mask_pred |= eff.mask

        dice, sens, spec = glow.mask.get_score(
            mask_pred=mask_pred, mask_target=mask_target,
            mask_active=mask_active)
        row.update(dice=dice, sens=sens, spec=spec)
        row['min_pval'] = (float(np.nanmin(ana.pval))
                           if getattr(ana, 'pval', None) is not None
                           else np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def _make_run_fnc(ana_kwargs_dict):
    """Bind ``ana_kwargs_dict`` so the cache only iterates trial-state."""
    def run_fnc(*, ds, extenter, effect_llr, seed):
        return run_trial(ds=ds, extenter=extenter,
                         effect_llr=effect_llr, seed=seed,
                         ana_kwargs_dict=ana_kwargs_dict)
    return run_fnc


def resolve_labels(patterns):
    """Expand a list of literal labels / fnmatch patterns into entries.

    Empty input means "everything".  Unknown literal labels raise; an
    fnmatch pattern that matches nothing also raises (so typos surface).
    """
    if not patterns:
        return list(CACHE_BY_LABEL.items())

    out = []
    seen = set()
    for pattern in patterns:
        if pattern in CACHE_BY_LABEL:
            matches = [pattern]
        else:
            matches = [k for k in CACHE_BY_LABEL if fnmatch(k, pattern)]
            if not matches:
                raise ValueError(f'no cache labels match: {pattern}')
        for k in matches:
            if k not in seen:
                seen.add(k)
                out.append((k, CACHE_BY_LABEL[k]))
    return out


def run(labels=None, n_jobs=1, verbose=True):
    """Run every selected cache through ``driver_local``."""
    entries = resolve_labels(labels or [])
    for label, (cache, ana_kwargs_dict) in entries:
        if verbose:
            print(f'\n=== {label} ({cache.folder}) ===')
        driver_local(cache, _make_run_fnc(ana_kwargs_dict),
                     n_jobs=n_jobs, verbose=verbose)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run paper benchmarks.')
    parser.add_argument('labels', nargs='*',
                        help='cache labels or fnmatch patterns (default: all)')
    parser.add_argument('-j', '--n-jobs', type=int, default=1,
                        help='parallel worker count (default: 1)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='suppress per-trial print lines')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    run(labels=args.labels, n_jobs=args.n_jobs, verbose=not args.quiet)
