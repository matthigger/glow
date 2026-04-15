#!/usr/bin/env python
"""One-time migration: rewrite config_hash in results.csv from old global
hashes to per-label hashes.

For each row, the new hash = config._config_hash_for_label(row['label']).

This is correct for labels whose analysis config hasn't changed.  For CET
(where z_flag was added), the old data was computed without z_flag, so those
rows get the *current* CET per-label hash — which won't match future runs
with z_flag=True.  We instead compute the old CET hash (without z_flag) so
the data is correctly marked as stale.

Run:
    python scripts/migrate_config_hashes.py [--dry-run]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

import glow.benchmark
from glow.benchmark.paper_config import CONFIG_BY_LABEL


# CET's config changed: z_flag was absent (False), now True.
# Compute what the per-label hash WOULD be for old CET config.
def _old_cet_hash(config):
    """Per-label hash for CET with old config (no z_flag)."""
    d = config._base_hash_dict()
    if config.ana_kwargs_dict and 'CET' in config.ana_kwargs_dict:
        cls, kw = config.ana_kwargs_dict['CET']
        old_kw = {k: v for k, v in kw.items() if k != 'z_flag'}
        d['ana'] = {'CET': config._ana_entry(cls, old_kw)}
    sig = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


def migrate(dry_run=False):
    p = glow.benchmark.get_path_result()

    for name, config in CONFIG_BY_LABEL.items():
        csv_path = p / name / 'results.csv'
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if df.empty or 'config_hash' not in df.columns:
            continue

        old_hashes = set(df['config_hash'].unique())

        # build label -> new hash mapping
        expected = config._get_expected_labels()
        label_to_hash = {}
        for lab in expected:
            label_to_hash[lab] = config._config_hash_for_label(lab)

        # for CET, use old hash (without z_flag) so rows are marked stale
        if 'CET' in expected and config.ana_kwargs_dict and 'CET' in config.ana_kwargs_dict:
            cet_kw = config.ana_kwargs_dict['CET'][1]
            if 'z_flag' in cet_kw:
                label_to_hash['CET'] = _old_cet_hash(config)

        # rewrite: only touch rows whose label is in our mapping
        new_hashes = df['config_hash'].copy()
        for lab, h in label_to_hash.items():
            mask = df['label'] == lab
            new_hashes[mask] = h

        n_changed = (df['config_hash'] != new_hashes).sum()
        new_hash_set = set(new_hashes.unique())

        print(f'{name}:')
        print(f'  old hashes: {old_hashes}')
        print(f'  new hashes: {new_hash_set}')
        print(f'  {n_changed}/{len(df)} rows changed')

        if not dry_run and n_changed > 0:
            # backup
            backup = csv_path.with_suffix('.csv.bak')
            shutil.copy2(csv_path, backup)
            print(f'  backup: {backup}')

            df['config_hash'] = new_hashes
            df.to_csv(csv_path, index=False)
            print(f'  written: {csv_path}')
        elif dry_run:
            print(f'  (dry run, no changes written)')


def verify():
    """After migration, verify each row's hash matches current per-label hash."""
    p = glow.benchmark.get_path_result()
    all_ok = True

    for name, config in CONFIG_BY_LABEL.items():
        csv_path = p / name / 'results.csv'
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if df.empty or 'config_hash' not in df.columns:
            continue

        expected = config._get_expected_labels()
        label_to_hash = {lab: config._config_hash_for_label(lab) for lab in expected}

        n_valid = 0
        n_stale = 0
        stale_labels = set()

        for _, row in df.iterrows():
            lab = row['label']
            expected_hash = label_to_hash.get(lab)
            if expected_hash is None:
                continue
            if row['config_hash'] == expected_hash:
                n_valid += 1
            else:
                n_stale += 1
                stale_labels.add(lab)

        status = 'OK' if n_stale == 0 else f'STALE: {stale_labels}'
        print(f'{name}: {n_valid} valid, {n_stale} stale  [{status}]')
        if n_stale > 0:
            all_ok = False

    return all_ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show changes without writing')
    args = parser.parse_args()

    print('=== Migration ===')
    migrate(dry_run=args.dry_run)

    if not args.dry_run:
        print('\n=== Verification ===')
        ok = verify()
        if ok:
            print('\nAll rows valid.')
        else:
            print('\nSome rows are stale (need re-running).')
