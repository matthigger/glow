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

        expected = set(config.runner.labels)
        label_to_hash = {lab: config.runner.hash(config, lab)
                         for lab in expected}

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

        expected = set(config.runner.labels)
        label_to_hash = {lab: config.runner.hash(config, lab) for lab in expected}

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
