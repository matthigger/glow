#!/usr/bin/env python
"""One-time migration for the segment-extenter rework.

Two changes are reconciled with on-disk results:

  1. ``Config.effect_extenter`` was added and is now part of ``base_recipe``,
     so every cache hash shifts. All existing segment results were produced
     with the implicit (and now explicit) default ``'minvar'``, so the new
     hash with ``effect_extenter='minvar'`` is what those rows should carry.

  2. The two segment configs are about to be renamed:
       segment_hcp → segment_minvar_hcp
       segment_wgn → segment_minvar_wgn
     so the cache folders are renamed to match.

Run once between the ``effect_extenter`` commit and the paper_config rename:
    python scripts/migrate_segment_extenter.py [--dry-run]
"""
import argparse
import hashlib
import json
import shutil

import pandas as pd

import glow.benchmark
from glow.benchmark.paper_config import CONFIG_BY_LABEL


# (old_folder, new_folder) — folders are renamed; config_label inside Config
# is not part of base_recipe so it does not affect the hash.
RENAMES = [
    ('segment_hcp', 'segment_minvar_hcp'),
    ('segment_wgn', 'segment_minvar_wgn'),
]


def _hash_recipe(recipe):
    sig = json.dumps(recipe, sort_keys=True, default=str)
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


def _old_and_new_hashes(config):
    """For every sub-label this runner emits, return {sub_label: (old, new)}.

    "old" reproduces the pre-effect_extenter recipe by stripping that key
    from the current recipe; "new" is the current recipe.
    """
    out = {}
    for sub_label in config.runner.labels:
        new_recipe = config.runner.recipe(config, sub_label)
        old_recipe = {k: v for k, v in new_recipe.items()
                      if k != 'effect_extenter'}
        out[sub_label] = (_hash_recipe(old_recipe), _hash_recipe(new_recipe))
    return out


def migrate(dry_run=False):
    base = glow.benchmark.get_path_result()

    for old_name, new_name in RENAMES:
        old_dir = base / old_name
        new_dir = base / new_name

        if not old_dir.exists():
            print(f'{old_name}: no cache folder, skipping')
            continue

        config = CONFIG_BY_LABEL.get(old_name) or CONFIG_BY_LABEL.get(new_name)
        if config is None:
            print(f'{old_name}: no matching config in CONFIG_BY_LABEL, '
                  f'skipping')
            continue

        # rewrite hashes in results.csv
        csv_path = old_dir / 'results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if 'config_hash' in df.columns and 'label' in df.columns:
                mapping = _old_and_new_hashes(config)
                new_col = df['config_hash'].copy()
                for sub_label, (old_h, new_h) in mapping.items():
                    sel = (df['label'] == sub_label) & \
                          (df['config_hash'] == old_h)
                    new_col[sel] = new_h
                n_changed = (df['config_hash'] != new_col).sum()
                print(f'{old_name}: {n_changed}/{len(df)} rows rehashed')
                for sub_label, (old_h, new_h) in mapping.items():
                    print(f'  {sub_label}: {old_h} → {new_h}')
                if not dry_run and n_changed > 0:
                    shutil.copy(csv_path, csv_path.with_suffix('.csv.bak'))
                    df['config_hash'] = new_col
                    df.to_csv(csv_path, index=False)
            else:
                print(f'{old_name}: results.csv missing expected columns, '
                      f'skipping rehash')
        else:
            print(f'{old_name}: no results.csv, skipping rehash')

        # rename folder
        if new_dir.exists():
            print(f'{new_name}: target folder already exists, leaving '
                  f'{old_name} in place')
            continue
        print(f'rename: {old_name}/ → {new_name}/')
        if not dry_run:
            old_dir.rename(new_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='print actions without modifying disk')
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
