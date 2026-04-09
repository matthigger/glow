#!/usr/bin/env python
"""One-time migration: rename 'f1' -> 'dice' and 'pct_max_f1' -> 'pct_max_dice'
in result JSON files.

Usage:
    python scripts/migrate_f1_to_dice.py <results_dir>

Edits files in-place. Skips files that already have 'dice' key.
"""

import json
import sys
from pathlib import Path

KEY_MAP = {
    'f1': 'dice',
    'pct_max_f1': 'pct_max_dice',
}


def migrate_file(path):
    with open(path) as f:
        d = json.load(f)

    if 'dice' in d:
        return False

    changed = False
    for old, new in KEY_MAP.items():
        if old in d:
            d[new] = d.pop(old)
            changed = True

    if changed:
        with open(path, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)

    return changed


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    root = Path(sys.argv[1])

    # migrate JSON result files
    files = list(root.rglob('*_result.json'))
    migrated = sum(1 for f in files if migrate_file(f))
    print(f'{migrated}/{len(files)} JSON files migrated in {root}')

    # remove cached results.csv (will be regenerated with new column names)
    for csv in root.rglob('results.csv'):
        csv.unlink()
        print(f'removed stale cache: {csv}')


if __name__ == '__main__':
    main()
