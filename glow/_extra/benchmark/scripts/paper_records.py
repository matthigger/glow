"""Export the provenance records behind the manuscript, and only those.

The published deposit is a record tree from which plot redraws every figure
and table the paper shows, with no compute and no imaging data. It holds each
paper cache's leaves (results.config_leaf_keys) plus every record they
descend from, so each row still reaches its cell's source, seed and plant.
PAPER_CACHES names the caches; a slice narrower than a whole cache (sweep_llr
is drawn at b = 1 only) is a filter on the leaf's cell record.

    python -m glow._extra.benchmark.scripts.paper_records --out <dir>

writes <dir>/records/<hash>.json, copied byte for byte from the store, and a
manifest.json counting records per cache and function. Unpack <dir> as the
glow data directory (see rerun.md) and plot reads it as its own store.
"""

import argparse
import collections
import json
import pathlib
import shutil

from glow._extra.benchmark import results
from glow._extra.benchmark.store import RECORDER

# cache name -> the b its leaves are kept at (None keeps every leaf), one
# entry per results cache a manuscript figure or table draws
PAPER_CACHES = {
    'segment': None,
    'prune': None,
    'null': None,
    'sweep_llr': 1,
    'vba_tune': None,
    'vba_stat': None,
    'runtime_num_vox': None,
    'runtime_1perm_num_vox': None,
    'runtime_1perm_n_perm_inner': None,
    'runtime_1perm_b': None,
    'runtime_1perm_nimg': None,
}


def kwargs_data_b(kwargs_data: dict) -> int:
    """Return the feature count a data cell declares.

    WGN declares b; HCP declares its features, and b is how many.
    """
    if kwargs_data.get('source') == 'hcp':
        return len(kwargs_data['hcp_feats'])
    return kwargs_data['b']


def select(cache_dict: dict) -> dict:
    """Pick each cache's leaves and the closure of their ancestors.

    Args:
        cache_dict (dict): cache name -> b to keep, or None for every leaf

    Returns:
        key_cache (dict): record key -> sorted list of the paper caches it
            belongs to, for every record exported
    """
    RECORDER.load()
    records = RECORDER.records
    key_of_uid = {rec['uid']: key for key, rec in records.items()
                  if rec.get('uid')}

    def parent_keys(key):
        return [key_of_uid[uid] for uid in records[key].get('parents') or ()
                if uid in key_of_uid and key_of_uid[uid] != key]

    key_cache = collections.defaultdict(set)
    for name, b in cache_dict.items():
        leaf_list = results.config_leaf_keys(name)
        if b is not None:
            leaf_list = [
                key for key in leaf_list
                if all(kwargs_data_b(records[p]['inputs']['kwargs_data']) == b
                       for p in parent_keys(key)
                       if records[p]['function'] == 'get_exp_effect')]
        todo = list(leaf_list)
        while todo:
            key = todo.pop()
            if name in key_cache[key]:
                continue
            key_cache[key].add(name)
            todo.extend(parent_keys(key))
    return {key: sorted(names) for key, names in key_cache.items()}


def export(key_cache: dict, out: pathlib.Path) -> dict:
    """Copy the selected records into out/records and write a manifest.

    Args:
        key_cache (dict): select's record key -> cache names
        out (pathlib.Path): destination directory, created if missing

    Returns:
        manifest (dict): {n_record, per_cache: {name: {function: count}}}
    """
    rec_dir = out / 'records'
    rec_dir.mkdir(parents=True, exist_ok=True)
    per_cache = collections.defaultdict(collections.Counter)
    for key, names in key_cache.items():
        shutil.copy2(RECORDER.folder / f'{key}.json', rec_dir / f'{key}.json')
        for name in names:
            per_cache[name][RECORDER.records[key]['function']] += 1
    manifest = {'n_record': len(key_cache),
                'per_cache': {name: dict(counts) for name, counts
                              in sorted(per_cache.items())}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def main(argv=None) -> None:
    """Select, copy and summarise the paper's records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=pathlib.Path, required=True,
                        help='destination; records land in <out>/records')
    args = parser.parse_args(argv)
    manifest = export(select(PAPER_CACHES), args.out)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
