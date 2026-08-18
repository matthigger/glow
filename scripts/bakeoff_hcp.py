"""Bake the GLOW arms off against the voxel arms on HCP, one seed at a time.

The head-to-head behind the arm choice: both GLOW arms (per-perm and split)
in both Ward modes, beside the voxel-wise arms they have to beat, on planted
HCP cells at a crop small enough that a whole seed finishes in minutes.

Every knob is a constant below -- edit and run:

    PYTHONPATH=<this worktree> ~/venv_glow/bin/python scripts/bakeoff_hcp.py

Feedback while it runs. A seed's cells are fit in one pass, then two files
are rewritten before the next seed starts:

  - OUT_JSONL, one row per (seed, effect_llr, variant): the raw scores, so
    nothing is lost if the run is interrupted and a rerun resumes rather
    than repeats (rows already present are skipped).
  - OUT_MD, a timestamped markdown summary over every row so far --
    Dice, regions declared, the false-region rate and seconds per fit.

Touches no cache and writes no record: the cells are built through the
benchmark's builders with their cache and recorder peeled off (see
call_uncached), so an ad-hoc run here cannot rewrite a benchmark cell's
provenance.
"""
import inspect
import json
import pathlib
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

from glow._extra.benchmark import hcp
from glow._extra.benchmark.config import EFFECT_LLR_GRID
from glow._extra.benchmark.data import (data_factory_hcp, data_recipe,
                                        effect_factory_single)
from glow._extra.benchmark.score import score_effects
from glow.analysis import (AnalysisCET, AnalysisGLOW, AnalysisGLOWSplit,
                           AnalysisVBA)
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks
from glow.effect import ExtenterMinVar, ExtenterSphere

# ---- the cell ---------------------------------------------------------
CROP_N_VOX = 5_000
B = 1
NUM_IMG = 100
EFFECT_N_VOX_FRAC = 0.1
# a slice of the paper's own effect_llr grid (taken off it, not retyped, so
# these are the same cells a real sweep would plant), cut to the band where
# the arms separate: below it every arm is at the floor, above it every arm
# saturates, and neither end tells us which arm to pick.
LLR_LIST = tuple(float(v) for v in EFFECT_LLR_GRID[4:9])
SEED_LIST = tuple(range(50))

# ---- the analyses -----------------------------------------------------
N_PERM_FWER = 500
N_PERM_INNER = 250
ALPHA_FWER = 0.05
FIT = dict(n_jobs=8, gpu='auto')

# ---- output -----------------------------------------------------------
OUT_DIR = pathlib.Path.home() / 'Dropbox' / 'glow' / 'results'
OUT_MD = OUT_DIR / 'bakeoff_hcp_5k.md'
OUT_JSONL = OUT_DIR / 'bakeoff_hcp_5k.jsonl'


def variant_dict() -> dict:
    """Build the arms to compare, all at one outer permutation count.

    Returns:
        dict: variant name -> an unfitted Analysis. Insertion order is the
            column order of the report.
    """
    base = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER)
    return {
        'split-Focus': AnalysisGLOWSplit(cluster_mode=ClusterMode.FOCUS,
                                         **base),
        'split-GLM': AnalysisGLOWSplit(cluster_mode=ClusterMode.GLM_ERROR,
                                       **base),
        'perperm-Focus': AnalysisGLOW(n_perm_inner=N_PERM_INNER,
                                      cluster_mode=ClusterMode.FOCUS, **base),
        'perperm-GLM': AnalysisGLOW(n_perm_inner=N_PERM_INNER,
                                    cluster_mode=ClusterMode.GLM_ERROR,
                                    **base),
        'VBA': AnalysisVBA(get_stat=get_hotel_tr, z_flag=False,
                           tfce_flag=False, **base),
        'VBA-TFCE': AnalysisVBA(get_stat=get_wilks, z_flag=True,
                                tfce_flag=True, **base),
        'CET': AnalysisCET(get_stat=get_hotel_tr, z_flag=False, **base),
    }


def call_uncached(fnc, *args, **kwargs):
    """Call a benchmark builder with its cache and recorder peeled off.

    inspect.unwrap walks past @MEMORY.cache and @RECORDER to the raw
    function. Worth the detour: an ad-hoc build landing on a benchmark
    cell's key rewrites that record with a fresh exp hash, dropping every
    finished leaf that consumed the old one out of config_results_df.

    Args:
        fnc: a decorated builder from glow._extra.benchmark.data.

    Returns:
        whatever the raw builder returns.
    """
    return inspect.unwrap(fnc)(*args, **kwargs)


def hcp_feats(seed: int) -> tuple:
    """The random b-subset the benchmark's data grid draws at this seed.

    Args:
        seed (int): the cell's seed.

    Returns:
        tuple[str]: sorted feature names, len B.
    """
    idx = np.random.default_rng(seed).choice(len(hcp.HCP_FEATS), size=B,
                                            replace=False)
    return tuple(sorted(hcp.HCP_FEATS[i] for i in idx))


def build(seed: int, llr: float) -> tuple:
    """Build one planted HCP cell.

    Args:
        seed (int): drives the feature subset, the crop and the design.
        llr (float): per-voxel effect_llr to plant.

    Returns:
        exp (Experiment): the cell, effect planted.
        mask_list (list): the one realized (X, Y, Z) bool support.
    """
    extenter = ExtenterSphere(n_vox=CROP_N_VOX, connected=True,
                              contiguous=True, seed=seed)
    kwargs_data = dict(source='hcp', hcp_feats=hcp_feats(seed), seed=seed,
                       extenter=extenter)
    exp = call_uncached(data_factory_hcp, hcp_feats=kwargs_data['hcp_feats'],
                        seed=seed, extenter=extenter)
    exp, mask_list = call_uncached(
        effect_factory_single, exp,
        parent_uid=data_recipe(kwargs_data).uid, effect_llr=float(llr),
        extenter_cls=ExtenterMinVar, n_vox_frac=EFFECT_N_VOX_FRAC,
        seed_from_exp=True)
    return exp, mask_list


def score_fit(ana, exp, mask_list) -> dict:
    """Score one fitted analysis against the planted support.

    Args:
        ana: a fitted Analysis.
        exp (Experiment): the cell it was fit on.
        mask_list (list): the planted (X, Y, Z) bool supports.

    Returns:
        {dice, n_pred, false_region}: Dice of the union of discoveries, the
            number of regions declared, and whether any one of them holds
            no effect voxel at all (the weak-FWER symptom).
    """
    out = score_effects(ana, mask_list, exp.mask_idx > -1)
    c = out['target']
    denom = 2 * c['tp'] + c['fp'] + c['fn']
    pred = out['pred']
    return dict(dice=0.0 if denom == 0 else 2 * c['tp'] / denom,
                n_pred=len(pred),
                false_region=bool(pred) and any(p['target'] == 0
                                                for p in pred))


def done_keys(path: pathlib.Path) -> set:
    """Read the (seed, llr, variant) triples a previous run already scored.

    Args:
        path (pathlib.Path): the JSONL of raw rows; may not exist.

    Returns:
        set: (seed, llr, variant) tuples to skip.
    """
    if not path.exists():
        return set()
    out = set()
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            out.add((row['seed'], row['llr'], row['variant']))
    return out


def read_rows(path: pathlib.Path) -> list:
    """Read every raw row written so far.

    Args:
        path (pathlib.Path): the JSONL of raw rows; may not exist.

    Returns:
        list[dict]: the rows, in write order.
    """
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


def _table(rows, field, name_list, fmt='{:.3f}') -> str:
    """Render one metric as a markdown llr-by-variant table of seed means.

    Args:
        rows (list[dict]): raw rows.
        field (str): the row key to average.
        name_list (list[str]): variant column order.
        fmt (str): format for a cell value.

    Returns:
        str: the markdown table.
    """
    acc = defaultdict(list)
    for row in rows:
        acc[(row['llr'], row['variant'])].append(row[field])
    llr_list = sorted({row['llr'] for row in rows})

    head = '| effect_llr | n | ' + ' | '.join(name_list) + ' |'
    rule = '|---' * (len(name_list) + 2) + '|'
    out = [head, rule]
    for llr in llr_list:
        n_seed = max((len(acc[(llr, name)]) for name in name_list),
                     default=0)
        cell_list = []
        for name in name_list:
            val = acc[(llr, name)]
            cell_list.append(fmt.format(float(np.mean(val))) if val else '-')
        out.append(f'| {llr:.6f} | {n_seed} | ' + ' | '.join(cell_list)
                   + ' |')
    return '\n'.join(out)


def write_md(rows, elapsed_sec: float) -> None:
    """Rewrite the markdown summary over every row so far.

    Args:
        rows (list[dict]): raw rows.
        elapsed_sec (float): wall time of this process so far.
    """
    name_list = list(variant_dict())
    seed_done = sorted({row['seed'] for row in rows})
    n_cell = len({(row['seed'], row['llr']) for row in rows})
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    text = f"""# GLOW arm bake-off -- HCP, {CROP_N_VOX} voxels

updated **{stamp}** | seeds finished **{len(seed_done)}** \
({', '.join(str(s) for s in seed_done) if seed_done else 'none'}) \
| cells {n_cell} | this process {elapsed_sec / 60:.1f} min

Cell: HCP, crop {CROP_N_VOX} voxels, b={B}, {NUM_IMG} subjects, planted
support {EFFECT_N_VOX_FRAC:.0%} of the crop. Test: n_perm_fwer=
{N_PERM_FWER}, alpha={ALPHA_FWER}; the per-perm arm draws
n_perm_inner={N_PERM_INNER}. Every arm sees the identical cell, so the
columns are paired seed by seed.

## Dice (mean over seeds)

{_table(rows, 'dice', name_list)}

## regions declared (mean over seeds)

{_table(rows, 'n_pred', name_list, fmt='{:.1f}')}

## false-region rate: a declared region holding no effect voxel

{_table(rows, 'false_region', name_list)}

## seconds per fit (mean over seeds)

{_table(rows, 'sec', name_list, fmt='{:.1f}')}
"""
    OUT_MD.write_text(text)


def main():
    """Fit every variant on every (seed, llr) cell, reporting each seed."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats = hcp_feats(SEED_LIST[0])
    if not hcp.is_bundle_present(feats):
        raise SystemExit(f'HCP bundle absent for {feats}')

    tic = time.perf_counter()
    skip = done_keys(OUT_JSONL)
    if skip:
        print(f'resuming: {len(skip)} rows already on disk')

    for seed in SEED_LIST:
        for llr in LLR_LIST:
            todo = {name: ana for name, ana in variant_dict().items()
                    if (seed, llr, name) not in skip}
            if not todo:
                continue
            exp, mask_list = build(seed, llr)
            with OUT_JSONL.open('a') as f:
                for name, ana in todo.items():
                    fit_tic = time.perf_counter()
                    ana.fit(exp, **FIT)
                    sec = time.perf_counter() - fit_tic
                    row = dict(seed=seed, llr=llr, variant=name, sec=sec,
                               **score_fit(ana, exp, mask_list))
                    f.write(json.dumps(row) + '\n')
                    f.flush()
                    print(f'seed {seed} llr {llr:.6f} {name:14s} '
                          f'{sec:7.1f}s dice {row["dice"]:.3f}', flush=True)
        write_md(read_rows(OUT_JSONL), time.perf_counter() - tic)
        print(f'--- seed {seed} done, {OUT_MD} updated', flush=True)


if __name__ == '__main__':
    main()
