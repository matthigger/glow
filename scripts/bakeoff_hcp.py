"""Bake the GLOW arms off against the voxel arms on HCP, one seed at a time.

The head-to-head behind the arm choice: both GLOW arms (per-perm and split)
in both Ward modes, beside the voxel-wise arms they have to beat, on planted
HCP cells at a crop small enough that a whole seed finishes in minutes.

Every knob is a constant below -- edit and run:

    PYTHONPATH=<this worktree> ~/venv_glow/bin/python scripts/bakeoff_hcp.py

Feedback while it runs, at two grains:

  - OUT_JSONL takes one row per (seed, effect_llr, variant) as each fit
    lands, so nothing is lost if the run is interrupted and a rerun resumes
    rather than repeats (rows already present are skipped).
  - OUT_MD is a timestamped markdown summary over every row so far --
    Dice, regions declared, the false-region rate and seconds per fit --
    rewritten after every cell, so it moves every few minutes rather than
    once a seed.

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
from glow._extra.benchmark.score import score_prune
from glow.analysis import (AnalysisCET, AnalysisGLOW, AnalysisGLOWBase,
                           AnalysisGLOWSplit, AnalysisVBA)
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks
from glow.analysis.prune import prune_by_rule, prune_oracle
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

# ---- the pruning rules re-selected off each GLOW fit ------------------
# label -> prune_by_rule kwargs. Every one of these is a re-selection off a
# fit already paid for (milliseconds against ~20 s), so the whole rule axis
# is free once the arm has been fit. 'oracle' is handled apart: it takes the
# planted support, so it is the headroom line rather than a rule -- what a
# perfect selector could still win off this fit. dp-n1 / dp-n4 penalize dp
# through its geometric prior (one planted effect, so n=1 is the matched
# prior and n=4 a milder one).
RULE_LIST = (
    ('greedy', dict(rule='greedy')),
    ('dp', dict(rule='dp')),
    ('dp-n1', dict(rule='dp', exp_n_eff=1.0)),
    ('dp-n4', dict(rule='dp', exp_n_eff=4.0)),
    ('single_max', dict(rule='single_max')),
    ('oracle', None),
)

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


def wanted(name: str, ana) -> list:
    """The row variants one fit of this arm produces.

    A GLOW arm yields its own row plus one per pruning rule, all off the one
    fit, so the fit is owed whenever any of them is missing -- which is what
    backfills the rule rows for seeds scored before they existed.

    Args:
        name (str): the variant name.
        ana: the (unfitted) Analysis.

    Returns:
        list[str]: variant names this fit writes.
    """
    if not isinstance(ana, AnalysisGLOWBase):
        return [name]
    return [name] + [f'{name}+{label}' for label, _ in RULE_LIST]


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


def rule_rows(ana, exp, mask_list) -> dict:
    """Re-select this fit's significant regions under every pruning rule.

    The rule axis costs one fit, not one fit per rule: a rule reads the
    fitted tree, the raw per-region LLR and the FWER-significant set, so
    every rule in RULE_LIST is scored off the same fit -- the same isolation
    the benchmark's prune cache buys with glow_fit_for_prune, here on
    whichever arm was fit rather than the split arm alone.

    The greedy row is a check as much as a result: the arms fit at
    prune_rule='greedy', so it must reproduce the arm's own Dice.

    Args:
        ana: a fitted GLOW arm (needs children, llr, fwer).
        exp (Experiment): the cell it was fit on.
        mask_list (list): the planted (X, Y, Z) bool supports.

    A rule row carries no false_region flag: score_prune returns the
    confusion counts of the union, not each region's overlap, so the flag
    would have to be invented. The report's table skips what a row lacks.

    Returns:
        dict: rule label -> {dice, n_pred, sec}.
    """
    llr = np.nan_to_num(ana.llr.astype(float), nan=0.0, posinf=0.0,
                        neginf=0.0)
    sig_reg_list = np.flatnonzero(ana.fwer.reg_sig).tolist()
    mask_active = exp.mask_idx > -1
    mask_target = np.zeros(exp.mask_idx.shape, dtype=bool)
    for mask in mask_list:
        mask_target |= mask

    out = {}
    for label, kwargs in RULE_LIST:
        tic = time.perf_counter()
        if kwargs is None:
            reg_out_list, _ = prune_oracle(sig_reg_list=sig_reg_list,
                                           children=ana.children,
                                           mask_target=mask_target,
                                           mask_idx=exp.mask_idx)
        else:
            reg_out_list, _ = prune_by_rule(
                sig_reg_list=sig_reg_list, children=ana.children, stat=llr,
                **kwargs)
        sec = time.perf_counter() - tic
        c = score_prune(reg_out_list, children=ana.children,
                        mask_idx=exp.mask_idx, mask_target_list=mask_list,
                        mask_active=mask_active)
        denom = 2 * c['tp'] + c['fp'] + c['fn']
        out[label] = dict(dice=0.0 if denom == 0 else 2 * c['tp'] / denom,
                          n_pred=c['n_selected'], sec=sec)
    return out


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
        if field in row:
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


def _rule_section(rows) -> str:
    """Render Dice and region count per pruning rule, one pair per GLOW arm.

    Every rule here read the same fit as its arm's own row, so a column
    difference is the selection rule and nothing else. oracle is the
    headroom line, not a method (see RULE_LIST).

    Args:
        rows (list[dict]): raw rows.

    Returns:
        str: the markdown, empty while no rule row exists yet.
    """
    label_list = [label for label, _ in RULE_LIST]
    arm_list = [name for name, ana in variant_dict().items()
                if isinstance(ana, AnalysisGLOWBase)]
    have = {row['variant'] for row in rows}
    out = []
    for arm in arm_list:
        col_list = [f'{arm}+{label}' for label in label_list]
        if not any(col in have for col in col_list):
            continue
        sub = [row for row in rows if row['variant'] in col_list]
        renamed = [dict(row, variant=row['variant'].split('+', 1)[1])
                   for row in sub]
        out.append(f'\n## pruning rules on {arm} (one shared fit per cell)'
                   f'\n\nDice\n\n{_table(renamed, "dice", label_list)}'
                   f'\n\nregions declared\n\n'
                   f'{_table(renamed, "n_pred", label_list, fmt="{:.1f}")}')
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

Cell: HCP, crop {CROP_N_VOX} voxels, b={B}, {NUM_IMG} subjects, planted \
support {EFFECT_N_VOX_FRAC:.0%} of the crop. Test: n_perm_fwer={N_PERM_FWER}, \
alpha={ALPHA_FWER}; the per-perm arm draws n_perm_inner={N_PERM_INNER}. \
Every arm sees the identical cell, so the columns are paired seed by seed.

## Dice (mean over seeds)

{_table(rows, 'dice', name_list)}

## regions declared (mean over seeds)

{_table(rows, 'n_pred', name_list, fmt='{:.1f}')}

## false-region rate: a declared region holding no effect voxel

{_table(rows, 'false_region', name_list)}

## seconds per fit (mean over seeds)

{_table(rows, 'sec', name_list, fmt='{:.1f}')}
{_rule_section(rows)}"""
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
                    if any((seed, llr, want) not in skip
                           for want in wanted(name, ana))}
            if not todo:
                continue
            exp, mask_list = build(seed, llr)
            with OUT_JSONL.open('a') as f:
                for name, ana in todo.items():
                    fit_tic = time.perf_counter()
                    ana.fit(exp, **FIT)
                    sec = time.perf_counter() - fit_tic
                    out = {name: dict(sec=sec,
                                      **score_fit(ana, exp, mask_list))}
                    if isinstance(ana, AnalysisGLOWBase):
                        for label, val in rule_rows(ana, exp,
                                                    mask_list).items():
                            out[f'{name}+{label}'] = val
                    for variant, val in out.items():
                        if (seed, llr, variant) in skip:
                            continue
                        f.write(json.dumps(dict(seed=seed, llr=llr,
                                                variant=variant,
                                                **val)) + '\n')
                        f.flush()
                    print(f'seed {seed} llr {llr:.6f} {name:14s} '
                          f'{sec:7.1f}s dice {out[name]["dice"]:.3f}',
                          flush=True)
            write_md(read_rows(OUT_JSONL), time.perf_counter() - tic)
        print(f'--- seed {seed} done, {OUT_MD} updated', flush=True)


if __name__ == '__main__':
    main()
