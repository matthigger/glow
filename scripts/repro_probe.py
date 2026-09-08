"""Test whether a benchmark cell reproduces bit for bit, and where it stops.

The catalogue's cells are declared, seeded and portable by identity (see
glow._extra.benchmark.recipe), which makes a cell's NAME the same on any
machine. This harness asks the other half of the question -- whether the
BYTES are -- by perturbing exactly the things a second machine changes and
diffing a cell end to end. Findings and the fixes they argue for are in
docs/notes/reproducibility_audit.md.

Four probes, each isolating one candidate cause:

  blas     rebuild the cell under a forced OpenBLAS micro-kernel. Emulates
           a different host CPU: numpy ships DYNAMIC_ARCH, so the gemm
           kernel is chosen at import from the CPU it finds.
  backend  fit one exp on every device / dtype / chunk the catalogue may
           pick, plus two n_jobs. Emulates a runner with or without a card.
  numba    cluster under a forced compile target. The njit tree carries no
           fastmath, so this is expected to be inert -- it is the control.
  jitter   nudge every entry of y by one ULP and refit. Emulates the
           round-off a different BLAS kernel leaves behind, without
           needing a second machine to produce it.

Nothing here writes: the builders are called unwrapped, so neither the
joblib cache nor the records dir is touched (a plain call would rewrite a
record under a new exp hash and orphan finished leaves).

    python scripts/repro_probe.py blas
    python scripts/repro_probe.py jitter --n-seed 5
"""
import argparse
import copy
import hashlib
import inspect
import math
import os
import pickle
import subprocess
import sys

import numpy as np

from glow._extra.benchmark.data import (DATA_FACTORY, EFFECT_FACTORY,
                                        data_recipe)
from glow._extra.benchmark.score import score_effects
from glow.analysis import AnalysisGLOW
from glow.analysis._fit_gpu import GpuConfig
from glow.analysis.cluster import cluster, ClusterMode
from glow.effect import ExtenterMinVar, ExtenterSphere
from glow.experiment.exper import ExperimentScaled

# The fields a fit is diffed on, shallowest cause first: a divergence in
# children is a different segmentation, in llr a different statistic, in
# pval a different decision.
FIELDS = ('children', 'size', 'llr', 'mu', 'std', 'max_stat', 'stat_obs',
          'pval', 'reg_sig')

# Kernels OpenBLAS DYNAMIC_ARCH offers that differ in their gemm blocking.
# Haswell is what this class of host selects, so it is the reference.
CORETYPE_LIST = ('Haswell', 'Nehalem', 'Prescott')

# Small enough to run a probe in seconds, large enough that the cell goes
# through every stage a catalogue cell does (crop, min-var plant, Ward,
# per-perm draws). Not a catalogue size: a probe measures the mechanism.
CROP_N_VOX = 4_000


def build_cell(seed: int, effect_llr: float, b: int = 2, num_img: int = 100):
    """Build one WGN cell, cache and records untouched.

    Mirrors what grid.get_kwargs_data_list / get_kwargs_effect_list declare
    for a WGN cell, at CROP_N_VOX rather than the catalogue's crop.

    Args:
        seed (int): the cell's realization seed.
        effect_llr (float): per-voxel effect strength.
        b (int): imaging features.
        num_img (int): images.

    Returns:
        exp_clean (Experiment): the effect-free cell.
        exp (Experiment): the planted cell.
        mask_target_list (list): the realized support, as a one-element list.
    """
    side = math.ceil(CROP_N_VOX ** (1 / 3))
    kwargs_data = dict(
        source='wgn', shape=(side,) * 3, b=b, num_img=num_img, seed=seed,
        extenter=ExtenterSphere(n_vox=CROP_N_VOX, connected=True,
                                contiguous=True, seed=seed))
    kwargs_build = {k: v for k, v in kwargs_data.items() if k != 'source'}
    exp_clean = inspect.unwrap(DATA_FACTORY['wgn'])(**kwargs_build)
    exp, mask_target_list = inspect.unwrap(EFFECT_FACTORY['single'])(
        exp_clean, parent_uid=data_recipe(kwargs_data).uid,
        effect_llr=effect_llr, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
        seed_from_exp=True)
    return exp_clean, exp, mask_target_list


def fit_capture(exp, mask_target_list, *, n_perm_fwer: int, n_perm_inner: int,
                **fit_params) -> dict:
    """Fit GLOW on exp and capture what a diff needs, arrays included.

    Args:
        exp (Experiment): the cell to fit.
        mask_target_list (list): the planted supports, for the score.
        n_perm_fwer (int): outer perms.
        n_perm_inner (int): inner draws per outer perm.
        **fit_params: forwarded to AnalysisGLOW.fit (n_jobs, gpu).

    Returns:
        capture (dict): {children, size, llr, mu, std, max_stat, stat_obs,
            pval, reg_sig, pred, score} -- the FIELDS arrays plus the
            discovered supports as voxel-index lists and the score dict.
    """
    ana = AnalysisGLOW(n_perm_fwer=n_perm_fwer, n_perm_inner=n_perm_inner,
                       cluster_mode=ClusterMode.FOCUS, prune_rule='greedy')
    ana.fit(exp, **fit_params)
    out = {name: np.asarray(getattr(ana, name, None)
                            if hasattr(ana, name)
                            else getattr(ana.fwer, name))
           for name in FIELDS}
    out['pred'] = [np.flatnonzero(np.asarray(eff.mask).ravel()).tolist()
                   for eff in ana.effect_list]
    out['score'] = score_effects(ana, mask_target_list,
                                 mask_active=exp.mask_idx > -1)
    return out


def _rel_diff(a, b) -> float:
    """Return the max absolute difference relative to a's scale (NaN=0).

    Casts through float, so a boolean mask reports its disagreement rate's
    scale rather than raising on the subtraction.
    """
    fa = np.nan_to_num(np.asarray(a, dtype=float))
    fb = np.nan_to_num(np.asarray(b, dtype=float))
    scale = np.abs(fa).max() or 1.0
    return float(np.abs(fa - fb).max() / scale)


def report_diff(label: str, base: dict, other: dict) -> None:
    """Print one field-by-field diff of two captures.

    Args:
        label (str): what the comparison is called in the output.
        base (dict): the reference capture (fit_capture output).
        other (dict): the capture to diff against it.
    """
    print(f'--- {label} ---')
    for name in FIELDS:
        a, b = base[name], other[name]
        if a.shape != b.shape:
            print(f'  {name:9s} SHAPE {a.shape} vs {b.shape}')
            continue
        equal = np.array_equal(a, b, equal_nan=
                               np.issubdtype(a.dtype, np.floating))
        if np.issubdtype(a.dtype, np.floating):
            detail = f'rel={_rel_diff(a, b):.3e}'
        else:
            detail = f'n_differ={int((a != b).sum())}'
        print(f'  {name:9s} bitwise={str(equal):5s} {detail}')
    print(f'  supports identical: {base["pred"] == other["pred"]}')
    print(f'  score identical:    {base["score"] == other["score"]}')


def probe_backend(args) -> None:
    """Diff one exp fit on every device / dtype / chunk / n_jobs available."""
    _, exp, masks = build_cell(args.seed, args.effect_llr)
    kwargs = dict(n_perm_fwer=args.n_perm_fwer,
                  n_perm_inner=args.n_perm_inner)
    cases = {'cpu_fp64_j1': dict(n_jobs=1),
             'cpu_fp64_j4': dict(n_jobs=4)}
    from glow.analysis import draws_gpu
    if draws_gpu.is_available():
        cases['gpu_fp64_chunk4'] = dict(n_jobs=1,
                                        gpu=GpuConfig(perm_chunk=4))
        cases['gpu_fp64_chunk16'] = dict(n_jobs=1,
                                         gpu=GpuConfig(perm_chunk=16))
        cases['gpu_fp32_chunk4'] = dict(
            n_jobs=1, gpu=GpuConfig(perm_chunk=4, acc_dtype=np.float32))
    else:
        print('no device visible: the fp32 / device rows are the ones a '
              'machine WITH a card would add')
    capture = {name: fit_capture(exp, masks, **kwargs, **fit_params)
               for name, fit_params in cases.items()}
    base = capture['cpu_fp64_j1']
    for name in list(cases)[1:]:
        report_diff(f'cpu_fp64_j1 vs {name}', base, capture[name])


def probe_numba(args) -> None:
    """Hash the Ward tree in every mode, under this compile target.

    Run it twice with NUMBA_CPU_NAME set differently (the runner below does)
    and compare the hashes; equal hashes mean the njit tree is portable.
    """
    _, exp, _ = build_cell(args.seed, args.effect_llr)
    exp_scaled = ExperimentScaled.from_exp(exp)
    digest = {}
    for mode in ClusterMode:
        children = np.asarray(cluster(exp_scaled, mode=mode))
        digest[mode.name] = hashlib.md5(children.tobytes()).hexdigest()[:16]
    print('NUMBA_CPU_NAME=%-16s %s'
          % (os.environ.get('NUMBA_CPU_NAME', 'host'),
             ' '.join(f'{k}={v}' for k, v in digest.items())))


def probe_jitter(args) -> None:
    """Refit each seed's cell with every entry of y nudged one ULP.

    The stand-in for a second machine: the nudge is the magnitude a
    different BLAS micro-kernel leaves in a float32 y, so what survives it
    is what would survive the move.
    """
    n_changed = 0
    for seed in range(args.n_seed):
        _, exp, masks = build_cell(seed, args.effect_llr)
        rng = np.random.default_rng(1000 + seed)
        up = rng.random(exp.y.shape) < 0.5
        y_jitter = np.where(up, np.nextafter(exp.y, np.float32(np.inf)),
                            np.nextafter(exp.y, np.float32(-np.inf)))
        exp_jitter = copy.deepcopy(exp)
        exp_jitter.y = np.asfortranarray(y_jitter.astype(exp.y.dtype))

        kwargs = dict(n_perm_fwer=args.n_perm_fwer,
                      n_perm_inner=args.n_perm_inner)
        base = fit_capture(exp, masks, **kwargs, n_jobs=args.n_jobs)
        other = fit_capture(exp_jitter, masks, **kwargs, n_jobs=args.n_jobs)
        tree = int((base['children'] != other['children']).sum())
        n_changed += tree > 0
        print(f'seed {seed}: children_differ={tree}/{base["children"].size} '
              f'z_rel={_rel_diff(base["stat_obs"], other["stat_obs"]):.1e} '
              f'pval_identical='
              f'{np.array_equal(base["pval"], other["pval"], equal_nan=True)} '
              f'score_identical={base["score"] == other["score"]}')
    print(f'trees moved by round-off: {n_changed}/{args.n_seed}')


def probe_blas(args) -> None:
    """Rebuild and refit the cell under each forced OpenBLAS kernel.

    OPENBLAS_CORETYPE is read at import, so each kernel needs its own
    process; this re-execs itself with --_child and diffs the pickles.
    """
    path_list = {}
    for coretype in CORETYPE_LIST:
        path = f'/tmp/repro_probe_{coretype}.pkl'
        env = {**os.environ, 'OPENBLAS_CORETYPE': coretype,
               'OPENBLAS_NUM_THREADS': '1'}
        subprocess.run(
            [sys.executable, __file__, 'blas', '--_child', path,
             '--seed', str(args.seed), '--effect-llr', str(args.effect_llr),
             '--n-perm-fwer', str(args.n_perm_fwer),
             '--n-perm-inner', str(args.n_perm_inner)],
            env=env, check=True)
        path_list[coretype] = path

    capture = {c: pickle.load(open(p, 'rb')) for c, p in path_list.items()}
    base = capture[CORETYPE_LIST[0]]
    for coretype in CORETYPE_LIST[1:]:
        other = capture[coretype]
        print(f'--- {CORETYPE_LIST[0]} vs {coretype} ---')
        for name in ('y_clean', 'exp_hash', 'support', 'y_plant'):
            if name == 'exp_hash':
                print(f'  {name:9s} bitwise='
                      f'{str(base[name] == other[name]):5s}')
                continue
            print(f'  {name:9s} bitwise='
                  f'{str(np.array_equal(base[name], other[name])):5s} '
                  f'rel={_rel_diff(base[name], other[name]):.3e}')
        report_diff(f'{CORETYPE_LIST[0]} vs {coretype} (fit)',
                    base['fit'], other['fit'])


def _blas_child(args) -> None:
    """Build, fit and pickle one cell for probe_blas's parent to diff."""
    import joblib
    exp_clean, exp, masks = build_cell(args.seed, args.effect_llr)
    out = dict(y_clean=exp_clean.y, exp_hash=joblib.hash(exp_clean),
               support=masks[0], y_plant=exp.y,
               fit=fit_capture(exp, masks, n_perm_fwer=args.n_perm_fwer,
                               n_perm_inner=args.n_perm_inner,
                               n_jobs=args.n_jobs))
    with open(args._child, 'wb') as handle:
        pickle.dump(out, handle)
    print(f'  built under OPENBLAS_CORETYPE='
          f'{os.environ.get("OPENBLAS_CORETYPE", "auto")}')


PROBE = {'blas': probe_blas, 'backend': probe_backend, 'numba': probe_numba,
         'jitter': probe_jitter}


def main(argv=None) -> None:
    """Parse the probe name and its knobs, then run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('probe', choices=sorted(PROBE))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-seed', type=int, default=5,
                        help='seeds for the jitter probe')
    parser.add_argument('--effect-llr', type=float, default=0.3)
    parser.add_argument('--n-perm-fwer', type=int, default=40)
    parser.add_argument('--n-perm-inner', type=int, default=40)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--_child', default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._child:
        _blas_child(args)
        return
    PROBE[args.probe](args)


if __name__ == '__main__':
    main()
