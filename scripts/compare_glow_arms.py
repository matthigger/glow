"""Fit both GLOW arms on the same experiments and print them side by side.

The head-to-head the two arms exist for: AnalysisGLOWSplit builds one Ward
tree on a held-out fold, AnalysisGLOW rebuilds it inside every outer
permutation (see glow.analysis._glow for what each buys). Every knob is a
constant below -- edit and run.

    python scripts/compare_glow_arms.py

Two numbers per arm per cell:

  - tree dice: the best Dice any region of the arm's tree achieves against
    the planted support (score_oracle_tree) -- segmentation quality alone,
    with no significance test or pruning in it. This is where the split
    costs something: its tree is built from FRAC_SEGMENT of the images.
  - found dice: the Dice of what the arm actually discovered, so the
    segmentation, the test and the pruning together.

The two are read together rather than either alone. A per-perm fit can
segment better and still declare less, and its p-values carry weak FWER
control only -- see AnalysisGLOW, and mind the z ceiling that N_PERM_INNER
imposes (a small inner count saturates every tree's z and the test loses
resolution).

The run touches no cache and writes nothing: it builds its own experiments
through Experiment.from_gauss rather than the benchmark's cached builders.
For the full grid (real data, every metric, many seeds) add an AnalysisGLOW
entry to glow._extra.benchmark.config.ana_kwargs_dict instead -- the arm is
a recipe like any other there, at ~250x the split arm's draws.
"""

import numpy as np

from glow._extra.benchmark.score import score_effects, score_oracle_tree
from glow.analysis import AnalysisGLOW, AnalysisGLOWSplit
from glow.analysis.cluster import ClusterMode
from glow.effect import EffectSynthetic, ExtenterSphere
from glow.experiment import Experiment
from glow.mask import stats_from_counts

# ---- the images -------------------------------------------------------
SHAPE = (10, 10, 10)
NUM_IMG = 20
B = 2
# one cell per (EFFECT_LLR, seed) pair; seeds drive noise, crop and design.
SEED_LIST = (0, 1, 2)

# ---- the planted effect -----------------------------------------------
# per-voxel LLR target: the region LLR is ~ EFFECT_LLR * support voxels.
EFFECT_LLR_LIST = (0.1, 0.2, 0.4)
EFFECT_RADIUS = 3

# ---- the analyses -----------------------------------------------------
N_PERM_FWER = 20
# inner draws per outer perm (the per-perm arm only). Caps every z at
# N_PERM_INNER / sqrt(N_PERM_INNER + 1), so keep it well above the z the
# effect reaches -- see AnalysisGLOW.
N_PERM_INNER = 50
ALPHA_FWER = 0.05
MIN_VOX = 2
CLUSTER_MODE = ClusterMode.FOCUS
FRAC_SEGMENT = 0.5
N_JOBS = 4


def get_dice(counts) -> float:
    """Return the Dice of one {tp, fp, tn, fn} count set.

    stats_from_counts works elementwise over array-likes and rejects bare
    scalars, so the counts go in as length-1 arrays.

    Args:
        counts (dict): {tp, fp, tn, fn} integer counts.

    Returns:
        dice (float): NaN where the counts leave it undefined (0/0).
    """
    arrays = {k: np.array([counts[k]]) for k in ('tp', 'fp', 'tn', 'fn')}
    return float(stats_from_counts(**arrays)['dice'][0])


def get_ana_list() -> list:
    """Build the two arms to compare, at the constants above.

    Returns:
        list: (label, Analysis) pairs, the split arm first.
    """
    shared = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER,
                  min_vox=MIN_VOX, cluster_mode=CLUSTER_MODE)
    return [('split', AnalysisGLOWSplit(frac_segment=FRAC_SEGMENT, **shared)),
            ('perperm', AnalysisGLOW(n_perm_inner=N_PERM_INNER, **shared))]


def run_cell(effect_llr: float, seed: int) -> dict:
    """Plant one effect and fit every arm on it.

    Args:
        effect_llr (float): per-voxel LLR target for the plant.
        seed (int): drives the images, the support and the design.

    Returns:
        dict: label -> {tree_dice, found_dice, min_pval, n_found}
    """
    exp = Experiment.from_gauss(b=B, a=2, shape=SHAPE, num_img=NUM_IMG,
                                seed=seed)
    eff = EffectSynthetic(
        extenter=ExtenterSphere(radius=EFFECT_RADIUS, seed=seed),
        effect_llr=effect_llr)
    exp_eff, mask_target = eff.fit(exp)
    mask_active = exp.mask_idx > -1

    out = {}
    for label, ana in get_ana_list():
        ana.fit(exp_eff, n_jobs=N_JOBS)
        score = score_effects(ana, [mask_target], mask_active)
        out[label] = dict(
            tree_dice=get_dice(score_oracle_tree(ana.children, mask_target,
                                                 exp.mask_idx)),
            found_dice=get_dice(score['target']),
            min_pval=score['min_pval'],
            n_found=score['n_pred'])
    return out


def main() -> None:
    """Run the grid and print one line per cell, then a per-llr mean."""
    labels = [label for label, _ in get_ana_list()]
    head = ''.join(f'{lab + " tree":>14}{lab + " found":>15}'
                   for lab in labels)
    print(f'{"llr":>5}{"seed":>5}{head}')

    cell_dict = {}
    for effect_llr in EFFECT_LLR_LIST:
        for seed in SEED_LIST:
            out = run_cell(effect_llr, seed)
            cell_dict[(effect_llr, seed)] = out
            row = ''.join(f'{out[lab]["tree_dice"]:14.3f}'
                          f'{out[lab]["found_dice"]:15.3f}'
                          for lab in labels)
            print(f'{effect_llr:5.2f}{seed:5d}{row}')

    print()
    for effect_llr in EFFECT_LLR_LIST:
        cells = [out for (llr, _), out in cell_dict.items()
                 if llr == effect_llr]
        parts = []
        for lab in labels:
            for key in ('tree_dice', 'found_dice'):
                mean = np.nanmean([c[lab][key] for c in cells])
                parts.append(f'{lab}.{key}={mean:.3f}')
        print(f'llr={effect_llr:.2f} mean ' + '  '.join(parts))


if __name__ == '__main__':
    main()
