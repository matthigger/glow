"""Fit GLOW on one synthetic experiment and open it in the viewer.

A single cell of the benchmark grid, start to finish: build the images,
plant an effect, fit AnalysisGLOWSplit, launch the dashboard on the result.
Every knob is a constant below -- edit and run.

    python scripts/run_glow_viewer.py

The run leaves no trace on disk. Every build goes through the undecorated
builder (see call_uncached), so the shared benchmark cache is neither read
nor written and no provenance record is filed, and the fit is handed
straight to the viewer rather than pickled anywhere. The whole cell is
therefore rebuilt and refit each time -- minutes at NUM_VOX 25k -- and
lives only as long as the viewer process.

EFFECT_LLR is per-voxel (size-normalized): the whole-region LLR the plant
targets is ~ EFFECT_LLR * (the support's voxel count), so a weak per-voxel
value over a wide support is still a detectable effect. The paper's grid
runs 0.003 (weakest) to 0.3 (strongest); see glow._extra.benchmark.config.

DEBUG fits with keep_stat=True, which adds the viewer's PERMUTATION panel
(the per-region histogram of the FWER draws).
"""

import inspect
import time

from glow._extra.benchmark.data import (DATA_FACTORY, EFFECT_FACTORY,
                                        data_recipe)
from glow.analysis import AnalysisGLOWSplit
from glow.analysis.cluster import ClusterMode
from glow.effect.extent import ExtenterMinVar, ExtenterSphere

# ---- the images -------------------------------------------------------
# 'wgn' (white gaussian noise) or 'hcp' (real diffusion maps, DUA-gated).
# 'hcp' is the one path that does write: it stages its npy bundle from the
# niftis if that is missing (the dataset, not the cache; see hcp.py).
SOURCE = 'wgn'
# voxels analysed: one connected sphere cropped out of the volume.
NUM_VOX = 25_000
# subjects. Half go to the segmentation fold, half carry the statistics.
NUM_IMG = 100
# imaging features per voxel (the multivariate response dimension).
B = 1
# drives the whole realization: the noise draw, the crop, the design.
SEED = 0

# ---- the planted effect -----------------------------------------------
# per-voxel LLR target; None plants nothing (the null / calibration case).
EFFECT_LLR = 0.003
# support size, as a fraction of the analysed volume.
EFFECT_N_VOX_FRAC = 0.1

# ---- the analysis -----------------------------------------------------
N_PERM_FWER = 5000
ALPHA_FWER = 0.05
# smallest region admitted to the FWER family. Pre-specify it; tuning it
# against results reintroduces the selection the split is there to remove.
MIN_VOX = 1
# FOCUS clusters on the contrast subspace, GLM_ERROR on the whole design
# space, NAIVE on raw y. GLM_ERROR is the arm the paper reports.
CLUSTER_MODE = ClusterMode.GLM_ERROR
# share of the subjects the Ward tree is built on.
FRAC_SEGMENT = 0.5
SPLIT_SEED = 0

# ---- debug mode -------------------------------------------------------
# True fits with keep_stat=True, which adds the viewer's PERMUTATION
# panel: the scatter's one point per region, opened up into a histogram of
# every draw behind it. Changes no result -- it only keeps the matrix the
# summary was already read off -- but costs (N_PERM_FWER + 1) x num_reg
# float64, ~200 MB in memory at the constants above.
DEBUG = True

# ---- running it -------------------------------------------------------
# False for CPU, True to require a device, 'auto' to take one if visible.
GPU = 'auto'
VERBOSE = True

# ---- the viewer -------------------------------------------------------
PORT = 8050
# region ceiling: a bigger tree is cut to the largest regions fitting under
# it (a size cutoff, so one region size is never split). 0 shows them all.
MAX_REGIONS = 10_000


def call_uncached(fnc, *args, **kwargs):
    """Call a benchmark builder with its cache and recorder peeled off.

    The builders in glow._extra.benchmark.data are wrapped @MEMORY.cache
    over @RECORDER, so calling one looks the cell up in the shared joblib
    cache and, on a miss, writes both a cache entry and a provenance
    record. inspect.unwrap walks past both to the raw function, which does
    the same build in memory alone. Worth the detour rather than a plain
    call: an ad-hoc build landing on a benchmark cell's key rewrites that
    record with a fresh exp hash, dropping every finished leaf that
    consumed the old one out of config_results_df.

    Args:
        fnc: a decorated builder (DATA_FACTORY / EFFECT_FACTORY value).

    Returns:
        whatever the raw builder returns.
    """
    return inspect.unwrap(fnc)(*args, **kwargs)


def build_exp():
    """Build the experiment and plant the effect, touching no cache.

    Returns:
        exp: the Experiment to fit, effect included.
        mask_target (np.array | None): the realized (X, Y, Z) bool support,
            or None when EFFECT_LLR is None.
    """
    extenter = ExtenterSphere(n_vox=NUM_VOX, connected=True, contiguous=True,
                              seed=SEED)
    if SOURCE == 'wgn':
        # the box is sized from NUM_VOX so the crop fits inside it
        side = round(NUM_VOX ** (1 / 3)) + 1
        kwargs_data = dict(source='wgn', shape=(side,) * 3, b=B,
                           num_img=NUM_IMG, seed=SEED, extenter=extenter)
    else:
        from glow._extra.benchmark import hcp
        kwargs_data = dict(source='hcp', hcp_feats=hcp.HCP_FEATS[:B],
                           seed=SEED, extenter=extenter)

    print(f'building {SOURCE} experiment ...')
    # source picks the builder, so it is not one of the builder's own kwargs
    kwargs_build = {k: v for k, v in kwargs_data.items() if k != 'source'}
    exp = call_uncached(DATA_FACTORY[SOURCE], **kwargs_build)
    print(f'  y={exp.y.shape}  x={exp.x.shape}')

    if EFFECT_LLR is None:
        print('  nothing planted (null case)')
        return exp, None

    exp, mask_target_list = call_uncached(
        EFFECT_FACTORY['single'], exp,
        parent_uid=data_recipe(kwargs_data).uid,
        effect_llr=float(EFFECT_LLR), extenter_cls=ExtenterMinVar,
        n_vox_frac=EFFECT_N_VOX_FRAC, seed_from_exp=True)
    mask_target = mask_target_list[0]
    print(f'  planted effect_llr={EFFECT_LLR} in '
          f'{int(mask_target.sum())} voxels')
    return exp, mask_target


def fit_glow(exp):
    """Fit AnalysisGLOWSplit on exp with the parameters above."""
    ana = AnalysisGLOWSplit(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER,
                            min_vox=MIN_VOX, cluster_mode=CLUSTER_MODE,
                            frac_segment=FRAC_SEGMENT,
                            split_seed=SPLIT_SEED, keep_stat=DEBUG)
    print(f'fitting {ana!r} ...')
    t0 = time.time()
    ana.fit(exp, gpu=GPU, verbose=VERBOSE)
    print(f'  fit in {time.time() - t0:.1f}s, '
          f'{len(ana.effect_list)} effect(s) discovered')
    return ana


def main():
    """Build, fit, and launch the viewer, all in memory."""
    exp, mask_target = build_exp()
    ana = fit_glow(exp)

    if ana.stat is not None:
        print(f'  debug mode: PERMUTATION panel on (draws {ana.stat.shape})')

    from glow._extra.viewer import launch
    launch(ana, exp, mask_target=mask_target, port=PORT,
           max_regions=MAX_REGIONS)


if __name__ == '__main__':
    main()
