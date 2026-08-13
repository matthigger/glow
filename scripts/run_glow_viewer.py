"""Fit GLOW on one synthetic experiment and open it in the viewer.

A single cell of the benchmark grid, start to finish: build the images,
plant an effect, fit AnalysisGLOW, launch the dashboard on the result.
Every knob is a constant below -- edit and run.

    python scripts/run_glow_viewer.py

The fit is the slow part (minutes at NUM_VOX 25k), so the fitted bundle is
written to BUNDLE_PATH and reused on the next run; set REFIT to force a
fresh one. The bundle is the {ana, exp, mask_target} pickle the viewer
takes directly, so it can also be opened on its own:

    python -m glow._extra.viewer <BUNDLE_PATH>

EFFECT_LLR is per-voxel (size-normalized): the whole-region LLR the plant
targets is ~ EFFECT_LLR * (the support's voxel count), so a weak per-voxel
value over a wide support is still a detectable effect. The paper's grid
runs 0.003 (weakest) to 0.3 (strongest); see glow._extra.benchmark.config.
"""

import gzip
import pathlib
import pickle
import time

from glow._extra.benchmark.data import (data_factory, data_recipe,
                                        effect_factory)
from glow.analysis import AnalysisGLOW
from glow.analysis.cluster import ClusterMode
from glow.effect.extent import ExtenterMinVar, ExtenterSphere

# ---- the images -------------------------------------------------------
# 'wgn' (white gaussian noise) or 'hcp' (real diffusion maps, DUA-gated).
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
N_PERM_FWER = 500
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

# ---- running it -------------------------------------------------------
# False for CPU, True to require a device, 'auto' to take one if visible.
GPU = 'auto'
VERBOSE = True

# ---- the viewer -------------------------------------------------------
PORT = 8050
# region ceiling: a bigger tree is cut to the largest regions fitting under
# it (a size cutoff, so one region size is never split). 0 shows them all.
MAX_REGIONS = 10_000

BUNDLE_PATH = (pathlib.Path.home() / '.local' / 'share' / 'glow' /
               'viewer_bundles' /
               f'{SOURCE}_vox{NUM_VOX}_b{B}_llr{EFFECT_LLR}_seed{SEED}.p.gz')
REFIT = False


def build_exp():
    """Build the experiment and plant the effect.

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
    exp = data_factory(**kwargs_data)
    print(f'  y={exp.y.shape}  x={exp.x.shape}')

    if EFFECT_LLR is None:
        print('  nothing planted (null case)')
        return exp, None

    exp, mask_target_list = effect_factory(
        exp, kind='single', parent_uid=data_recipe(kwargs_data).uid,
        effect_llr=float(EFFECT_LLR), extenter_cls=ExtenterMinVar,
        n_vox_frac=EFFECT_N_VOX_FRAC, seed_from_exp=True)
    mask_target = mask_target_list[0]
    print(f'  planted effect_llr={EFFECT_LLR} in '
          f'{int(mask_target.sum())} voxels')
    return exp, mask_target


def fit_glow(exp):
    """Fit AnalysisGLOW on exp with the parameters above."""
    ana = AnalysisGLOW(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER,
                       min_vox=MIN_VOX, cluster_mode=CLUSTER_MODE,
                       frac_segment=FRAC_SEGMENT, split_seed=SPLIT_SEED)
    print(f'fitting {ana!r} ...')
    t0 = time.time()
    ana.fit(exp, gpu=GPU, verbose=VERBOSE)
    print(f'  fit in {time.time() - t0:.1f}s, '
          f'{len(ana.effect_list)} effect(s) discovered')
    return ana


def main():
    """Build or load the bundle, then launch the viewer on it."""
    if BUNDLE_PATH.exists() and not REFIT:
        print(f'loading cached fit from {BUNDLE_PATH}')
        # our own bundle, written by this script -- see the module docstring
        with gzip.open(BUNDLE_PATH, 'rb') as f:
            bundle = pickle.load(f)
    else:
        exp, mask_target = build_exp()
        ana = fit_glow(exp)
        bundle = dict(ana=ana, exp=exp, mask_target=mask_target)
        BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(BUNDLE_PATH, 'wb') as f:
            pickle.dump(bundle, f)
        print(f'wrote {BUNDLE_PATH}')

    from glow._extra.viewer import launch
    launch(bundle['ana'], bundle['exp'], mask_target=bundle['mask_target'],
           port=PORT, max_regions=MAX_REGIONS)


if __name__ == '__main__':
    main()
