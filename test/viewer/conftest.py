"""Shared fixtures for viewer tests.

Builds a small AnalysisGLOW once per session (expensive) and exposes the
derived objects (DataFrame, target stats, etc.) that the viewer uses.
"""

import numpy as np
import pytest

from glow.experiment.exper import Experiment
from glow.effect import EffectSynthetic
from glow.analysis import AnalysisGLOW
from glow.viewer.data import prep_df, get_feature_columns, compute_target_stats


@pytest.fixture(scope='session')
def demo_analysis():
    """Small 3D analysis (5x5x5 cube, sphere effect) for viewer tests."""
    shape = (5, 5, 5)
    center = np.array([s // 2 for s in shape])
    coords = np.indices(shape).reshape(3, -1).T
    dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
    mask_sphere = (dist <= 2.0).reshape(shape)

    exp = Experiment.from_gauss(b=1, num_img=10, shape=shape, seed=0, a=2)
    exp_eff, _ = EffectSynthetic.impose(
        exp, effect_llr=2.0, mask=mask_sphere, seed=0)
    ana = AnalysisGLOW(exp_eff, n_perm_fwer=5, n_perm_inner=10).fit()
    return ana, mask_sphere


@pytest.fixture(scope='session')
def demo_analysis_2d():
    """Small 2D analysis (8x8, circle effect) for viewer layout tests."""
    shape = (8, 8)
    center = np.array([s // 2 for s in shape])
    coords = np.indices(shape).reshape(2, -1).T
    dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
    mask_circle = (dist <= 2.5).reshape(shape)

    exp = Experiment.from_gauss(b=1, num_img=6, shape=shape, seed=42, a=2)
    exp_eff, _ = EffectSynthetic.impose(
        exp, effect_llr=2.0, mask=mask_circle, seed=42)
    ana = AnalysisGLOW(exp_eff, n_perm_fwer=5, n_perm_inner=10).fit()
    return ana, mask_circle


@pytest.fixture(scope='session')
def ana_2d(demo_analysis_2d):
    return demo_analysis_2d[0]


@pytest.fixture(scope='session')
def mask_target_2d(demo_analysis_2d):
    return demo_analysis_2d[1]


@pytest.fixture(scope='session')
def ana(demo_analysis):
    return demo_analysis[0]


@pytest.fixture(scope='session')
def mask_target(demo_analysis):
    return demo_analysis[1]


@pytest.fixture(scope='session')
def df_no_target(ana):
    return prep_df(ana)


@pytest.fixture(scope='session')
def df_with_target(ana, mask_target):
    return prep_df(ana, mask_target=mask_target)


@pytest.fixture(scope='session')
def target_stats(ana, mask_target):
    return compute_target_stats(ana, mask_target)


@pytest.fixture(scope='session')
def feature_cols(df_with_target):
    return get_feature_columns(df_with_target)
