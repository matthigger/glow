"""Shared fixtures for viewer tests.

Builds a small AnalysisGLOW once per session (expensive) and exposes the
derived objects (DataFrame, target stats, etc.) that the viewer uses.
"""

import numpy as np
import pytest

from glow.experiment.exper import Experiment
from glow.effect import EffectSynthetic
from glow.analysis import AnalysisGLOW
from glow._extra.viewer.data import prep_df, get_feature_columns, compute_target_stats


@pytest.fixture(scope='session')
def demo_analysis():
    """Small 3D analysis (5x5x5 cube, sphere effect) for viewer tests.

    Returns (ana, exp_eff, mask_sphere): the analysis no longer stores the
    experiment, so the fixture exposes exp_eff for the viewer to consume.
    """
    shape = (5, 5, 5)
    center = np.array([s // 2 for s in shape])
    coords = np.indices(shape).reshape(3, -1).T
    dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
    mask_sphere = (dist <= 2.0).reshape(shape)

    # 20 images so each fold of GLOW's split still clears the design
    exp = Experiment.from_gauss(b=1, num_img=20, shape=shape, seed=0, a=2)
    exp_eff = EffectSynthetic(mask=mask_sphere, effect_llr=2.0, seed=0).fit(exp)[0]
    ana = AnalysisGLOW(n_perm_fwer=5).fit(exp_eff)
    return ana, exp_eff, mask_sphere


@pytest.fixture(scope='session')
def demo_analysis_2d():
    """Small 2D analysis (8x8, circle effect) for viewer layout tests.

    Returns (ana, exp_eff, mask_circle); see demo_analysis.
    """
    shape = (8, 8)
    center = np.array([s // 2 for s in shape])
    coords = np.indices(shape).reshape(2, -1).T
    dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
    mask_circle = (dist <= 2.5).reshape(shape)

    # 20 images so each fold of GLOW's split still clears the design
    exp = Experiment.from_gauss(b=1, num_img=20, shape=shape, seed=42, a=2)
    exp_eff = EffectSynthetic(mask=mask_circle, effect_llr=2.0, seed=42).fit(exp)[0]
    ana = AnalysisGLOW(n_perm_fwer=5).fit(exp_eff)
    return ana, exp_eff, mask_circle


@pytest.fixture(scope='session')
def ana_2d(demo_analysis_2d):
    return demo_analysis_2d[0]


@pytest.fixture(scope='session')
def exp_2d(demo_analysis_2d):
    return demo_analysis_2d[1]


@pytest.fixture(scope='session')
def mask_target_2d(demo_analysis_2d):
    return demo_analysis_2d[2]


@pytest.fixture(scope='session')
def ana(demo_analysis):
    return demo_analysis[0]


@pytest.fixture(scope='session')
def exp(demo_analysis):
    return demo_analysis[1]


@pytest.fixture(scope='session')
def mask_target(demo_analysis):
    return demo_analysis[2]


@pytest.fixture(scope='session')
def ana_stat(exp):
    """The 3D demo experiment refit with keep_stat=True.

    A separate fixture rather than a flag on demo_analysis: the viewer
    builds a different layout for an analysis that kept its draws, so both
    it and the default (which has none) have to stay under test.
    """
    return AnalysisGLOW(n_perm_fwer=5, keep_stat=True).fit(exp)


@pytest.fixture(scope='session')
def ana_stat_2d(exp_2d):
    """The 2D demo experiment refit with keep_stat=True; see ana_stat."""
    return AnalysisGLOW(n_perm_fwer=5, keep_stat=True).fit(exp_2d)


@pytest.fixture(scope='session')
def df_no_target(ana, exp):
    return prep_df(ana, exp)


@pytest.fixture(scope='session')
def df_with_target(ana, exp, mask_target):
    return prep_df(ana, exp, mask_target=mask_target)


@pytest.fixture(scope='session')
def target_stats(exp, mask_target):
    return compute_target_stats(exp, mask_target)


@pytest.fixture(scope='session')
def feature_cols(df_with_target):
    return get_feature_columns(df_with_target)
