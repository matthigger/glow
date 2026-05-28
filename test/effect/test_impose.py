import numpy as np
import pytest

from glow.effect.impose import compute_offset
from glow.experiment import Experiment
from glow.analysis.mancova import get_mancova
from glow.analysis.mancova import get_llr


@pytest.mark.parametrize('effect_llr_exp', [0, 0.5, 1, 3])
def test_compute_offset(effect_llr_exp):
    """Offset achieves target LLR (round-trip through get_mancova/get_llr)."""
    exp = Experiment.from_gauss(seed=0)
    x, y, contrast = exp.x, exp.y, exp.contrast

    offset, sigma_scale = compute_offset(
        x=x, y=y, contrast=contrast, effect_llr=effect_llr_exp)

    assert sigma_scale is None

    _y = y + offset[..., np.newaxis]
    e, h, _ = get_mancova(x=x, y=_y, contrast=contrast)
    effect_llr_obs = get_llr(e, h, n=1)

    assert np.isclose(effect_llr_exp, effect_llr_obs)
