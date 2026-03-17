import numpy as np

from glow.effect.impose import compute_offset
from glow.experiment import Experiment, get_mancova
from glow.experiment.mancova import get_llr, get_roughness
from glow.experiment.sigma import stretch_sigma


def test_compute_offset():
    """Offset achieves target LLR (no roughness constraint)."""
    for seed in range(1):
        exp = Experiment.from_gauss(seed=seed)
        x, y, contrast = exp.x, exp.y, exp.contrast

        for effect_llr_exp in [0, 0.5, 1, 3]:
            offset, sigma_scale = compute_offset(
                x=x, y=y, contrast=contrast, effect_llr=effect_llr_exp)

            assert sigma_scale is None

            _y = y + offset[..., np.newaxis]
            e, h, _ = get_mancova(x=x, y=_y, contrast=contrast)
            effect_llr_obs = get_llr(e, h, size_normalize=True)

            assert np.isclose(effect_llr_exp, effect_llr_obs)


def _verify_offset_roughness(x, y, contrast, target_llr, target_rough):
    """Helper: impose and verify both LLR and roughness."""
    offset, sigma_scale = compute_offset(
        x=x, y=y, contrast=contrast,
        effect_llr=target_llr, roughness=target_rough,
    )
    assert sigma_scale is not None and sigma_scale > 0

    y_new = y + offset[..., np.newaxis]
    y_new = stretch_sigma(y_new, sigma_scale)
    e, h, sigma = get_mancova(x=x, y=y_new, contrast=contrast)

    llr_obs = get_llr(e, h, size_normalize=True)
    rough_obs = get_roughness(e, sigma)

    assert np.isclose(target_llr, llr_obs, atol=1e-3), \
        f'LLR mismatch: {llr_obs:.4f} vs {target_llr}'
    assert np.isclose(target_rough, rough_obs, atol=1e-2), \
        f'roughness mismatch: {rough_obs:.4f} vs {target_rough}'


def test_compute_offset_roughness():
    """Joint optimisation hits both LLR and roughness targets."""
    exp = Experiment.from_gauss(seed=0, shape=(4, 4, 4), num_img=50, b=2)
    x, y, contrast = exp.x, exp.y, exp.contrast

    for target_llr in [0.3, 1.0]:
        for target_rough in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _verify_offset_roughness(x, y, contrast, target_llr, target_rough)


def test_roughness_sigma_scale_1_when_none():
    """When roughness=None, sigma_scale is 1.0."""
    exp = Experiment.from_gauss(seed=0, shape=(3, 3, 3), num_img=20, b=2)
    offset, sigma_scale = compute_offset(
        x=exp.x, y=exp.y, contrast=exp.contrast,
        effect_llr=0.5, roughness=None,
    )
    assert sigma_scale is None
