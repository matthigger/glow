from glow.effect.impose import *
from glow.experiment import Experiment
from glow.experiment import get_mancova
from glow.experiment.mancova import get_llr


def test_compute_offset():
    for seed in range(1):
        exp = Experiment.from_gauss(seed=seed)
        x, y, contrast = exp.x, exp.y, exp.contrast

        for effect_llr_exp in [0, 0.5, 1, 3]:
            offset = compute_offset(x=x, y=y, contrast=contrast,
                                    effect_llr=effect_llr_exp)

            _y = y + offset[..., np.newaxis]
            e, h, _ = get_mancova(x=x, y=_y, contrast=contrast)
            effect_llr_obs = get_llr(e, h, size_normalize=True)

            assert np.isclose(effect_llr_exp, effect_llr_obs)
