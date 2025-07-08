from hglm.effect.impose import *
from hglm.experiment import Experiment
from hglm.experiment import get_manova


def test_compute_offset():
    for seed in range(1):
        exp = Experiment.from_gauss(seed=seed)
        x, y, contrast = exp.x, exp.y, exp.contrast

        for f_ratio_exp in [0, 1, 10, 100]:
            # compute offset and space_cov_scale needed to achieve f stat
            offset = compute_offset(x=x, y=y, contrast=contrast,
                                    f_ratio=f_ratio_exp)

            # apply offset & compute f_ratio
            _y = y + offset[..., np.newaxis]
            e, h = get_manova(x, _y, contrast)
            f_ratio_obs = get_f_ratio(e, h)

            assert np.isclose(f_ratio_exp, f_ratio_obs)
