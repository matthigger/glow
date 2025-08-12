from hglm.effect.impose import *
from hglm.experiment import Experiment
from hglm.experiment import get_mancova


def test_compute_offset():
    for seed in range(1):
        exp = Experiment.from_gauss(seed=seed)
        x, y, contrast = exp.x, exp.y, exp.contrast

        for hotel_tr_exp in [0, 1, 10, 100]:
            # compute offset and space_cov_scale needed to achieve hotel_tr
            offset = compute_offset(x=x, y=y, contrast=contrast,
                                    hotel_tr=hotel_tr_exp)

            # apply offset & compute hotel_tr
            _y = y + offset[..., np.newaxis]
            e, h, _ = get_mancova(x=x, y=_y, contrast=contrast)
            hotel_tr_obs = get_hotel_tr(e, h)

            assert np.isclose(hotel_tr_exp, hotel_tr_obs)
