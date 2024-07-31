from hglm.effect.impose import *
from hglm.experiment import scale_sigma
from hglm.f_stat import *
from ..helper import generate_dummy_data


def test_compute_offset():
    # range(100) takes ~35 seconds to run
    for seed in range(1):
        # generate dummy data
        x, y, contrast = generate_dummy_data(seed=seed)

        for f_stat_exp in (0, 1, 10, 100, 1e6):
            for rough_exp in (None, 0, 1, 10):
                # compute offset and space_cov_scale needed to achieve f stat
                offset, sigma_gain, rough = compute_offset(x=x, y=y,
                                                           contrast=contrast,
                                                           f_stat=f_stat_exp,
                                                           rough=rough_exp)

                # impose effect that f stat is achieved
                _y = y + offset[..., np.newaxis]
                assert (sigma_gain is None) == (rough_exp is None), \
                    'sigma_gain only when we pass desired roughness'
                if sigma_gain is not None:
                    _y = scale_sigma(y=_y, gain=sigma_gain)

                # confirm proper f stat achieved
                f_stat_obs = get_f_stat(x=x, y=_y, contrast=contrast)
                assert np.isclose(f_stat_obs, f_stat_exp)

                if rough_exp is not None:
                    # ensure desired roughness is achieved
                    assert np.isclose(rough_exp, rough)
