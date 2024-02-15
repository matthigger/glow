from hrba.f_stat import *
from hrba.effect.impose import *
from ..helper import generate_dummy_data


def test_compute_offset():
    # range(100) takes ~35 seconds to run
    for seed in range(1):
        # generate dummy data
        x, y, contrast = generate_dummy_data(seed=seed)

        for f_stat_exp in (0, 1, 10, 100, 1e6):
            # validate that f stat is achieved
            offset = compute_offset(x=x, y=y, contrast=contrast,
                                    f_stat=f_stat_exp)
            f_stat_obs = get_f_stat(x=x, y=y + offset[..., np.newaxis],
                                    contrast=contrast)
            assert np.isclose(f_stat_obs, f_stat_exp)
