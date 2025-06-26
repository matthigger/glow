from hglm.effect.impose import *
from hglm.experiment import Experiment

def test_pval_to_stat():
    rng = np.random.default_rng(seed=0)

    samples = rng.normal(loc=0, scale=1, size=10)
    kde = scipy.stats.gaussian_kde(samples, bw_method=1)

    cdf = lambda z: kde.integrate_box_1d(-np.inf, z)
    cdf0 = cdf(0)
    cdf_clip = lambda z: max((cdf(z) - cdf0) / (1 - cdf0), 0)

    for pval in [0, 0.1, 0.5, 0.9, .9999]:
        z = pval_to_f_ratio(kde, pval)
        cdf_val = cdf_clip(z)
        assert np.isclose(cdf_val, 1 - pval, atol=1e-4)

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