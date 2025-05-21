from itertools import product

from hglm.effect.impose import *
from hglm.experiment import get_manova, wilks_to_chi2, get_wilks, Experiment, \
    scale_sigma


def test_compute_offset():
    for seed, pval_exp, rough in product(range(10),
                                         np.logspace(0, -3, 3),
                                         (None, 0, .5, 1)):
        exp = Experiment.from_gauss(seed=seed)
        x, y, contrast = exp.x, exp.y, exp.contrast
        b, num_img, num_vox = y.shape

        # compute offset and space_cov_scale needed to achieve f stat
        offset, sigma_gain, _rough = compute_offset(x=x, y=y,
                                                    contrast=contrast,
                                                    pval=pval_exp,
                                                    rough=rough)

        if rough is not None:
            assert np.isclose(_rough, rough, atol=1e-6), \
                'roughness not achieved'

        # apply offset & scale spatial covariance
        _y = y + offset[..., np.newaxis]
        if sigma_gain is not None:
            _y = scale_sigma(y=_y, gain=sigma_gain)

        # ensure proper pval achieved
        e, h = get_manova(x, _y, contrast)
        wilks = get_wilks(e, h)
        chi2, df = wilks_to_chi2(wilks, a=contrast.sum(), b=b, n=num_img)
        pval = 1 - scipy.stats.chi2.cdf(chi2, df=df)

        assert np.isclose(pval, pval_exp, atol=1e-4)
