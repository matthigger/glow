from hglm.effect.impose import *
from hglm.experiment import get_manova, wilks_to_chi2, get_wilks, Experiment
from hglm.f_stat import *


def test_compute_offset():
    for seed in range(1):
        exp = Experiment.from_gauss(seed=seed)
        x, y, contrast = exp.x, exp.y, exp.contrast
        b, num_img, num_vox = y.shape

        for pval_exp in np.logspace(0, -3, 5):
            # compute offset and space_cov_scale needed to achieve f stat
            offset = compute_offset(x=x, y=y,
                                    contrast=contrast,
                                    pval=pval_exp)
            _y = y + offset[..., np.newaxis]

            # ensure proper pval achieved
            e, h = get_manova(x, _y, contrast)
            wilks = get_wilks(e, h)
            chi2, df = wilks_to_chi2(wilks, a=contrast.sum(), b=b, n=num_img)
            pval = 1 - scipy.stats.chi2.cdf(chi2, df=df)

            assert np.isclose(pval, pval_exp, atol=1e-4)
