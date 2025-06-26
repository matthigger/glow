from hglm.effect.impose import *
from hglm.experiment import Experiment


def test_compute_offset():
    for seed in range(1):
        exp = Experiment.from_gauss(seed=seed)
        x, y, contrast = exp.x, exp.y, exp.contrast

        for pval_exp in np.logspace(0, -3, 5):
            # compute offset and space_cov_scale needed to achieve f stat
            compute_offset(x=x, y=y, contrast=contrast, pval=pval_exp)

            # todo: ensure cdf of kde is proper