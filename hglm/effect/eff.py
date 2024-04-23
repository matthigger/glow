import numpy as np
from scipy.stats import f

from hglm.f_stat import get_f_stat, get_f_degrees


class Effect:
    """ stores an effect

    anything passed to constructor besides mask & beta are stored as
    attribute (useful for storage, but won't impact use of object)

    Attributes:
        mask (np.array): True where effect, false otherwise
        y_mean (np.array): (b, num_img) mean imaging feature observed in region
    """

    @classmethod
    def from_exp_mask(cls, exp, mask, **kwargs):
        # build y corresponding to effect region
        vox_list = exp.mask_idx[mask]
        y = exp.y[..., tuple(vox_list)]

        return cls.from_x_y_contrast(x=exp.x, y=y, contrast=exp.contrast,
                                     mask=mask, **kwargs)

    @classmethod
    def from_x_y_contrast(cls, x, y, contrast, **kwargs):
        # avoid circular dependency effect.eff & experiment.analysis
        from hglm.experiment.regress import ComputeRegress, get_size_yout_ybar

        b, num_img, reg_size = y.shape

        # compute f stat & f's p-value
        f_stat = get_f_stat(x, y, contrast)
        a = (~contrast).sum(), contrast.size
        dfn, dfd = get_f_degrees(num_img, reg_size, a)
        p_val = 1 - f.cdf(f_stat, dfn=dfn, dfd=dfd)

        # compute eps
        x = x[~contrast, :], x
        comp_reg = tuple(ComputeRegress(_x) for _x in x)
        args = get_size_yout_ybar(y)
        eps = tuple(_comp_reg.get_eps(*args) for _comp_reg in comp_reg)

        # compute y_mean & beta
        y_mean = y.mean(axis=2)
        beta = tuple(y_mean @ np.linalg.pinv(_x) for _x in x)

        return cls(f_stat=f_stat, f_stat_p_val=p_val, beta=beta, y_mean=y_mean,
                   eps=eps, **kwargs)

    def __init__(self, mask, y_mean, **kwargs):
        self.mask = mask
        self.y_mean = y_mean

        # all other inputs are to be stored
        self.__dict__.update(kwargs)
