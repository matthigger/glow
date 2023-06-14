import numpy as np
from scipy.stats import f

from hrba.f_stat import get_f_stat, get_f_degrees, get_tr_eps


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
        # compute f stat
        f_stat = get_f_stat(x, y, contrast)

        # compute p_val
        dfn, dfd = get_f_degrees(y, contrast)
        p_val = f.cdf(f_stat, dfn=dfn, dfd=dfd)

        # compute tr_eps
        x = [x[~contrast, :], x]
        tr_eps = tuple(get_tr_eps(_x, y) for _x in x)

        y_mean = y.mean(axis=2)
        beta = tuple(y_mean @ np.linalg.pinv(_x) for _x in x)

        return cls(f_stat=f_stat, p_val=p_val, tr_eps=tr_eps, beta=beta,
                   y_mean=y_mean, **kwargs)

    def __init__(self, mask, y_mean, **kwargs):
        self.mask = mask
        self.y_mean = y_mean

        # all other inputs are to be stored
        self.__dict__.update(kwargs)
