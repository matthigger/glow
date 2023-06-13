import numpy as np
from scipy.stats import f

from hrba.f_stat import get_f_stat, get_f_degrees, get_tr_eps


class Effect:
    """ stores an effect

    anything passed to constructor besides mask & beta are stored as
    attribute (useful for storage, but won't impact use of object)

    Attributes:
        mask (np.array): True where effect, false otherwise
        beta (tuple): two (a, b) arrays MMSE mapping from x to y,
            corresponding to reduced and full models respectively
    """

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
                   **kwargs)

    def __init__(self, mask, beta=None, **kwargs):
        self.mask = mask
        self.beta = beta

        # all other inputs are to be stored
        self.__dict__.update(kwargs)

    def compute_sens_spec(self, mask_estimate):
        raise NotImplementedError
