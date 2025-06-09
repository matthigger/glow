import scipy.stats

from hglm.experiment import get_manova, wilks_to_chi2, get_wilks


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
        b, num_img, num_vox = y.shape

        # compute stats (to be stored)
        e, h = get_manova(x, y, contrast)
        wilks = get_wilks(e, h)
        chi2, df = wilks_to_chi2(wilks, contrast=contrast, b=b, n=num_img)
        pval = 1 - scipy.stats.chi2.cdf(chi2, df=df)

        return cls(y_mean=y.mean(axis=2), e=e, h=h, wilks=wilks, chi2=chi2,
                   pval=pval, **kwargs)

    def __init__(self, mask, y_mean, **kwargs):
        self.mask = mask
        self.y_mean = y_mean

        # all other inputs are to be stored
        self.__dict__.update(kwargs)
