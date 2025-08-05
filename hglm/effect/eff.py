from hglm.experiment import get_manova


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
        e, h = get_manova(x=x, y=y, contrast=contrast)
        return cls(y_mean=y.mean(axis=2), e=e, h=h, **kwargs)

    def __init__(self, mask, y_mean, **kwargs):
        self.mask = mask
        self.y_mean = y_mean

        # all other inputs are to be stored
        self.__dict__.update(kwargs)
