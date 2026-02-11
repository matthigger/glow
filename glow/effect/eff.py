import numpy as np

from glow.experiment import get_mancova


class Effect:
    """stores an effect region.

    extra kwargs passed to the constructor are stored as attributes.

    Attributes:
        mask (np.array): boolean, True inside the effect
        y_mean (np.array): (b, num_img) mean imaging feature in region
    """

    @classmethod
    def from_exp_mask(cls, exp, mask, **kwargs):
        """build an Effect from an experiment and a boolean mask."""
        vox_list = exp.mask_idx[mask]
        y = exp.y[..., tuple(vox_list)]

        return cls.from_x_y_contrast(x=exp.x, y=y, contrast=exp.contrast,
                                     mask=mask, **kwargs)

    @classmethod
    def from_x_y_contrast(cls, x, y, contrast, **kwargs):
        """build an Effect from raw design, image and contrast arrays."""
        e, h, _ = get_mancova(x=x, y=y, contrast=contrast)
        return cls(y_mean=y.mean(axis=2), e=e, h=h, **kwargs)

    def __init__(self, mask, y_mean, **kwargs):
        self.mask = mask
        self.y_mean = y_mean

        # all other inputs are to be stored
        self.__dict__.update(kwargs)

    def is_close(self, other, rtol=1e-5, atol=1e-8):
        """check approximate equality of two effects."""
        # check mask shape and values
        if self.mask.shape != other.mask.shape:
            return False
        if not np.array_equal(self.mask, other.mask):
            return False
        
        # check y_mean shape and values
        if self.y_mean.shape != other.y_mean.shape:
            return False
        if not np.allclose(self.y_mean, other.y_mean, rtol=rtol, atol=atol):
            return False
        
        return True
