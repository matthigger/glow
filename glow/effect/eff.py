import numpy as np

from glow.experiment import get_mancova


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
        e, h, _ = get_mancova(x=x, y=y, contrast=contrast)
        return cls(y_mean=y.mean(axis=2), e=e, h=h, **kwargs)

    def __init__(self, mask, y_mean, **kwargs):
        self.mask = mask
        self.y_mean = y_mean

        # all other inputs are to be stored
        self.__dict__.update(kwargs)

    def is_close(self, other, rtol=1e-5, atol=1e-8):
        """compare two effects for approximate equality
        
        Args:
            other (Effect): effect to compare to
            rtol (float): relative tolerance for numerical comparison
            atol (float): absolute tolerance for numerical comparison
            
        Returns:
            bool: True if effects are approximately equal
        """
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
