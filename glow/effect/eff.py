import numpy as np

from glow.analysis.mancova import get_mancova


class Effect:
    """An effect region with MANCOVA decomposition.

    Attributes:
        mask (np.array): boolean, True inside the effect
        y_mean (np.array): (b, num_img) mean imaging feature in region
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        seed (int): random seed used to sample the effect extent
        effect_llr (float): size-normalized LLR of the imposed effect
        reg_idx (int): region index in the Ward hierarchy (discovery)
        pval_fwer (float): FWER-corrected p-value (discovery)
        meta (dict): optional metadata — not used by analysis
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

    def __init__(self, mask, y_mean, *, e=None, h=None,
                 seed=None, effect_llr=None, reg_idx=None,
                 pval_fwer=None, meta=None):
        self.mask = mask
        self.y_mean = y_mean
        self.e = e
        self.h = h
        self.seed = seed
        self.effect_llr = effect_llr
        self.reg_idx = reg_idx
        self.pval_fwer = pval_fwer
        self.meta = meta if meta is not None else {}

    def is_close(self, other, rtol=1e-5, atol=1e-8):
        """check approximate equality of two effects."""
        if self.mask.shape != other.mask.shape:
            return False
        if not np.array_equal(self.mask, other.mask):
            return False
        if self.y_mean.shape != other.y_mean.shape:
            return False
        if not np.allclose(self.y_mean, other.y_mean, rtol=rtol, atol=atol):
            return False
        return True
