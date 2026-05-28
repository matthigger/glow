"""Effect objects: planted synthetic effects and estimated effect regions."""

import numpy as np

from glow.analysis.mancova import get_mancova
from glow.util import HashBySlots


class EffectEstimate:
    """An effect region with MANCOVA decomposition.

    Attributes:
        mask (np.array): boolean, True inside the effect
        y_mean (np.array): (b, num_img) mean imaging feature in region
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        seed (int): random seed used to sample the effect extent
        effect_llr (float): size-normalized LLR of the imposed effect.
            This is the per-voxel LLR contribution; the LLR you observe
            for the planted region is approximately effect_llr * |mask|
        reg_idx (int): region index in the Ward hierarchy (discovery)
        pval_fwer (float): FWER-corrected p-value (discovery)
        meta (dict): optional metadata, not used by analysis
    """

    @classmethod
    def from_exp_mask(cls, exp, mask, **kwargs):
        """Build an EffectEstimate from an experiment and a boolean mask."""
        vox_list = exp.mask_idx[mask]
        y = exp.y[..., tuple(vox_list)]

        return cls.from_x_y_contrast(x=exp.x, y=y, contrast=exp.contrast,
                                     mask=mask, **kwargs)

    @classmethod
    def from_x_y_contrast(cls, x, y, contrast, **kwargs):
        """Build an EffectEstimate from raw design, image and contrast arrays."""
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

    def is_close(self, other, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check approximate equality of two effects.

        Args:
            other: the EffectEstimate to compare against
            rtol (float): relative tolerance passed to np.allclose
            atol (float): absolute tolerance passed to np.allclose

        Returns:
            bool: True if masks match exactly and y_mean values are close
        """
        if self.mask.shape != other.mask.shape:
            return False
        if not np.array_equal(self.mask, other.mask):
            return False

        if self.y_mean.shape != other.y_mean.shape:
            return False
        if not np.allclose(self.y_mean, other.y_mean, rtol=rtol, atol=atol):
            return False
        return True

# Back-compat alias: existing call sites and external users may still reference
# Effect. Discovery code paths should migrate to EffectEstimate directly.
Effect = EffectEstimate

class EffectSynthetic(HashBySlots):
    """A planted (synthetic) effect.

    Operation parameters (set at __init__):
        extenter (Extenter | None): how to sample the support. XOR
            with mask.
        mask (np.array | None): pre-known boolean support. XOR with
            extenter. Frozen on assignment so the hash is stable.
        effect_llr (float): per-voxel LLR target.
        seed (int | None): RNG seed for extenter sampling.

    Fit outputs (populated by .fit()):
        mask_ (np.array): realized boolean support.
        offset_ (np.array): (b, num_img) offset added per voxel.
        sigma_scale_ (float | None): factor applied to within-region
            sigma.
    """

    __slots__ = ('extenter', 'mask', 'effect_llr', 'seed',
                 'mask_', 'offset_', 'sigma_scale_')

    def __init__(self, *, extenter=None, mask=None, effect_llr,
                 seed=None):
        if (extenter is None) == (mask is None):
            raise ValueError('extenter xor mask required')
        self.extenter = extenter
        if mask is not None:
            mask = np.ascontiguousarray(mask, dtype=bool)
            mask.flags.writeable = False
        self.mask = mask
        self.effect_llr = float(effect_llr)
        self.seed = None if seed is None else int(seed)
        # fit outputs (sklearn trailing underscore convention)
        self.mask_ = None
        self.offset_ = None
        self.sigma_scale_ = None

    def fit(self, exp):
        """Sample support, compute offset, and impose the effect.

        Populates the fit-output attributes (mask_, offset_, sigma_scale_).

        Args:
            exp: the experiment to plant the effect into; must already have
                x and contrast (call .sample_x() first)

        Returns:
            the experiment with the effect imposed
        """
        # local import keeps glow.effect import-time cycle-free
        from .impose import compute_offset

        assert exp.x is not None, 'x/contrast needed; call .sample_x()'

        if self.mask is not None:
            mask = self.mask
        else:
            mask = self.extenter(
                y=exp.y, mask_idx=exp.mask_idx, seed=self.seed)

        effect_idx = exp.mask_idx[mask]
        y = exp.y[:, :, effect_idx]
        offset, sigma_scale = compute_offset(
            x=exp.x, y=y, contrast=exp.contrast,
            effect_llr=self.effect_llr)

        self.mask_ = mask
        self.offset_ = offset
        self.sigma_scale_ = sigma_scale
        return self.apply(exp)

    def apply(self, exp):
        """Add the fitted offset to an experiment, returning the result."""
        return exp.add_offset(self.offset_, mask=self.mask_,
                              sigma_scale=self.sigma_scale_)
