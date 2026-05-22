import numpy as np

from glow.analysis.mancova import get_mancova


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
            for the planted region is approximately ``effect_llr × |mask|``
        reg_idx (int): region index in the Ward hierarchy (discovery)
        pval_fwer (float): FWER-corrected p-value (discovery)
        meta (dict): optional metadata — not used by analysis
    """

    @classmethod
    def from_exp_mask(cls, exp, mask, **kwargs):
        """build an EffectEstimate from an experiment and a boolean mask."""
        vox_list = exp.mask_idx[mask]
        y = exp.y[..., tuple(vox_list)]

        return cls.from_x_y_contrast(x=exp.x, y=y, contrast=exp.contrast,
                                     mask=mask, **kwargs)

    @classmethod
    def from_x_y_contrast(cls, x, y, contrast, **kwargs):
        """build an EffectEstimate from raw design, image and contrast arrays."""
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

# Back-compat alias: existing call sites and external users may still reference
# ``Effect``.  Discovery code paths should migrate to ``EffectEstimate`` directly.
Effect = EffectEstimate

class EffectSynthetic:
    """A planted (synthetic) effect: realized mask + offset values, with
    provenance for replay during slim-pickle rehydration.

    Attributes:
        mask (np.array): boolean, True inside the planted region
        offset (np.array): (b, num_img) offset added per voxel in mask
        sigma_scale (float | None): factor applied to within-region sigma
        seed (int): provenance — seed used to draw the extenter
        effect_llr (float): provenance — per-voxel LLR target requested
        extenter_kind (str | None): provenance — extenter class name
        extenter_args (dict | None): provenance — extenter constructor args
    """

    def __init__(self, mask, offset, *, sigma_scale=None,
                 seed=None, effect_llr=None,
                 extenter_kind=None, extenter_args=None):
        self.mask = mask
        self.offset = offset
        self.sigma_scale = sigma_scale
        self.seed = seed
        self.effect_llr = effect_llr
        self.extenter_kind = extenter_kind
        self.extenter_args = extenter_args

    def apply(self, exp):
        """Apply this synthetic effect's mask + offset to ``exp``.

        The recipe step preserves both literal arrays so rehydration is
        exact (decision: reproducibility > marginal storage savings).
        """
        # clause item: lets explore removing all these recipe step things.  remind me what purpose they serve (its not clear to me) and, if it doesn't merit the complication, then we can get rid of it.  at the very least, could we use a decorator pattern to do this more gracefully?
        recipe_step = {
            'op': 'add_offset',
            'args': {'mask': self.mask,
                     'offset': self.offset,
                     'sigma_scale': self.sigma_scale},
        }
        return exp.add_offset(self.offset, mask=self.mask,
                              sigma_scale=self.sigma_scale,
                              recipe_step=recipe_step)

    # clause item: question: to make this simpler, maybe we shouldn't support the mask xor extenter pattern, if the user already has an extenter its only 1 line for them to ask before calling this function while its many to run it inside ... seems simpler, right?
    @classmethod
    def impose(cls, exp, *, effect_llr, extenter=None, mask=None,
               seed=None, **kwargs):
        """Synthesize and apply a new effect.

        Optimization runs once; the realized (mask, offset) are captured
        in the returned EffectSynthetic so later replay reproduces the
        post-imposition y exactly.

        Returns:
            (Experiment, EffectSynthetic): post-imposition experiment and
                the synthetic effect that was applied.
        """
        # local import keeps glow.effect import-time cycle-free
        from .impose import compute_offset

        assert exp.x is not None, 'x/contrast needed; call .sample_x()'
        assert (mask is None) != (extenter is None), \
            'either mask xor extenter required'

        if mask is None:
            mask = extenter(y=exp.y, mask_idx=exp.mask_idx, seed=seed)

        effect_idx = exp.mask_idx[mask]
        y = exp.y[:, :, effect_idx]
        offset, sigma_scale = compute_offset(
            x=exp.x, y=y, contrast=exp.contrast,
            effect_llr=effect_llr)

        synth = cls(
            mask=mask, offset=offset, sigma_scale=sigma_scale,
            seed=seed, effect_llr=effect_llr,
            extenter_kind=type(extenter).__name__ if extenter else None,
            extenter_args=(dict(extenter.__dict__) if extenter is not None
                           else None),
        )
        return synth.apply(exp), synth
