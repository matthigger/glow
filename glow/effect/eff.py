"""Effect objects: planted synthetic effects and estimated effect regions."""

import numpy as np

from glow.analysis.mancova import get_mancova


class EffectEstimate:
    """An effect region with MANCOVA decomposition.

    Attributes:
        mask (np.array): boolean, True inside the effect
        y_mean (np.array): (b, num_img) mean imaging feature in region
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        reg_idx (int): region index in the Ward hierarchy (discovery)
        pval_fwer (float): FWER-corrected p-value (discovery)
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

    def __init__(self, mask, y_mean, *, e=None, h=None, reg_idx=None,
                 pval_fwer=None):
        self.mask = mask
        self.y_mean = y_mean
        self.e = e
        self.h = h
        self.reg_idx = reg_idx
        self.pval_fwer = pval_fwer

    def __repr__(self):
        """Class name + region size and whichever identity scalars are set.

        num_vox (the mask size) always shows; reg_idx / pval_fwer (set on a
        discovered effect) and effect_llr / seed (set on a planted one) show
        only when present, so a printed effect reads in the viewer or a
        recorded cell instead of '<...object at 0x...>'. The heavy arrays
        (y_mean, e, h) are omitted.
        """
        parts = []
        if self.mask is not None:
            parts.append(f'num_vox={int(self.mask.sum())}')
        for name in ('reg_idx', 'pval_fwer', 'effect_llr', 'seed'):
            v = getattr(self, name)
            if v is not None:
                parts.append(f'{name}={v}')
        return f'{type(self).__name__}({", ".join(parts)})'

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


class EffectSynthetic:
    """A planted (synthetic) effect spec.

    The spec fully determines the planted effect; fit(exp) returns the
    modified experiment and the realized support as an (exp_eff, mask) pair,
    leaving the spec unchanged. Supply exactly one of extenter or mask. When
    extenter is given it carries its own RNG seed; the seed field here
    drives only the imposed direction (angle).

    The benchmark pipeline always builds effects by extenter (mask stays None);
    the explicit mask path is kept for ad-hoc use (the viewer).

    Attributes:
        effect_llr (float): per-voxel LLR target.
        extenter (Extenter | None): how to sample the support. XOR with mask.
        mask (np.array | None): pre-known boolean support, made read-only on
            assignment. XOR with extenter.
        seed (int | None): RNG seed for the imposed direction when angle is
            given (it sets the rotation reference).
        angle (float | None): if given, impose the effect along a direction
            sampled at this rotation (degrees) from seed
            (glow.effect.impose.sample_beta_direction, then impose_effect);
            effects sharing a seed are separated by their angle difference.
            If None, the direction is inherited from the data
            (glow.effect.impose.compute_offset).
        purge_interest (bool): when imposing a direction, subtract the
            region's existing interest coefficient so the recovered effect
            equals the imposed direction exactly.
    """

    def __init__(self, *, effect_llr, extenter=None, mask=None, seed=None,
                 angle=None, purge_interest=True):
        if (extenter is None) == (mask is None):
            raise ValueError('extenter xor mask required')
        if angle is not None and seed is None:
            raise ValueError('angle requires seed (it sets the rotation '
                             'reference for the imposed direction)')
        self.effect_llr = float(effect_llr)
        self.extenter = extenter
        self.seed = None if seed is None else int(seed)
        self.angle = None if angle is None else float(angle)
        self.purge_interest = bool(purge_interest)
        if mask is None:
            self.mask = None
        else:
            mask = np.ascontiguousarray(mask, dtype=bool)
            mask.flags.writeable = False
            self.mask = mask

    def __repr__(self):
        """Class name + the planting spec: effect_llr, whichever of
        extenter / mask defines the support (extenter rendered recursively),
        and a set seed / angle.
        """
        parts = [f'effect_llr={self.effect_llr}']
        if self.extenter is not None:
            parts.append(f'extenter={self.extenter!r}')
        if self.mask is not None:
            parts.append(f'mask=<{int(self.mask.sum())} vox>')
        for name in ('seed', 'angle'):
            v = getattr(self, name)
            if v is not None:
                parts.append(f'{name}={v}')
        return f'{type(self).__name__}({", ".join(parts)})'

    def fit(self, exp):
        """Sample support, compute offset, impose the effect.

        The offset direction is either inherited from the data (the default,
        via compute_offset) or, when angle is given, imposed along a
        direction sampled from seed (via impose_effect).

        Args:
            exp: the experiment to plant the effect into; must already have
                x and contrast (call .sample_x() first)

        Returns:
            exp_eff: the experiment with the effect imposed.
            mask (np.array): the realized boolean support, same shape as
                exp.mask_idx.
        """
        # local import keeps glow.effect import-time cycle-free
        from .impose import compute_offset, impose_effect, sample_beta_direction

        assert exp.x is not None, 'x/contrast needed; call .sample_x()'

        if self.mask is not None:
            mask = self.mask
        else:
            mask = self.extenter(y=exp.y, mask_idx=exp.mask_idx)

        effect_idx = exp.mask_idx[mask]
        y = exp.y[:, :, effect_idx]

        if self.angle is not None:
            b = exp.y.shape[0]
            a1 = int(np.asarray(exp.contrast).sum())
            beta_direction = sample_beta_direction(
                a1=a1, b=b, angle=self.angle, seed=self.seed)
            offset = impose_effect(
                x=exp.x, y=y, contrast=exp.contrast,
                beta_direction=beta_direction, effect_llr=self.effect_llr,
                purge_interest=self.purge_interest)
            sigma_scale = None
        else:
            offset, sigma_scale = compute_offset(
                x=exp.x, y=y, contrast=exp.contrast,
                effect_llr=self.effect_llr)

        exp_eff = exp.add_offset(offset, mask=mask, sigma_scale=sigma_scale)
        return exp_eff, mask
