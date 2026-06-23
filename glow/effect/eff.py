"""Effect objects: planted synthetic effects and estimated effect regions."""

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class EffectSynthetic:
    """A planted (synthetic) effect: a frozen, immutable spec.

    The spec fully determines the planted effect; fit(exp) returns the
    modified experiment and the realized support as an (exp_eff, mask) pair,
    leaving the spec unchanged. Supply exactly one of extenter or mask. When
    extenter is given it carries its own RNG seed; the seed field here
    drives only the imposed direction (angle).

    The benchmark pipeline always builds effects by extenter (mask stays None),
    so the recorded provenance (to_record) is the extenter recipe; the explicit
    mask path is kept for ad-hoc use (the viewer) but is not recorded.

    Unlike the extenter / data-source specs, EffectSynthetic carries an
    ndarray (mask) and is never a cache key, so it uses identity equality
    (eq=False) rather than a value hash.

    Attributes:
        effect_llr (float): per-voxel LLR target.
        extenter (Extenter | None): how to sample the support. XOR with mask.
        mask (np.array | None): pre-known boolean support, frozen on
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

    effect_llr: float
    extenter: object = None
    mask: object = None
    seed: int = None
    angle: float = None
    purge_interest: bool = True

    def __post_init__(self):
        if (self.extenter is None) == (self.mask is None):
            raise ValueError('extenter xor mask required')
        if self.angle is not None and self.seed is None:
            raise ValueError('angle requires seed (it sets the rotation '
                             'reference for the imposed direction)')
        object.__setattr__(self, 'effect_llr', float(self.effect_llr))
        object.__setattr__(self, 'seed',
                           None if self.seed is None else int(self.seed))
        object.__setattr__(self, 'angle',
                           None if self.angle is None else float(self.angle))
        object.__setattr__(self, 'purge_interest', bool(self.purge_interest))
        if self.mask is not None:
            mask = np.ascontiguousarray(self.mask, dtype=bool)
            mask.flags.writeable = False
            object.__setattr__(self, 'mask', mask)

    def to_record(self) -> dict:
        """JSON-friendly recipe for provenance: the spec without the mask.

        The recorder serializes any value exposing to_record (see
        glow.benchmark.recorder). The realized support is omitted: in the
        pipeline it is a deterministic function of the extenter, and recording
        it would dump a boolean array. extenter recurses via its own to_record.
        """
        return {'kind': type(self).__name__,
                'effect_llr': self.effect_llr,
                'extenter': (None if self.extenter is None
                             else self.extenter.to_record()),
                'seed': self.seed,
                'angle': self.angle,
                'purge_interest': self.purge_interest}

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
