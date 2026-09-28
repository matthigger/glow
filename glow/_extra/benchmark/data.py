"""Realize a benchmark cell's Experiment: clean data, then a planted effect.

build_clean builds an image-only Experiment, samples a design matrix x onto
it, and crops it to an Extenter's support -- the clean (effect-free) data, per
source (build_clean_wgn / build_clean_hcp). plant_effect then adds synthetic
effect(s) to a clean Experiment, returning it and the list of realized
supports, one for a single effect and two for a split (the cleaving figure).

Neither is cached. They are the two halves of realizing one cell, whose small
payload is what gets stored instead of the Experiment they produce (see
.cell), and a clean Experiment costs less to rebuild than to unpickle.
"""

from glow.effect import EffectSynthetic, Extenter, ExtenterSplit
from glow.experiment import ExperimentImageOnly

from . import hcp

# WGN's voxel edge in mm: HCP's grid (hcp.py bundle affine), so an fwhm in mm
# is one kernel in voxels on both sources. Not a kwargs_data knob, since only
# smoothing reads it and GLOW's WGN cells are identical either way.
WGN_VOX_MM = 2.0


def _sample_x_and_crop(exp_img, *, a: int, contrast, has_bias: bool,
                       extenter: Extenter, seed: int):
    """Sample the design matrix x onto an image-only experiment, then crop.

    Args:
        exp_img: image-only Experiment (y of shape (b, num_img, num_vox)).
        a (int): design-matrix feature count; ignored when contrast is given.
        contrast (np.array): (a,) boolean, True for features of interest, or
            None to sample a interest-only features.
        has_bias (bool): prepend a bias (all-ones) column to x.
        extenter (Extenter): extent whose support the experiment is cropped
            to, or None for no crop.
        seed (int): RNG seed for the x sample.

    Returns:
        Experiment with x and contrast attached, cropped to the extenter
        support when set, and the voxels that carry no signal dropped.
    """
    # sample_x requires exactly one of a / contrast
    exp = exp_img.sample_x(a=None if contrast is not None else a,
                           contrast=contrast, seed=seed, add_bias=has_bias)
    if extenter is not None:
        # data-driven extenters (ExtenterMinVar) need y; geometric ones
        # ignore it
        exp = exp.apply_mask(extenter(mask_idx=exp.mask_idx, y=exp.y))
    # after the crop, so the dropped count is over the volume actually
    # analysed, and here rather than in any one recipe's fit so every
    # method in the sweep tests the same voxels
    return exp.drop_constant_vox()


def build_clean_wgn(*, shape: tuple = (5, 5, 5), b: int = 2,
                    num_img: int = 100, a: int = 1, contrast=None,
                    has_bias: bool = True, extenter: Extenter = None,
                    seed: int = 0):
    """Build a white-Gaussian-noise Experiment (no planted effect).

    Args:
        shape (tuple[int]): spatial shape of an image; num_vox is its product.
        b (int): imaging features per voxel (y channels).
        num_img (int): number of images (subjects).
        a (int): design-matrix feature count (see _sample_x_and_crop).
        contrast (np.array): (a,) boolean of features of interest, or None.
        has_bias (bool): prepend a bias column to x.
        extenter (Extenter): extent to crop to, or None.
        seed (int): RNG seed shared by the image draw and the x sample.

    Returns:
        Experiment with y of shape (b, num_img, num_vox), x, and contrast,
        cropped to the extenter support when set, on a WGN_VOX_MM grid.
    """
    exp_img = ExperimentImageOnly.from_gauss(
        shape=shape, b=b, num_img=num_img, seed=seed, vox_mm=WGN_VOX_MM)
    return _sample_x_and_crop(exp_img, a=a, contrast=contrast,
                              has_bias=has_bias, extenter=extenter, seed=seed)


def build_clean_hcp(*, hcp_feats: tuple = hcp.HCP_FEATS, a: int = 1,
                    contrast=None, has_bias: bool = True,
                    extenter: Extenter = None, seed: int = 0):
    """Build an HCP-YA diffusion-microstructure Experiment (no planted effect).

    Args:
        hcp_feats (tuple[str]): subset of hcp.HCP_FEATS to load, in order;
            its length is b.
        a (int): design-matrix feature count (see _sample_x_and_crop).
        contrast (np.array): (a,) boolean of features of interest, or None.
        has_bias (bool): prepend a bias column to x.
        extenter (Extenter): extent to crop to, or None.
        seed (int): RNG seed for the x sample.

    Returns:
        Experiment with x and contrast attached, cropped to the extenter
        support when set.
    """
    # build from the per-feature npy bundle, not the niftis directly: the same
    # arrays from_search would load (so the experiment hashes identically), but
    # through one path that needs neither the niftis nor a DUA prompt once the
    # bundle exists -- see hcp.py. The archive's brain mask is the analysis
    # support (NODDI isovf is legitimately zero in-brain, so the maps can't
    # infer it).
    exp_img = hcp.build_exp_img_from_bundle(hcp_feats)
    return _sample_x_and_crop(exp_img, a=a, contrast=contrast,
                              has_bias=has_bias, extenter=extenter, seed=seed)


def build_clean(kwargs_data):
    """Build one clean (effect-free) Experiment from a data cell.

    The uncached half of a cell's realization: the clean Experiment is an
    intermediate that no longer earns a cache entry of its own, since the
    cell payload that replaces it is three orders of magnitude smaller
    (.cell).

    Args:
        kwargs_data (dict): one data cell, e.g. {'source': 'wgn', ...}; the
            source key selects the builder, the rest are its kwargs.

    Returns:
        exp: the clean Experiment, cropped and screened.

    Raises:
        ValueError: source is neither 'wgn' nor 'hcp'.
    """
    source = kwargs_data['source']
    if source not in CLEAN_BUILDER:
        raise ValueError(f"source must be 'wgn' or 'hcp', got {source!r}")
    kwargs = {k: v for k, v in kwargs_data.items() if k != 'source'}
    return CLEAN_BUILDER[source](**kwargs)


def plant_effect(exp, *, seed: int, kind: str = 'single', effect_llr,
                 extenter_cls, n_vox_frac=0.1, angle=None,
                 extenter_kwargs=None):
    """Plant one effect cell's synthetic effect(s) on a clean Experiment.

    The one implementation of the plant, shared by every caller. The support
    extenter is built here from extenter_cls, the resolved n_vox (n_vox_frac
    of the analysis volume), the placement seed and any extenter_kwargs, so a
    caller passes ingredients rather than a constructed Extenter.

    kind 'single' grows one support and imposes the effect along the
    direction the data already carries. kind 'split' grows one support,
    bisects it spectrally (ExtenterSplit), and imposes an effect on each half
    at angles 0 and angle, so the two differ only in orientation -- the
    cleaving setup, whose halves may differ in size because a data-driven
    Fiedler cut is not perfectly even. One seed drives the placement and the
    direction pair both.

    Args:
        exp: clean Experiment (a build_clean output) to add the effect(s) to.
        seed (int): support placement (and, for a split, direction) seed.
        kind (str): 'single' (one effect) or 'split' (two adjacent effects).
        effect_llr (float): per-voxel (size-normalized) LLR target per
            effect; the whole-region LLR observed is ~ effect_llr * n_vox
            (see glow.effect.impose).
        extenter_cls (type[Extenter]): Extenter subclass sampling the
            support, built as extenter_cls(n_vox=n_vox, seed=seed).
        n_vox_frac (float): support size as a fraction of the analysis
            volume (the count of mask_idx > -1), resolved to
            round(n_vox_frac * num_vox). For a split it is the combined
            size, cut into halves.
        angle (float): feature-direction angle between a split's two
            effects, in degrees. Required by kind 'split', unused otherwise.
        extenter_kwargs (dict | None): extra keywords for the extenter
            constructor, e.g. {'vox_init': 'center'} to plant a geometric
            support concentric with the analysis crop rather than wherever
            the seed lands (a 10% support in a spherical crop is clipped by
            the crop boundary from most uniformly drawn seed voxels).

    Returns:
        exp: the Experiment with the effect(s) added.
        mask_target_list (list): the realized (X, Y, Z) bool supports, one
            per planted effect, in plant order.

    Raises:
        ValueError: kind is neither 'single' nor 'split'.
    """
    n_vox = round(n_vox_frac * int((exp.mask_idx > -1).sum()))
    kwargs_ext = dict(n_vox=n_vox, seed=seed, **(extenter_kwargs or {}))
    if kind == 'single':
        extenter = extenter_cls(**kwargs_ext)
        exp, mask = EffectSynthetic(extenter=extenter,
                                    effect_llr=effect_llr).fit(exp)
        return exp, [mask]
    if kind == 'split':
        splitter = ExtenterSplit(base=extenter_cls(**kwargs_ext))
        mask0, mask1 = splitter.fit(mask_idx=exp.mask_idx, y=exp.y)
        for mask, ang in ((mask0, 0.0), (mask1, float(angle))):
            exp = EffectSynthetic(mask=mask, effect_llr=effect_llr,
                                  angle=ang, seed=seed).fit(exp)[0]
        return exp, [mask0, mask1]
    raise ValueError(f"kind must be 'single' or 'split', got {kind!r}")


# the one place a source name maps to the builder that realizes it, so a
# reader and a runner cannot disagree about what a cell means.
CLEAN_BUILDER = {'wgn': build_clean_wgn, 'hcp': build_clean_hcp}
