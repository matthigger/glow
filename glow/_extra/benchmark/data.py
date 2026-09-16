"""Build benchmark Experiments: clean data, then a planted synthetic effect.

data_factory_wgn / data_factory_hcp each build an image-only Experiment,
sample a design matrix x onto it, and crop it to an Extenter's support -- the
clean (effect-free) data; data_factory dispatches to them on source.
effect_factory then plants synthetic effect(s) on a clean Experiment, returning
the planted Experiment and the list of realized supports; like data_factory it
is a dispatcher (on kind) over the recorded builders effect_factory_single (one
effect) / effect_factory_split (two adjacent effects for the cleaving figure),
which share one (exp, mask_target_list) contract so the driver threads the
support list into the leaf whatever the effect count. (Raw, effect-free runs
skip the effect stage and feed a clean Experiment straight to an Analysis.)

Every build is memoised on disk (MEMORY) with the recorder nested inside the
cache, so only a real (cache-miss) build is recorded, keyed by joblib's own
args hash -- a record lines up one-to-one with its cached artifact. All
builders share MEMORY / RECORDER, so a clean build and the plant that consumes
it link into one provenance DAG.
"""
import joblib

from glow.effect import EffectSynthetic, Extenter, ExtenterSplit
from glow.experiment import ExperimentImageOnly

from . import hcp
from .file import get_path_cache, get_path_records
from .recipe import raw_fnc, recipe_for_call, seed_from_uid
from .recorder import Recorder

# disk memoisation of the experiment builds, keyed on the build inputs, so a
# repeated (source, ...) cell is loaded rather than rebuilt across runs.
MEMORY = joblib.Memory(get_path_cache(), verbose=0)

# captures each build's inputs / output / timing for provenance (see Recorder).
# Keyed by joblib's args hash, mirrored to the records dir beside the cache.
# Every edge is declared: a consumer is passed its parent's uid, so nothing
# here hashes an array to discover lineage (see Recorder).
RECORDER = Recorder(folder=get_path_records())


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


@MEMORY.cache
@RECORDER(output_name='exp')
def data_factory_wgn(*, shape: tuple = (5, 5, 5), b: int = 2,
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
        cropped to the extenter support when set.
    """
    exp_img = ExperimentImageOnly.from_gauss(
        shape=shape, b=b, num_img=num_img, seed=seed)
    return _sample_x_and_crop(exp_img, a=a, contrast=contrast,
                              has_bias=has_bias, extenter=extenter, seed=seed)


@MEMORY.cache
@RECORDER(output_name='exp')
def data_factory_hcp(*, hcp_feats: tuple = hcp.HCP_FEATS, a: int = 1,
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
    """
    kwargs = {k: v for k, v in kwargs_data.items() if k != 'source'}
    # the builder's own body, not its memoised wrapper
    return raw_fnc(DATA_FACTORY[kwargs_data['source']])(**kwargs)


def plant_effect(exp, *, seed: int, kind: str = 'single', effect_llr,
                 extenter_cls, n_vox_frac=0.1, angle=None):
    """Plant one effect cell's synthetic effect(s) on a clean Experiment.

    The one implementation of the plant, shared by every caller. The support
    extenter is built here from extenter_cls, the resolved n_vox (n_vox_frac
    of the analysis volume) and the placement seed, so a caller passes
    ingredients rather than a constructed Extenter.

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

    Returns:
        exp: the Experiment with the effect(s) added.
        mask_target_list (list): the realized (X, Y, Z) bool supports, one
            per planted effect, in plant order.

    Raises:
        ValueError: kind is neither 'single' nor 'split'.
    """
    n_vox = round(n_vox_frac * int((exp.mask_idx > -1).sum()))
    if kind == 'single':
        extenter = extenter_cls(n_vox=n_vox, seed=seed)
        exp, mask = EffectSynthetic(extenter=extenter,
                                    effect_llr=effect_llr).fit(exp)
        return exp, [mask]
    if kind == 'split':
        splitter = ExtenterSplit(base=extenter_cls(n_vox=n_vox, seed=seed))
        mask0, mask1 = splitter.fit(mask_idx=exp.mask_idx, y=exp.y)
        for mask, ang in ((mask0, 0.0), (mask1, float(angle))):
            exp = EffectSynthetic(mask=mask, effect_llr=effect_llr,
                                  angle=ang, seed=seed).fit(exp)[0]
        return exp, [mask0, mask1]
    raise ValueError(f"kind must be 'single' or 'split', got {kind!r}")


def data_factory(source: str, **kwargs):
    """Build a clean Experiment from 'wgn' or 'hcp' (forwards kwargs).

    Args:
        source (str): 'wgn' or 'hcp', selecting the builder kwargs go to.

    Returns:
        the selected builder's Experiment.

    Raises:
        ValueError: if source is neither 'wgn' nor 'hcp'.
    """
    if source not in DATA_FACTORY:
        raise ValueError(f"source must be 'wgn' or 'hcp', got {source!r}")
    return DATA_FACTORY[source](**kwargs)


# Each planting builder takes parent_uid as its first keyword-only parameter,
# ahead of the defaulted ones: joblib's filter_args resolves an omitted default
# by indexing from the end of the signature, so a required parameter after a
# defaulted one makes every call raise (see the note in run.py).
def data_recipe(kwargs_data):
    """Return the recipe of one data cell, building nothing.

    Args:
        kwargs_data (dict): one data_factory cell, e.g. {'source': 'wgn', ...}.

    Returns:
        recipe (Recipe): the clean exp's declared identity; its uid is the
            parent_uid every consumer of that exp is passed.
    """
    kwargs = {k: v for k, v in kwargs_data.items() if k != 'source'}
    return recipe_for_call(DATA_FACTORY[kwargs_data['source']], kwargs)


def effect_recipe(kwargs_effect, parent_uid: str):
    """Return the recipe of one effect cell planted on a given clean exp.

    Args:
        kwargs_effect (dict): one effect_factory cell, e.g. {'kind': 'single',
            ...}.
        parent_uid (str): the clean exp's uid (data_recipe(...).uid).

    Returns:
        recipe (Recipe): the planted exp's declared identity.
    """
    kwargs = {k: v for k, v in kwargs_effect.items() if k != 'kind'}
    return recipe_for_call(EFFECT_FACTORY[kwargs_effect.get('kind', 'single')],
                           kwargs, parents=(parent_uid,))


def effect_factory(exp, *, kind: str = 'single', **kwargs):
    """Plant synthetic effect(s) on a clean Experiment (dispatches on kind).

    A thin dispatcher (the recorded work is in the per-kind builders, so a
    record names the concrete builder -- mirrors data_factory dispatching to
    data_factory_wgn / data_factory_hcp). Every builder shares one contract,
    (exp, **kwargs) -> (exp_eff, mask_target_list): the planted Experiment and
    the list of realized supports (one entry for 'single', two for 'split'), so
    the driver threads mask_target_list into the leaf as-is.

    Args:
        exp: clean Experiment (a data_factory output) to add the effect(s) to.
        kind (str): 'single' (effect_factory_single) or 'split' (two adjacent
            effects, effect_factory_split).
        **kwargs: forwarded to the selected builder, including the parent_uid
            it requires (the clean exp's uid; see data_recipe).

    Returns:
        exp: the Experiment with the effect(s) added.
        mask_target_list (list): the realized (X, Y, Z) bool supports, one per
            planted effect (in plant order).

    Raises:
        ValueError: if kind is neither 'single' nor 'split'.
    """
    if kind not in EFFECT_FACTORY:
        raise ValueError(f"kind must be 'single' or 'split', got {kind!r}")
    return EFFECT_FACTORY[kind](exp, **kwargs)


@MEMORY.cache(ignore=['exp'])
@RECORDER(output_name_list=['exp', 'mask_target_list'], ignore=['exp'])
def effect_factory_single(exp, *, parent_uid: str, effect_llr, extenter_cls,
                          n_vox_frac=0.1):
    """Plant one synthetic effect on a clean Experiment.

    The support extenter is built here from extenter_cls + the resolved
    n_vox (n_vox_frac of the analysis volume) + a placement seed derived from
    parent_uid, so the caller passes ingredients, not a constructed Extenter.
    The whole effect is encapsulated here; the analysis crop (a separate
    extenter in data_factory) is untouched.

    The placement seed comes from the parent's declared uid, so each data
    realization plants somewhere of its own, constant across the effect_llr
    grid that shares one clean exp, and fixed by a declaration rather than by
    any array's bytes.

    Args:
        exp: clean Experiment (a data_factory output) to add the effect to.
        effect_llr (float): per-voxel (size-normalized) LLR target; the
            whole-region LLR observed is ~ effect_llr * n_vox, where n_vox is
            n_vox_frac of the analysis volume (see glow.effect.impose).
        extenter_cls (type[Extenter]): Extenter subclass sampling the support,
            built as extenter_cls(n_vox=n_vox, seed=...) (e.g. ExtenterMinVar).
        n_vox_frac (float): support size as a fraction of the analysis volume
            (num_vox = count of mask_idx > -1); resolved to
            round(n_vox_frac * num_vox).
        parent_uid (str): the clean exp's declared uid. Required: exp is
            ignored by the cache, so this is what distinguishes one clean
            experiment's planting from another's, and it seeds the
            placement.

    Returns:
        exp: the Experiment with the effect added.
        mask_target_list (list): the single realized (X, Y, Z) bool support,
            as a one-element list.
    """
    return plant_effect(exp, seed=seed_from_uid(parent_uid), kind='single',
                        effect_llr=effect_llr, extenter_cls=extenter_cls,
                        n_vox_frac=n_vox_frac)


@MEMORY.cache(ignore=['exp'])
@RECORDER(output_name_list=['exp', 'mask_target_list'], ignore=['exp'])
def effect_factory_split(exp, *, parent_uid: str, effect_llr, extenter_cls,
                         angle, n_vox_frac=0.1):
    """Plant two adjacent equal-LLR effects at a controlled direction angle.

    The cleaving setup: an extenter_cls extent of n_vox voxels (n_vox_frac of
    the analysis volume) is grown from its own seeded start and spectrally
    bisected (ExtenterSplit) into two contiguous halves, with one effect
    planted on each -- same effect_llr,
    feature directions angle degrees apart (angles 0 and angle), so the two
    effects differ only in orientation. score_effects then scores the union and
    each half in turn (target0 / target1; the merge-cost signal).

    The base extent is placed by the extenter from the seed parent_uid
    derives, as effect_factory_single seeds its own, and that one seed drives
    both the placement and the direction pair. A data-driven Fiedler cut is not
    perfectly even, so the halves may differ in size (visible in
    target0 / target1's voxel counts).

    Args:
        exp: clean Experiment (a data_factory output) to add the effects to.
        effect_llr (float): per-voxel LLR target for each effect.
        extenter_cls (type[Extenter]): Extenter subclass grown then bisected,
            built as extenter_cls(n_vox=n_vox, seed=...) (e.g. ExtenterMinVar).
        n_vox_frac (float): combined two-effect support as a fraction of the
            analysis volume (split into halves); resolved to
            round(n_vox_frac * num_vox).
        angle (float): feature-direction angle between the two effects (deg).
        parent_uid (str): the clean exp's declared uid. Required: exp is
            ignored by the cache, so this is what distinguishes one clean
            experiment's planting from another's, and it seeds the
            placement.

    Returns:
        exp: the Experiment with both effects added.
        mask_target_list (list): the two realized half-supports [mask0, mask1]
            (mask0 | mask1 is the grown extent, the two disjoint).
    """
    return plant_effect(exp, seed=seed_from_uid(parent_uid), kind='split',
                        effect_llr=effect_llr, extenter_cls=extenter_cls,
                        n_vox_frac=n_vox_frac, angle=angle)


# the dispatch tables data_factory / effect_factory select a builder from, and
# the one place a source / kind name maps to its op -- data_recipe and
# effect_recipe name a cell's uid off the same tables, so a reader and a runner
# cannot disagree about which builder a cell means.
DATA_FACTORY = {'wgn': data_factory_wgn, 'hcp': data_factory_hcp}
EFFECT_FACTORY = {'single': effect_factory_single,
                  'split': effect_factory_split}
