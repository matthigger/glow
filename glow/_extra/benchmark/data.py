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
cache, so a cache hit returns the stored result and only a real (cache-miss)
build is recorded. The recorder keys each build by joblib's own args hash (see
Recorder), so a record lines up one-to-one with the cached artifact on disk.
All builders share MEMORY / RECORDER, so a clean build and the plant that
consumes it link into one provenance DAG.
"""
import joblib
import numpy as np

from glow.effect import EffectSynthetic, Extenter, ExtenterSplit
from glow.experiment import Experiment, ExperimentImageOnly

from . import hcp
from .file import get_path_cache, get_path_records
from .recipe import recipe_for_call, seed_from_uid
from .recorder import Recorder

# disk memoisation of the experiment builds, keyed on the build inputs, so a
# repeated (source, ...) cell is loaded rather than rebuilt across runs.
MEMORY = joblib.Memory(get_path_cache(), verbose=0)

# captures each build's inputs / output / timing for provenance (see Recorder).
# Keyed by joblib's args hash, mirrored to the records dir beside the cache.
# link_types=(Experiment,) makes flatten_to_df draw an edge wherever an
# Experiment produced by one build is consumed by another (data_factory's clean
# exp -> effect_factory's plant); scalars / Extenters / masks never link, so
# trivial values forge no spurious edges (see Recorder).
RECORDER = Recorder(folder=get_path_records(), link_types=(Experiment,))


def _with_canonical_y(exp):
    """Return exp with an owning, canonically-strided (F-contiguous) y.

    Both the joblib.Memory cache key and the provenance link (the
    RECORDER's input / output content hashes) are joblib.hash of the whole
    Experiment, which folds in y's memory layout -- not just its bytes.
    apply_mask crops y to a non-owning view, and at b=1 the length-1 feature
    axis carries an ambiguous stride that pickling does not preserve, so a
    freshly built exp and the same exp reloaded from the disk cache hash
    differently (their strides differ though their bytes do not). That
    silently breaks the cache (a warm-cache / resumed run misses and
    recomputes) and orphans the leaf in flatten_to_df (its exp-input hash no
    longer matches the recorded build's output hash) the moment a build is
    served from cache.

    A fresh copy gives y owning storage with canonical strides whose hash
    survives the pickle round-trip. We copy in Fortran order (not the usual
    C): that is the layout from_gauss / the HCP loader produce and that the
    scaling path is written for (ExperimentScaled.prep's einsum, the
    order='F' cov reshape), so the hash is fixed without flipping the layout
    the analysis hot loop expects. asfortranarray would not do: the
    degenerate view is already flagged F-contiguous, so it would return it
    unchanged and fix nothing.
    """
    exp.y = exp.y.copy(order='F')
    return exp


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
    exp = exp.drop_constant_vox()
    # canonicalise y's layout so the cached/recorded exp hashes stably (see
    # helper)
    return _with_canonical_y(exp)


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
    # the single build path that also works on an AWS worker, where the bundle
    # is pre-staged from S3 (no niftis, no DUA prompt) -- see hcp.py / the aws
    # package. The archive's brain mask is the analysis support (NODDI isovf is
    # legitimately zero in-brain, so the maps can't infer it).
    exp_img = hcp.build_exp_img_from_bundle(hcp_feats)
    return _sample_x_and_crop(exp_img, a=a, contrast=contrast,
                              has_bias=has_bias, extenter=extenter, seed=seed)


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


def _resolve_seed(exp, seed, seed_from_exp: bool, seed_from_parent: bool,
                  parent_uid) -> int:
    """Resolve a placement seed from exactly one of the three seed sources.

    All three answer the same need: the driver shares one effect grid across
    every data cell, so a single fixed seed would plant identically everywhere.
    A derived seed gives each data realization its own placement, constant
    across the effect_llr grid (which shares one clean exp).

    seed_from_parent derives it from the parent's uid -- a declaration, so the
    placement cannot drift with a BLAS kernel. seed_from_exp derives it from
    joblib.hash(exp) instead, which ties the planted support to the clean exp's
    bytes; it reproduces cells planted that way, so prefer seed_from_parent for
    new work (see glow._extra.benchmark.recipe).

    Args:
        exp: the clean Experiment the effect is planted on.
        seed (int | None): explicit placement seed.
        seed_from_exp (bool): derive the seed from a hash of exp's bytes.
        seed_from_parent (bool): derive the seed from parent_uid.
        parent_uid (str | None): the clean exp's declared uid; required by
            seed_from_parent.

    Returns:
        the resolved placement seed.

    Raises:
        ValueError: unless exactly one source is given, or seed_from_parent
            without a parent_uid.
    """
    given = [seed is not None, bool(seed_from_exp), bool(seed_from_parent)]
    if sum(given) != 1:
        raise ValueError('pass exactly one of seed / seed_from_exp / '
                         'seed_from_parent')
    if seed_from_parent:
        if not parent_uid:
            raise ValueError('seed_from_parent needs a parent_uid')
        return seed_from_uid(parent_uid)
    if seed_from_exp:
        return int(joblib.hash(exp), 16)
    return seed


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
                          n_vox_frac=0.1, seed: int = None,
                          seed_from_exp: bool = False,
                          seed_from_parent: bool = False):
    """Plant one synthetic effect on a clean Experiment.

    The support extenter is built here from extenter_cls + the resolved n_vox
    (n_vox_frac of the analysis volume) + a placement seed (seed XOR
    seed_from_exp; see _resolve_seed), so the caller passes ingredients, not a
    constructed Extenter. The whole effect is encapsulated here; the analysis
    crop (a separate extenter in data_factory) is untouched.

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
        seed (int): support placement seed; pass exactly one seed source.
        seed_from_exp (bool): derive the seed from a hash of exp's bytes.
        seed_from_parent (bool): derive the seed from parent_uid (preferred;
            see _resolve_seed).
        parent_uid (str): the clean exp's declared uid. Required: exp is
            ignored by the cache, so this is what distinguishes one clean
            experiment's planting from another's.

    Returns:
        exp: the Experiment with the effect added.
        mask_target_list (list): the single realized (X, Y, Z) bool support,
            as a one-element list.

    Raises:
        ValueError: unless exactly one seed source is given.
    """
    seed = _resolve_seed(exp, seed, seed_from_exp, seed_from_parent,
                         parent_uid)
    n_vox = round(n_vox_frac * int((exp.mask_idx > -1).sum()))
    extenter = extenter_cls(n_vox=n_vox, seed=seed)
    exp, mask = EffectSynthetic(
        extenter=extenter, effect_llr=effect_llr).fit(exp)
    # canonicalise y's layout so the planted exp hashes stably (see
    # _with_canonical_y)
    return _with_canonical_y(exp), [mask]


@MEMORY.cache(ignore=['exp'])
@RECORDER(output_name_list=['exp', 'mask_target_list'], ignore=['exp'])
def effect_factory_split(exp, *, parent_uid: str, effect_llr, extenter_cls,
                         angle, n_vox_frac=0.1, seed: int = None,
                         seed_from_exp: bool = False,
                         seed_from_parent: bool = False):
    """Plant two adjacent equal-LLR effects at a controlled direction angle.

    The cleaving setup: an extenter_cls extent of n_vox voxels (n_vox_frac of
    the analysis volume) is grown from its own seeded start and spectrally
    bisected (ExtenterSplit) into two contiguous halves, with one effect
    planted on each -- same effect_llr,
    feature directions angle degrees apart (angles 0 and angle), so the two
    effects differ only in orientation. score_effects then scores the union and
    each half in turn (target0 / target1; the merge-cost signal).

    The base extent is placed by the extenter from the resolved seed (as
    effect_factory_single seeds its extenter), so it picks its own start --
    ExtenterMinVar (the cleaving cache's base) grows the lowest-variance region
    and bisects it into roughly equal halves, the same data-driven support the
    single-effect caches use. The one resolved seed (seed XOR seed_from_exp)
    drives both the placement and the direction pair. A data-driven Fiedler cut
    is not perfectly even, so the halves may differ in size (visible in
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
        seed (int): placement + direction seed; pass exactly one seed source.
        seed_from_exp (bool): derive the seed from a hash of exp's bytes.
        seed_from_parent (bool): derive the seed from parent_uid (preferred;
            see _resolve_seed).
        parent_uid (str): the clean exp's declared uid. Required: exp is
            ignored by the cache, so this is what distinguishes one clean
            experiment's planting from another's.

    Returns:
        exp: the Experiment with both effects added.
        mask_target_list (list): the two realized half-supports [mask0, mask1]
            (mask0 | mask1 is the grown extent, the two disjoint).

    Raises:
        ValueError: unless exactly one seed source is given.
    """
    seed = _resolve_seed(exp, seed, seed_from_exp, seed_from_parent,
                         parent_uid)
    n_vox = round(n_vox_frac * int((exp.mask_idx > -1).sum()))
    splitter = ExtenterSplit(base=extenter_cls(n_vox=n_vox, seed=seed))
    mask0, mask1 = splitter.fit(mask_idx=exp.mask_idx, y=exp.y)
    e0 = EffectSynthetic(mask=mask0, effect_llr=effect_llr, angle=0.0,
                         seed=seed)
    e1 = EffectSynthetic(mask=mask1, effect_llr=effect_llr, angle=float(angle),
                         seed=seed)
    exp = e1.fit(e0.fit(exp)[0])[0]
    # canonicalise y's layout so the planted exp hashes stably (see
    # _with_canonical_y)
    return _with_canonical_y(exp), [mask0, mask1]


# the dispatch tables data_factory / effect_factory select a builder from, and
# the one place a source / kind name maps to its op -- data_recipe and
# effect_recipe name a cell's uid off the same tables, so a reader and a runner
# cannot disagree about which builder a cell means.
DATA_FACTORY = {'wgn': data_factory_wgn, 'hcp': data_factory_hcp}
EFFECT_FACTORY = {'single': effect_factory_single,
                  'split': effect_factory_split}
