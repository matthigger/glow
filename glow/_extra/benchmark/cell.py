"""One benchmark cell: the little that rebuilds its Experiment.

A cell is a data cell crossed with one effect. get_exp_effect realizes it
once -- load the images, crop to the analysis support, screen the constant
voxels, sample the effect support, solve its offset -- and caches only what a
rebuild needs: the source metadata, the realized masks, the design, and the
plant's constant offset. ExpEffect.build reads the images back and re-applies
the offset, which costs a small fraction of the fit it feeds, so a leaf
rebuilds its Experiment rather than unpickling one (the measurements are in
docs/notes/experiment_source_inflate.md).

The payload holds what came back; the declared kwargs beside it hold what was
asked for, and only those name the cell (exp_effect_recipe). No mask, offset
or design can reach an identity: recipe.canon raises on an array too large to
be a declaration.

A rebuild is validated rather than trusted (see ExpEffect.build): the images
must hash to what they hashed when the cell was realized, and each plant's
effect_llr must re-measure on the rebuilt data.
"""

import joblib
import numpy as np

from glow.analysis.mancova import get_llr, get_mancova
from glow.experiment import Experiment
from glow.mask import get_mask_idx

from .data import build_clean, plant_effect
from .recipe import recipe_for_call, seed_from_uid
from .store import MEMORY, RECORDER

# how close a rebuilt plant's per-voxel LLR must come to its target. Loose
# enough to survive a change of summation order (a different BLAS or thread
# count moves the measured LLR by ~1e-4 relative at the weakest effect on the
# grid, wider than the float32 gap between the target and what the solver
# realized), tight enough that every way a rebuild can go wrong -- the wrong
# mask, a mis-ordered cohort, a missing offset -- moves it by orders of
# magnitude.
LLR_RTOL = 1e-3


def y_value_hash(y) -> str:
    """Return a hash of y's values, whatever memory layout they are in.

    What a rebuild can honestly promise is the same intensities in the same
    voxel order, not the same bytes: a crop's layout is whatever numpy's
    advanced indexing hands back, which varies with the shape it was given,
    while a source loads in one fixed order. Canonicalizing to C order
    before hashing compares the images rather than their arrangement.

    Args:
        y (np.array): (b, num_img, num_vox) intensities

    Returns:
        hash (str): joblib.hash of y in C order
    """
    return joblib.hash(np.ascontiguousarray(y))


# the last cell built, as {uid: Experiment}. A cell's leaves each need the
# same Experiment, and the driver runs them back to back, so the first leaf
# builds it and its siblings read it here. One entry: the next cell drops it,
# which is what keeps a whole grid's experiments from accumulating.
_BUILD_MEMO = {}


class ExpEffect:
    """One realized benchmark cell, as the little that rebuilds it.

    Attributes:
        uid (str): the cell's declared uid (exp_effect_recipe), which its
            leaves are passed as their parent
        kwargs_data (dict): the declared data cell
        kwargs_effect (dict | None): the declared effect cell; None is the
            null (effect-free) path
        source (ImageSource): where the images are read from
        mask_ana (np.array): (X, Y, Z) boolean, the analysed voxels: the
            crop with drop_constant_vox's screen already applied
        mask_dead (np.array): (X, Y, Z) boolean, what that screen cut
        effect_list (list): one record per planted effect, in plant order,
            {mask (X, Y, Z) bool, offset (b, num_img) float} -- the support
            and the constant added to every voxel of it. Empty on the null
            path.
        x (np.array): (a, num_img) design matrix, bias row included
        contrast (np.array): (a,) boolean, True for features of interest
        y_hash (str): the clean y's value hash over mask_ana, before any
            effect was planted (see y_value_hash)
    """

    def __init__(self, *, uid: str, kwargs_data: dict, kwargs_effect,
                 source, mask_ana, mask_dead, effect_list: list, x, contrast,
                 y_hash: str):
        """Store one realized cell (see the class Attributes)."""
        self.uid = uid
        self.kwargs_data = kwargs_data
        self.kwargs_effect = kwargs_effect
        self.source = source
        self.mask_ana = mask_ana
        self.mask_dead = mask_dead
        self.effect_list = effect_list
        self.x = x
        self.contrast = contrast
        self.y_hash = y_hash

    def __repr__(self):
        """A compact identity string, which is what a record stores.

        The screen's cost rides in it (num_vox_dropped), recorded once on the
        cell it happened to rather than copied onto every leaf below.
        """
        return (f'{type(self).__name__}(uid={self.uid[:8]}, '
                f'num_vox={int(self.mask_ana.sum())}, '
                f'num_vox_dropped={int(self.mask_dead.sum())}, '
                f'num_effect={len(self.effect_list)})')

    @property
    def mask_target_list(self) -> list:
        """Return the planted supports, one (X, Y, Z) bool mask per effect."""
        return [effect['mask'] for effect in self.effect_list]

    def build(self):
        """Rebuild this cell's raw Experiment, validating what comes back.

        Reads the images over mask_ana, attaches the stored design, and adds
        each plant's offset back over its own support. The offsets go on
        through add_offset, so they land in patch_list too and an inflate can
        replay them onto voxels loaded later (a smoothing kernel's halo).

        Raw, not scaled. This is the experiment the cell realized, in the
        space the offset was solved in, and it is what every Analysis.fit
        wants: each scales on the way in (ExperimentScaled.from_exp,
        idempotent), and one of them has to split the images first, which a
        fitted transform refuses -- a tree built on one fold would have seen
        the other fold's mean and covariance.

        Returns:
            exp (Experiment): the cell's experiment, offsets applied

        Raises:
            ValueError: the images are not the ones this cell was realized
                from, or a plant does not re-measure to its effect_llr
        """
        y = self.source.load(self.mask_ana)
        got = y_value_hash(y)
        if got != self.y_hash:
            raise ValueError(
                f'{self.source!r} no longer reads the images this cell was '
                f'built from: y hashes {got}, the cell recorded '
                f'{self.y_hash}')

        exp = Experiment(y=y, mask_idx=get_mask_idx(self.mask_ana),
                         mask_dead=self.mask_dead, source=self.source,
                         meta=self.source.get_meta(),
                         x=self.x, contrast=self.contrast)
        for effect in self.effect_list:
            exp = exp.add_offset(effect['offset'], mask=effect['mask'])
        self._check_effects(exp)

        return exp

    def _check_effects(self, exp) -> None:
        """Re-measure each plant's per-voxel LLR against its target.

        The end-to-end check on a rebuild: the LLR of the planted region can
        only come out right if the images, the analysis mask, the support, the
        offset and the design are all the ones the cell was realized from. It
        runs on the raw experiment, before scaling, since that is the space
        the offset was solved in.

        Args:
            exp: the rebuilt raw Experiment, offsets applied

        Raises:
            ValueError: a plant's measured LLR is not its declared target
                (within LLR_RTOL)
        """
        if not self.effect_list:
            return
        want = self.kwargs_effect['effect_llr']
        for idx, effect in enumerate(self.effect_list):
            vox_idx = exp.mask_idx[effect['mask']]
            e, h, _ = get_mancova(x=exp.x, y=exp.y[:, :, vox_idx],
                                  contrast=exp.contrast)
            got = get_llr(e, h, n=1)
            if not np.isclose(got, want, rtol=LLR_RTOL, atol=0):
                raise ValueError(
                    f'effect {idx} of cell {self.uid[:8]} re-measures to a '
                    f'per-voxel LLR of {got:.6g}, not the {want:.6g} it was '
                    f'planted at')


def data_uid(kwargs_data) -> str:
    """Return the declared uid of a cell's data half.

    The uid of the cell with this data and no effect, which is what every
    cell sharing that data has in common. It seeds the effect placement, so a
    support lands somewhere of its own per data realization and holds across
    the effect grid above it: an llr sweep varies the strength of one plant
    rather than moving it.

    Args:
        kwargs_data (dict): one data cell.

    Returns:
        uid (str): the declared uid of that data with no effect planted.
    """
    return exp_effect_recipe(kwargs_data).uid


def exp_effect_recipe(kwargs_data, kwargs_effect=None):
    """Return the declared identity of one cell, building nothing.

    Args:
        kwargs_data (dict): one data cell, e.g. {'source': 'wgn', ...}.
        kwargs_effect (dict | None): one effect cell, or None for the null
            path.

    Returns:
        recipe (Recipe): the cell's declared identity; its uid is the
            parent_uid every leaf measuring that cell is passed.
    """
    return recipe_for_call(get_exp_effect,
                           {'kwargs_data': kwargs_data,
                            'kwargs_effect': kwargs_effect})


@MEMORY.cache
@RECORDER(output_name='cell', recurse_list=['kwargs_data', 'kwargs_effect'])
def get_exp_effect(kwargs_data, kwargs_effect=None) -> ExpEffect:
    """Realize one benchmark cell, returning the little that rebuilds it.

    The whole cost of a cell is here -- the images read, the analysis crop
    sampled, the constant voxels screened, the support grown, the offset
    solved -- and none of it is in the payload, which is why a leaf can
    rebuild cheaply and a cache of thousands of cells stays small.

    The support placement is seeded from the uid of the cell's data half
    (data_uid), so each data realization plants somewhere of its own, held
    across the effect grid above it, and fixed by a declaration rather than
    by any array's bytes.

    Both arguments are declarative, so both name the cell: nothing is
    filtered from the key. They are recursed into the record (recurse_list),
    so a flattened row carries one column per knob.

    Args:
        kwargs_data (dict): one data cell (see .data.build_clean).
        kwargs_effect (dict | None): one effect cell (see .data.plant_effect),
            or None to plant nothing (the null / FWER-calibration path).

    Returns:
        cell (ExpEffect): the realized cell (see the class Attributes).
    """
    uid = exp_effect_recipe(kwargs_data, kwargs_effect).uid
    exp = build_clean(kwargs_data)

    # the clean images as loaded, which is what a rebuild reads back and what
    # the effect was solved against
    hash_y = y_value_hash(exp.y)
    mask_ana = exp.mask_idx > -1

    if kwargs_effect is not None:
        exp, _ = plant_effect(exp, seed=seed_from_uid(data_uid(kwargs_data)),
                              **kwargs_effect)

    # patch_list is already the record of what was added, one entry per
    # plant, with each mask clipped to the voxels it reached
    effect_list = [{'mask': patch['mask'], 'offset': patch['offset']}
                   for patch in exp.patch_list]

    return ExpEffect(uid=uid, kwargs_data=kwargs_data,
                     kwargs_effect=kwargs_effect, source=exp.source,
                     mask_ana=mask_ana, mask_dead=exp.mask_dead,
                     effect_list=effect_list, x=exp.x, contrast=exp.contrast,
                     y_hash=hash_y)


def build_cell(cell: ExpEffect):
    """Return cell's raw Experiment, reusing the last one built.

    What a leaf calls instead of ExpEffect.build: a cell's leaves all measure
    one Experiment and the driver runs them back to back, so the first builds
    it and the rest read it back (see _BUILD_MEMO).

    Args:
        cell (ExpEffect): the cell to build.

    Returns:
        exp (Experiment): the cell's experiment (see ExpEffect.build for why
            it is raw).
    """
    if cell.uid not in _BUILD_MEMO:
        _BUILD_MEMO.clear()
        _BUILD_MEMO[cell.uid] = cell.build()
    return _BUILD_MEMO[cell.uid]
