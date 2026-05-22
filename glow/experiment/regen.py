"""Recipe-based regeneration of stripped Experiment data.

When an :class:`~glow.experiment.exper.Experiment` is pickled with a
registered recipe (``exp.meta['recipe']``), its ``y`` and ``x`` arrays
are dropped from the pickle and reconstructed on demand by looking up
the recipe ``source`` in :data:`REGEN_REGISTRY` and calling the
registered function with ``recipe['args']``.

The registry sits in this module so the analysis layer never needs to
import benchmark / experiment factories.
"""

import pickle
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@contextmanager
def force_full_pickle(exp):
    """Temporarily disable slim pickling so ``pickle.dumps(exp)`` inlines
    ``y`` and ``x``.

    Use for cross-machine transfers where the receiver cannot rehydrate
    from the sender's local paths — e.g. the per-permutation AWS path
    where ``exp`` is built on the user's laptop, uploaded to S3, and
    pulled by Docker workers that don't have the same filesystem layout.

    Removes ``meta['recipe']`` for the duration of the context, so
    ``__getstate__`` falls into the "no recipe" branch (full pickle,
    silenced warning).  Restores the recipe on exit, including on
    exception paths.
    """
    meta = getattr(exp, 'meta', None)
    if meta is None or 'recipe' not in meta:
        yield exp
        return
    saved = meta.pop('recipe')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FullPickleNotice)
            yield exp
    finally:
        meta['recipe'] = saved


REGEN_REGISTRY: Dict[str, Callable] = {}
STEP_REPLAY: Dict[str, Callable] = {}


def register_step(name):
    def decorator(fn):
        STEP_REPLAY[name] = fn
        return fn
    return decorator


@register_step('apply_mask')
def _replay_apply_mask(exp, args):
    return exp.apply_mask(args['mask'])


@register_step('add_offset')
def _replay_add_offset(exp, args):
    return exp.add_offset(args['offset'], mask=args.get('mask'),
                          sigma_scale=args.get('sigma_scale'))


@register_step('bootstrap_img')
def _replay_bootstrap_img(exp, args):
    return exp.bootstrap_img(args['n'], seed=args.get('seed'),
                             noise_scale=args.get('noise_scale', 0))


@register_step('permute')
def _replay_permute(exp, args):
    return exp.permute(args['perm_idx'])


@register_step('scale')
def _replay_scale(exp, args):
    """Replay a previously-recorded ExperimentScaled.from_exp transform.

    The original ``mean_orig`` and ``pre_scale`` were captured at
    ``from_exp`` time and ride in the step args.  We construct a new
    ExperimentScaled and overwrite the freshly-derived prep with the
    stored arrays so the result is bitwise identical to the original
    (no drift from re-fitting).  Note that ``ExperimentScaled.from_exp``
    itself appends a 'scale' step; we use the lower-level constructor
    here to avoid double-appending.
    """
    from .exper import ExperimentScaled
    import numpy as np
    mean_orig = args['mean_orig']
    pre_scale = args['pre_scale']
    # Construct via __init__ then override the fitted transform with the
    # stored values to guarantee determinism even on rounding-edge inputs.
    new = ExperimentScaled(y=exp.y, x=getattr(exp, 'x', None),
                           contrast=getattr(exp, 'contrast', None),
                           mask_idx=exp.mask_idx,
                           meta=dict(exp.meta) if exp.meta else None)
    if not (np.allclose(new.mean_orig, mean_orig)
            and np.allclose(new.pre_scale, pre_scale)):
        # re-prep with the stored transform to stay bit-identical
        new.mean_orig = mean_orig
        new.pre_scale = pre_scale
        new.y = new.prep(exp.y)
    return new


class FullPickleNotice(UserWarning):
    """Emitted when an Experiment is pickled with y inline because no
    registered recipe is available."""


warnings.simplefilter('once', FullPickleNotice)


def register_regen(name):
    def decorator(fn):
        REGEN_REGISTRY[name] = fn
        return fn
    return decorator


@register_regen('gauss')
def _regen_gauss(**args):
    # 'a' presence distinguishes Experiment.from_gauss (full, with x)
    # from ExperimentImageOnly.from_gauss (image-only).
    from .exper import Experiment, ExperimentImageOnly
    if 'a' in args or 'contrast' in args:
        return Experiment.from_gauss(**args)
    return ExperimentImageOnly.from_gauss(**args)


@register_regen('image_paths')
def _regen_image_paths(paths, channel_names=None):
    from .exper import ExperimentImageOnly
    import pandas as pd
    df = pd.DataFrame.from_dict(paths, orient='index')
    return ExperimentImageOnly.from_paths(df, channel_names=channel_names)


@dataclass(frozen=True)
class PickleStatus:
    is_loaded: bool
    recipe_source: Optional[str]
    regen_registered: bool
    will_slim_on_pickle: bool
    y_mb: float
    x_mb: float
    recipe_kb: float
    estimated_pickle_mb: float

    def __str__(self):
        loaded = 'loaded' if self.is_loaded else 'slim'
        src = self.recipe_source or '(no recipe)'
        reg = 'registered' if self.regen_registered else 'unregistered'
        slim = 'slim-on-pickle' if self.will_slim_on_pickle else 'full-on-pickle'
        return (f'PickleStatus({loaded}, source={src} [{reg}], {slim}, '
                f'y={self.y_mb:.1f} MB, x={self.x_mb*1024:.1f} KB, '
                f'recipe={self.recipe_kb:.1f} KB, '
                f'est_pickle={self.estimated_pickle_mb:.1f} MB)')


def compute_pickle_status(exp, extra_bytes=0):
    """Return a :class:`PickleStatus` for ``exp``.

    Args:
        exp: an Experiment (or ExperimentImageOnly).
        extra_bytes: additional bytes to count toward the estimate.
    """
    y = getattr(exp, 'y', None)
    x = getattr(exp, 'x', None)
    is_loaded = y is not None
    y_bytes = y.nbytes if y is not None else 0
    x_bytes = x.nbytes if x is not None else 0
    y_mb = y_bytes / 1024**2
    x_mb = x_bytes / 1024**2

    meta = getattr(exp, 'meta', None) or {}
    recipe = meta.get('recipe')
    src = recipe.get('source') if recipe else None
    registered = bool(src and src in REGEN_REGISTRY)

    will_slim = is_loaded and registered

    if recipe is not None:
        recipe_bytes = len(pickle.dumps(recipe))
    else:
        recipe_bytes = 0
    recipe_kb = recipe_bytes / 1024

    # x is always preserved (no general regen interface for user-supplied
    # designs); only y is dropped on slim pickle.
    if will_slim:
        est_bytes = x_bytes + recipe_bytes + extra_bytes
    else:
        est_bytes = y_bytes + x_bytes + recipe_bytes + extra_bytes
    est_mb = est_bytes / 1024**2

    return PickleStatus(
        is_loaded=is_loaded,
        recipe_source=src,
        regen_registered=registered,
        will_slim_on_pickle=will_slim,
        y_mb=y_mb,
        x_mb=x_mb,
        recipe_kb=recipe_kb,
        estimated_pickle_mb=est_mb,
    )
