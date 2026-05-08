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
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


REGEN_REGISTRY: Dict[str, Callable] = {}


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
def _regen_image_paths(paths):
    from .exper import ExperimentImageOnly
    import pandas as pd
    df = pd.DataFrame.from_dict(paths, orient='index')
    return ExperimentImageOnly.from_paths(df)


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
        extra_bytes: additional bytes counted against
            ``estimated_pickle_mb`` (used by
            :meth:`AnalysisGLOW.pickle_status` to add analysis arrays).
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

    if will_slim:
        est_bytes = recipe_bytes + extra_bytes
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
