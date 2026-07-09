"""Run-bundle (de)serialization shared by the driver and the worker.

The driver ships one run bundle per submission as a pickle on S3; the worker
downloads + unpickles it and runs its array-index cell (see
glow._extra.aws.driver / glow._extra.aws.worker). The bundle is the four-tuple

    (cells, kwargs_fnc_list, fnc_ref, config_name)

-- the resolved planted cells (each a (kwargs_data, kwargs_effect) pair) and
the shared fnc-kwargs grid (params, pickled by value), the leaf fnc as an
import reference (see fnc_to_ref), and the cache label.

fnc rides as a 'module:qualname' reference, not a pickled object, for two
reasons: a memoised leaf is not picklable by reference (run_ana is wrapped by
@MEMORY.cache, so the name resolves to the cache wrapper, not the inner
function), and -- more importantly -- importing it on the worker binds it to
the worker's own MEMORY / RECORDER (the synced cache dir), whereas shipping it
by value would drag the driver's cache location across and break warm resume.
fnc is code; only the params travel.
"""

import importlib
from typing import Callable


def fnc_to_ref(fnc) -> str:
    """Return a 'module:qualname' import reference for a leaf fnc.

    The fnc must be importable by qualified name on the worker -- a top-level
    function (run_ana and its siblings are; a @MEMORY.cache wrapper of one
    still proxies __module__ / __qualname__). A lambda, a nested function, or a
    __main__-level function has no importable reference and is rejected here so
    the driver fails before submitting a doomed job.

    Args:
        fnc (Callable): the leaf measurement, e.g. run_ana.

    Returns:
        ref (str): 'module:qualname', e.g.
            'glow._extra.benchmark.run:run_ana'.

    Raises:
        ValueError: fnc has no importable module:qualname reference.
    """
    module = getattr(fnc, '__module__', None)
    qualname = getattr(fnc, '__qualname__', None)
    if not module or not qualname or module == '__main__' or '<' in qualname:
        raise ValueError(
            f'fnc {fnc!r} has no importable module:qualname reference; an '
            f'AWS leaf fnc must be a top-level function in an installed '
            f'module')
    return f'{module}:{qualname}'


def fnc_from_ref(ref: str) -> Callable:
    """Return the leaf fnc named by a 'module:qualname' import reference."""
    module, qualname = ref.split(':', 1)
    return getattr(importlib.import_module(module), qualname)
