"""bundle: the leaf fnc rides as an importable module:qualname reference."""

import pytest

from glow._extra.aws.bundle import fnc_from_ref, fnc_to_ref
from glow._extra.benchmark.run import run_ana


def test_fnc_ref_round_trips_to_same_object():
    ref = fnc_to_ref(run_ana)
    assert ref == 'glow._extra.benchmark.run:run_ana'
    # resolving on the worker yields the module's own (memoised) fnc
    assert fnc_from_ref(ref) is run_ana


def test_fnc_to_ref_rejects_unimportable():
    # a lambda has no importable reference -> rejected on the driver
    with pytest.raises(ValueError):
        fnc_to_ref(lambda exp, **kw: None)
