"""Tests for declared-recipe identity (glow._extra.benchmark.recipe).

The portability tests are the regression guard for the failure this module
replaces: an identity derived from array bytes differs between two CPUs that
compute the same experiment, so a leaf computed on one machine could never link
to an ancestor built on another.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from glow._extra.benchmark import recipe
from glow._extra.benchmark.recipe import (ComputedArrayError, Recipe, canon,
                                          canon_json, recipe_id, seed_from_uid)
from glow.effect.extent import ExtenterMinVar, ExtenterSphere


def test_recipe_id_is_deterministic_and_key_order_independent():
    """Same declaration in any dict order gives one uid."""
    a = recipe_id('run_stat', {'stat_name': 'wilks', 'ana': 'VBA'})
    b = recipe_id('run_stat', {'ana': 'VBA', 'stat_name': 'wilks'})
    assert a == b == recipe_id('run_stat', {'stat_name': 'wilks',
                                            'ana': 'VBA'})
    assert len(a) == recipe.UID_LEN
    assert set(a) <= set('0123456789abcdef')


def test_recipe_id_varies_with_every_field():
    """op, kwargs, parents (incl. order) and impl_version each change the uid."""
    base = recipe_id('run_stat', {'stat_name': 'wilks'}, parents=['p0'])
    assert base != recipe_id('run_ana', {'stat_name': 'wilks'},
                             parents=['p0'])
    assert base != recipe_id('run_stat', {'stat_name': 'pillai'},
                             parents=['p0'])
    assert base != recipe_id('run_stat', {'stat_name': 'wilks'},
                             parents=['p1'])
    assert base != recipe_id('run_stat', {'stat_name': 'wilks'},
                             parents=['p0', 'p1'])
    assert (recipe_id('op', {}, parents=['a', 'b'])
            != recipe_id('op', {}, parents=['b', 'a']))
    assert base != recipe_id('run_stat', {'stat_name': 'wilks'},
                             parents=['p0'], impl_version=99)


def test_recipe_id_separates_near_equal_floats():
    """A float kwarg round-trips exactly, so a swept grid step stays distinct."""
    assert (recipe_id('e', {'effect_llr': 0.03})
            != recipe_id('e', {'effect_llr': 0.030000000000000013}))


def test_canon_renders_class_and_parameter_only_object():
    """A class canonicalises by import path, an Extenter by its repr."""
    assert (canon(ExtenterMinVar)
            == 'glow.effect.extent.ExtenterMinVar')
    extenter = ExtenterSphere(seed=36, n_vox=25000)
    assert canon(extenter) == repr(extenter)
    assert 'seed=36' in canon(extenter)


def test_canon_prefers_canon_spec():
    """An object defining canon_spec() declares itself through it."""

    class Declared:
        """A stand-in for an op parameter that declares its own spec."""

        def canon_spec(self):
            """Return the declaration."""
            return {'kind': 'declared', 'n': 3}

    assert canon(Declared()) == {'kind': 'declared', 'n': 3}


def test_canon_containers_recurse_and_sets_are_order_free():
    """Tuples render as lists, sets by sorted content, dicts key-sorted."""
    assert canon((1, (2, 3))) == [1, [2, 3]]
    assert canon({'b': 1, 'a': 2}) == {'a': 2, 'b': 1}
    assert canon({'x', 'y'}) == canon({'y', 'x'})
    assert canon(np.float32(1.5)) == 1.5
    assert canon(np.int64(7)) == 7


def test_canon_small_array_is_structural():
    """A declarative array (a contrast) renders as nested lists, not a hash."""
    assert canon(np.array([True, False])) == [True, False]
    assert canon(np.zeros((2, 2))) == [[0.0, 0.0], [0.0, 0.0]]


def test_canon_rejects_a_computed_array():
    """An array beyond MAX_CANON_SIZE raises rather than keying on bytes."""
    big = np.zeros(recipe.MAX_CANON_SIZE + 1)
    with pytest.raises(ComputedArrayError, match='computed array bytes'):
        canon(big)
    with pytest.raises(ComputedArrayError):
        recipe_id('run_stat', {'y': big})


def test_recipe_id_never_hashes_an_array(no_array_hashing):
    """Building a uid touches joblib's array hasher not at all."""
    uid = recipe_id('data_factory_wgn',
                    {'shape': (30, 30, 30), 'contrast': np.array([True]),
                     'extenter': ExtenterSphere(seed=1, n_vox=10)})
    assert len(uid) == recipe.UID_LEN


def test_recipe_id_is_portable_across_blas_kernel_and_hash_seed():
    """A uid is identical under a different BLAS kernel and hash seed.

    The exact regression: the planted Experiment's bytes (and so its
    joblib.hash) change with the BLAS microkernel, which is why identity may not
    come from them.
    """
    kwargs = {'stat_name': 'wilks', 'n': 3, 'flag': True, 'x': 0.125}
    expected = recipe_id('run_stat', kwargs, parents=['abc'])
    code = (
        'import json, sys;'
        'from glow._extra.benchmark.recipe import recipe_id;'
        'kwargs = json.loads(sys.argv[1]);'
        "print(recipe_id('run_stat', kwargs, parents=['abc']))"
    )
    env = dict(os.environ, OPENBLAS_CORETYPE='Nehalem', PYTHONHASHSEED='1')
    out = subprocess.run([sys.executable, '-c', code, json.dumps(kwargs)],
                         capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == expected


def test_canon_json_is_diffable():
    """canon_json exposes the fields an id is built from."""
    text = canon_json('run_stat', {'stat_name': 'wilks'}, parents=['p'])
    payload = json.loads(text)
    assert payload['op'] == 'run_stat'
    assert payload['kwargs'] == {'stat_name': 'wilks'}
    assert payload['parents'] == ['p']
    assert payload['impl_version'] == recipe.IMPL_VERSION['run_stat']
    assert payload['canon_version'] == recipe.CANON_VERSION


def test_impl_version_bump_invalidates_ids(monkeypatch):
    """Bumping an op's IMPL_VERSION changes every id built from it."""
    before = recipe_id('run_stat', {'stat_name': 'wilks'})
    monkeypatch.setitem(recipe.IMPL_VERSION, 'run_stat',
                        recipe.IMPL_VERSION['run_stat'] + 1)
    assert recipe_id('run_stat', {'stat_name': 'wilks'}) != before


def test_recipe_chains_parents_and_reports_its_record_form():
    """Recipe.child threads the parent uid; as_dict is the record payload."""
    data = Recipe('data_factory_wgn', {'seed': 36})
    effect = data.child('effect_factory_single', {'effect_llr': 0.03})
    leaf = effect.child('run_stat', {'stat_name': 'wilks'})
    assert effect.parents == (data.uid,)
    assert leaf.parents == (effect.uid,)
    assert leaf.uid == recipe_id('run_stat', {'stat_name': 'wilks'},
                                 parents=[effect.uid])
    as_dict = effect.as_dict()
    assert as_dict['uid'] == effect.uid
    assert as_dict['op'] == 'effect_factory_single'
    assert as_dict['kwargs'] == {'effect_llr': 0.03}
    assert as_dict['parents'] == [data.uid]
    assert as_dict['impl_version'] == recipe.IMPL_VERSION[
        'effect_factory_single']
    assert Recipe('op', {'a': 1}) == Recipe('op', {'a': 1})
    assert len({Recipe('op', {'a': 1}), Recipe('op', {'a': 1})}) == 1


def test_seed_from_uid_is_stable_and_bounded():
    """A uid-derived seed is reproducible, in range, and uid-specific."""
    assert seed_from_uid('abc') == seed_from_uid('abc')
    assert seed_from_uid('abc') != seed_from_uid('abd')
    assert 0 <= seed_from_uid('abc') < 2 ** 32
    assert 0 <= seed_from_uid('abc', bits=8) < 2 ** 8
