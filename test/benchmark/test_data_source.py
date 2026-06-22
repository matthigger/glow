"""Tests for glow.benchmark.data: DataSource family."""

import json

import numpy as np
import pandas as pd
import pytest

from glow.benchmark import hcp
from glow.benchmark.data import DataSource, DataSourceWGN, DataSourceHCP
from glow.effect import ExtenterSphere
from glow.experiment.exper import ExperimentImageOnly, NoBiasTermWarning
from glow.mask import get_mask_idx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _wgn(**kw):
    """cheap WGN source — shape (2,2,2), b=1, num_img=8."""
    defaults = dict(shape=(2, 2, 2), b=1, num_img=8, seed=0)
    defaults.update(kw)
    return DataSourceWGN(**defaults)


@pytest.fixture(autouse=True)
def _clear_exp_cache():
    """DataSource._exp_cache is class-level (shared by design); clear
    between tests so cross-test state doesn't bleed into spies / counts."""
    DataSource._exp_cache.clear()
    yield
    DataSource._exp_cache.clear()


@pytest.fixture
def small_wgn():
    return _wgn()


# ---------------------------------------------------------------------------
# 1. Identity & hashing  (native frozen-dataclass __hash__ / __eq__)
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_equal_same_params(self):
        assert _wgn() == _wgn()
        assert hash(_wgn()) == hash(_wgn())

    def test_int_bool_coercion(self):
        # __post_init__ casts; equivalent values must agree
        a = DataSourceWGN(a=2, has_bias=True, seed=0,
                          shape=(2, 2, 2), b=1, num_img=8)
        b = DataSourceWGN(a=2.0, has_bias=1, seed=0.0,
                          shape=(2, 2, 2), b=1, num_img=8)
        assert a == b
        assert hash(a) == hash(b)

    def test_shape_tuple_coercion(self):
        a = DataSourceWGN(shape=(2, 2, 2), b=1, num_img=8, seed=0)
        b = DataSourceWGN(shape=[2, 2, 2], b=1, num_img=8, seed=0)
        assert a == b
        assert hash(a) == hash(b)

    @pytest.mark.parametrize('attr, value', [
        # 'a'/'a_nuisance'/'has_bias'/'seed'/'extenter' are base-class fields;
        # native __hash__ / __eq__ cover inherited fields, so each changes
        # identity.
        ('a', 2),
        ('a_nuisance', 1),
        ('has_bias', False),
        ('seed', 1),
        ('extenter', ExtenterSphere(n_vox=4)),
        ('shape', (3, 2, 2)),
        ('b', 2),
        ('num_img', 16),
    ])
    def test_each_attribute_changes_hash(self, attr, value):
        a = _wgn()
        b = _wgn(**{attr: value})
        assert a != b, f'{attr} should affect identity'
        assert hash(a) != hash(b), f'{attr} should affect hash'

    def test_cross_subclass_inequality(self):
        # different DataSource subclasses are distinct (dataclass __eq__
        # checks the class), and their stable ids (to_json, folds in 'kind')
        # differ too. HCP construction is pure, so no stub is needed.
        wgn = _wgn(a=1, a_nuisance=0, has_bias=True, seed=0)
        hcp_ds = DataSourceHCP(hcp_feats=('fa',),
                               a=1, a_nuisance=0, has_bias=True, seed=0)
        assert wgn != hcp_ds
        assert wgn.to_json() != hcp_ds.to_json()

    def test_usable_as_collection_key(self):
        # equal-but-distinct sources collapse as dict keys and set members
        assert {_wgn(): 'x'}[_wgn()] == 'x'
        assert len({_wgn(), _wgn()}) == 1

    def test_extenter_in_identity(self):
        a = _wgn(extenter=ExtenterSphere(n_vox=4))
        b = _wgn(extenter=ExtenterSphere(n_vox=4))
        c = _wgn(extenter=ExtenterSphere(n_vox=5))
        assert a == b and hash(a) == hash(b)
        assert a != c and hash(a) != hash(c)

    def test_non_serialisable_extenter_to_json_raises(self):
        # an extenter that isn't JSON-serialisable can't produce a stable id
        class _NotSerialisable:
            pass

        ds = _wgn(extenter=_NotSerialisable())
        with pytest.raises(TypeError):
            ds.to_json()


# ---------------------------------------------------------------------------
# 2. to_dict / to_json (JSON-friendly recipe used for stable cross-process id)
# ---------------------------------------------------------------------------

class TestToDict:
    def test_contains_kind(self):
        assert _wgn().to_dict()['kind'] == 'DataSourceWGN'

    def test_class_attrs_not_in_identity(self):
        # _exp_cache is a class attribute (not a field) and so never enters
        # identity — locks the contract that the memo cache can't perturb it.
        assert '_exp_cache' not in _wgn().to_dict()

    def test_includes_base_fields(self):
        d = _wgn().to_dict()
        for s in ('a', 'a_nuisance', 'has_bias', 'seed', 'extenter'):
            assert s in d, f'base field {s!r} missing from to_dict'

    def test_includes_subclass_fields(self):
        d = _wgn().to_dict()
        for s in ('shape', 'b', 'num_img'):
            assert s in d, f'subclass field {s!r} missing'

    def test_hcp_includes_hcp_field(self):
        # to_dict touches no disk, so DataSourceHCP needs no stub
        assert DataSourceHCP().to_dict()['hcp_feats'] == hcp.HCP_FEATS

    def test_json_serialisable(self):
        # WGN with no extenter is plain-types only
        json.loads(_wgn().to_json())

    def test_json_serialisable_with_extenter(self):
        # extenter is a nested DataclassJSON → to_dict recurses to its dict
        ds = _wgn(extenter=ExtenterSphere(n_vox=4))
        d = json.loads(ds.to_json())
        assert d['extenter']['kind'] == 'ExtenterSphere'

    def test_stable_across_memoization(self, small_wgn):
        before = small_wgn.to_dict()
        _ = small_wgn.exp        # populates the memo cache
        assert small_wgn.to_dict() == before


# ---------------------------------------------------------------------------
# 3. .exp memoization + freezing
# ---------------------------------------------------------------------------

class TestExpProperty:
    def test_exp_memoised(self, small_wgn):
        assert small_wgn.exp is small_wgn.exp

    def test_get_called_once(self, monkeypatch, small_wgn):
        # frozen+slots blocks instance attribute writes, so patch the class
        calls = []
        real_get = DataSourceWGN._get

        def spy(self):
            calls.append(1)
            return real_get(self)

        monkeypatch.setattr(DataSourceWGN, '_get', spy)
        for _ in range(3):
            small_wgn.exp
        assert len(calls) == 1

    def test_exp_ndarrays_frozen(self, small_wgn):
        exp = small_wgn.exp
        for name, val in exp.__dict__.items():
            if isinstance(val, np.ndarray):
                assert val.flags.writeable is False, f'{name} not frozen'
                with pytest.raises(ValueError):
                    val[...] = 0

    def test_exp_non_array_attrs_unchanged(self, small_wgn):
        # meta survives (still a dict, contents not coerced)
        assert isinstance(small_wgn.exp.meta, dict)

    def test_hash_stable_across_memoization(self, small_wgn):
        h0 = hash(small_wgn)
        small_wgn.exp
        assert hash(small_wgn) == h0

    def test_two_equal_sources_share_cache(self):
        # cache is keyed by `self` via hash+eq, so equal-but-distinct
        # sources resolve to the same cached Experiment.
        a = _wgn()
        b = _wgn()
        assert a == b
        assert a.exp is b.exp


# ---------------------------------------------------------------------------
# 4. Shared X construction (_sample_x_and_crop)
# ---------------------------------------------------------------------------

class TestSampleXShared:
    def test_x_shape_default(self):
        ds = _wgn()                          # a=1, a_nuisance=0, has_bias=True
        assert ds.exp.x.shape == (2, 8)      # 1 bias + 0 nuis + 1 interest

    def test_x_shape_no_bias(self):
        ds = _wgn(a=1, a_nuisance=0, has_bias=False)
        # no all-ones row -> ExperimentImageOnly warns (expected here)
        with pytest.warns(NoBiasTermWarning):
            assert ds.exp.x.shape == (1, 8)

    def test_x_shape_with_nuisance(self):
        ds = _wgn(a=2, a_nuisance=3, has_bias=True)
        assert ds.exp.x.shape == (1 + 3 + 2, 8)

    def test_contrast_layout(self):
        ds = _wgn(a=2, a_nuisance=3, has_bias=True)
        contrast = ds.exp.contrast
        # first 1 + 3 are False (bias + nuisance); last 2 are True (interest)
        assert contrast.dtype == bool
        np.testing.assert_array_equal(
            contrast, [False] * 4 + [True] * 2)

    def test_x_seeded_reproducible(self):
        a = _wgn(seed=0).exp.x
        b = _wgn(seed=0).exp.x
        np.testing.assert_array_equal(a, b)

    def test_x_seed_differs(self):
        a = _wgn(seed=0).exp.x
        b = _wgn(seed=1).exp.x
        assert not np.array_equal(a, b)

    def test_x_dtype_matches_y(self, small_wgn):
        assert small_wgn.exp.x.dtype == small_wgn.exp.y.dtype


# ---------------------------------------------------------------------------
# 5. Optional crop via `extenter`
# ---------------------------------------------------------------------------

class TestExtenterCrop:
    def test_no_extenter_full_volume(self):
        ds = DataSourceWGN(shape=(3, 3, 3), b=1, num_img=8, seed=0)
        assert (ds.exp.mask_idx >= 0).sum() == 27

    def test_extenter_crops(self):
        ds = DataSourceWGN(
            shape=(5, 5, 5), b=1, num_img=8, seed=0,
            extenter=ExtenterSphere(n_vox=8, seed=0))
        assert (ds.exp.mask_idx >= 0).sum() == 8

    def test_crop_uses_extenter_seed(self):
        # the crop seed lives in the extenter now (not the DataSource seed):
        # same extenter -> same cropped voxels; a differently-seeded extenter
        # -> different voxels (sphere is randomly placed).
        ext0 = ExtenterSphere(n_vox=8, seed=0)
        ext1 = ExtenterSphere(n_vox=8, seed=1)
        a = DataSourceWGN(shape=(5, 5, 5), b=1, num_img=8, seed=0,
                          extenter=ext0).exp
        b = DataSourceWGN(shape=(5, 5, 5), b=1, num_img=8, seed=0,
                          extenter=ext0).exp
        c = DataSourceWGN(shape=(5, 5, 5), b=1, num_img=8, seed=0,
                          extenter=ext1).exp
        np.testing.assert_array_equal(a.mask_idx >= 0, b.mask_idx >= 0)
        assert not np.array_equal(a.mask_idx >= 0, c.mask_idx >= 0)


# ---------------------------------------------------------------------------
# 6. WGN subclass
# ---------------------------------------------------------------------------

class TestDataSourceWGN:
    def test_default_kwargs(self):
        ds = DataSourceWGN()
        assert ds.shape == (5, 5, 5)
        assert ds.b == 2
        assert ds.num_img == 100

    def test_get_uses_from_gauss(self):
        # WGN builds y via ExperimentImageOnly.from_gauss; assert the
        # observable result reflects the kwargs.
        exp = _wgn(seed=42, b=3, num_img=4, shape=(2, 2, 2)).exp
        assert exp.y.shape == (3, 4, 8)        # b, num_img, 2*2*2 voxels
        assert exp.y.dtype == np.float32
        other = _wgn(seed=43, b=3, num_img=4, shape=(2, 2, 2)).exp
        assert not np.array_equal(exp.y, other.y)

    def test_y_shape(self, small_wgn):
        assert small_wgn.exp.y.shape == (1, 8, 8)   # b, num_img, num_vox

    def test_y_reproducible(self):
        a = _wgn(seed=0).exp.y
        b = _wgn(seed=0).exp.y
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 7. HCP subclass
# ---------------------------------------------------------------------------

class TestDataSourceHCP:
    """Identity is the feature subset; the df is derived in _get (no field
    holds a DataFrame), so construction touches no disk."""

    def test_hcp_feats_default(self):
        assert DataSourceHCP().hcp_feats == hcp.HCP_FEATS

    def test_hcp_feats_tuple_coercion(self):
        assert DataSourceHCP(hcp_feats=['fa']).hcp_feats == ('fa',)

    def test_hcp_feats_in_identity(self):
        a = DataSourceHCP(hcp_feats=('fa',))
        b = DataSourceHCP(hcp_feats=('fa', 'md'))
        assert a != b
        assert hash(a) != hash(b)

    def test_construction_is_pure(self):
        # df is derived in _get, not stored as a field
        ds = DataSourceHCP(hcp_feats=('fa',))
        assert not hasattr(ds, 'df')
        assert 'df' not in ds.to_dict()
