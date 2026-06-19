"""Tests for glow.benchmark.data: DataSource family."""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from glow.benchmark import hcp
from glow.benchmark.data import (
    DataSource,
    DataSourceWGN,
    DataSourceDataFrame,
    DataSourceHCP,
)
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


def _stub_exp_img_only(shape=(2, 2, 2), b=1, num_img=8):
    """build an ExperimentImageOnly directly, without nifti / from_paths."""
    rng = np.random.default_rng(0)
    num_vox = int(np.prod(shape))
    y = rng.standard_normal((b, num_img, num_vox)).astype(np.float32)
    mask_idx = get_mask_idx(np.ones(shape))
    return ExperimentImageOnly(y=y, mask_idx=mask_idx)


def _patch_hcp_df(monkeypatch):
    """stub the HCP loader seams so DataSourceHCP builds with no download.

    DataSourceHCP resolves its df inline (hcp.ensure_hcp_data +
    ExperimentImageOnly._search_files); patch both to return a per-feature
    path frame whose columns are exactly the searched features, so the
    real column filtering still runs and no download / nibabel is hit.
    """
    monkeypatch.setattr(hcp, 'ensure_hcp_data', lambda: pathlib.Path('/fake'))

    def fake_search_files(folder, sbj_regex, glob_dict):
        return pd.DataFrame(
            {feat: [f'{feat}_s0.nii', f'{feat}_s1.nii'] for feat in glob_dict},
            index=['s0', 's1'])

    monkeypatch.setattr(ExperimentImageOnly, '_search_files',
                        staticmethod(fake_search_files))


# ---------------------------------------------------------------------------
# 1. Identity & hashing
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_equal_same_params(self):
        assert _wgn() == _wgn()
        assert hash(_wgn()) == hash(_wgn())

    def test_int_bool_coercion(self):
        # constructor casts; equivalent values must agree
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
        # 'a' is a BASE-class slot. Regression: HashBySlots._identity_dict
        # only walked type(self).__slots__, dropping base slots; DataSource
        # overrides it to walk the MRO so base attrs land in identity. If that
        # regresses, changing 'a' would no longer change the hash.
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
        # different DataSource subclasses with overlapping slots must
        # still be distinct (type(self) is type(other) in __eq__ + class
        # name in identity dict).
        wgn = _wgn(a=1, a_nuisance=0, has_bias=True, seed=0)
        df = DataSourceDataFrame(
            df=pd.DataFrame({'fa': ['x']}, index=['s']),
            a=1, a_nuisance=0, has_bias=True, seed=0)
        assert wgn != df
        assert hash(wgn) != hash(df)

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

    def test_unhashable_extenter_raises(self):
        class _NotHashable:
            pass

        ds = _wgn(extenter=_NotHashable())
        # not json-serialisable → hash raises (TypeError from json.dumps)
        with pytest.raises(TypeError):
            hash(ds)


# ---------------------------------------------------------------------------
# 2. _identity_dict (JSON-friendly recipe used for cache keys / YAML)
# ---------------------------------------------------------------------------

class TestIdentityDict:
    def test_contains_kind(self):
        d = _wgn()._identity_dict()
        assert d['kind'] == 'DataSourceWGN'

    def test_class_attrs_not_in_identity(self):
        # _exp_cache is a class attribute (not in __slots__) and so
        # never enters identity — locks the contract that the memo
        # cache can't perturb hashing.
        d = _wgn()._identity_dict()
        assert '_exp_cache' not in d

    def test_includes_base_slots(self):
        d = _wgn()._identity_dict()
        for s in ('a', 'a_nuisance', 'has_bias', 'seed', 'extenter'):
            assert s in d, f'base slot {s!r} missing from identity dict'

    def test_includes_subclass_slots(self):
        d = _wgn()._identity_dict()
        for s in ('shape', 'b', 'num_img'):
            assert s in d, f'subclass slot {s!r} missing'

    def test_hcp_includes_hcp_slot(self, monkeypatch):
        _patch_hcp_df(monkeypatch)
        d = DataSourceHCP()._identity_dict()
        assert d['hcp_feats'] == hcp.HCP_FEATS

    def test_json_serialisable(self):
        # WGN with no extenter is plain-types only
        ds = _wgn()
        json.dumps(ds._identity_dict(), sort_keys=True)

    def test_json_serialisable_with_extenter(self):
        ds = _wgn(extenter=ExtenterSphere(n_vox=4))
        # extenter is nested HashBySlots → _canon recurses to its dict
        json.dumps(ds._identity_dict(), sort_keys=True)

    def test_stable_across_memoization(self, small_wgn):
        before = small_wgn._identity_dict()
        _ = small_wgn.exp        # populates _exp
        after = small_wgn._identity_dict()
        assert before == after


# ---------------------------------------------------------------------------
# 3. .exp memoization + freezing
# ---------------------------------------------------------------------------

class TestExpProperty:
    def test_exp_memoised(self, small_wgn):
        assert small_wgn.exp is small_wgn.exp

    def test_get_called_once(self, monkeypatch, small_wgn):
        # __slots__ blocks instance attribute writes, so patch the class
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
        # sources resolve to the same cached Experiment (matches the
        # identity-caching intent in REFACTOR_OUTLINE.md).
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
            extenter=ExtenterSphere(n_vox=8))
        assert (ds.exp.mask_idx >= 0).sum() == 8

    def test_crop_uses_data_seed(self):
        # same seed → same cropped voxels; different seed → likely
        # different voxels (sphere is randomly placed).
        a = DataSourceWGN(shape=(5, 5, 5), b=1, num_img=8, seed=0,
                          extenter=ExtenterSphere(n_vox=8)).exp
        b = DataSourceWGN(shape=(5, 5, 5), b=1, num_img=8, seed=0,
                          extenter=ExtenterSphere(n_vox=8)).exp
        c = DataSourceWGN(shape=(5, 5, 5), b=1, num_img=8, seed=1,
                          extenter=ExtenterSphere(n_vox=8)).exp
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
        # WGN builds y via ExperimentImageOnly.from_gauss; rather than spy on
        # the call, assert the observable result reflects the kwargs:
        # y.shape == (b, num_img, prod(shape)), float32 by default, and the
        # seed actually drives the draw (different seed -> different data).
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
# 7. DataFrame subclass (concrete — accepts user df)
# ---------------------------------------------------------------------------

class TestDataSourceDataFrame:
    def _df(self, n=2, cols=('fa',)):
        return pd.DataFrame(
            {c: [f'/tmp/sbj{i}_{c}.nii' for i in range(n)] for c in cols},
            index=[f'sbj{i}' for i in range(n)])

    def test_df_required(self):
        with pytest.raises(TypeError):
            DataSourceDataFrame()           # missing required df=

    def test_df_stored(self):
        df = self._df()                      # already sorted
        ds = DataSourceDataFrame(df=df)
        pd.testing.assert_frame_equal(ds.df, df)

    def test_df_stored_sorted_by_index(self):
        # construct a df with shuffled index
        unsorted = pd.DataFrame(
            {'fa': ['/tmp/c.nii', '/tmp/a.nii', '/tmp/b.nii']},
            index=['sbj2', 'sbj0', 'sbj1'])
        ds = DataSourceDataFrame(df=unsorted)
        assert list(ds.df.index) == ['sbj0', 'sbj1', 'sbj2']

    def test_input_order_does_not_affect_identity(self):
        # same content in two different row orders → equal sources
        a = pd.DataFrame({'fa': ['x', 'y']}, index=['s1', 's0'])
        b = pd.DataFrame({'fa': ['y', 'x']}, index=['s0', 's1'])
        assert DataSourceDataFrame(df=a) == DataSourceDataFrame(df=b)
        assert hash(DataSourceDataFrame(df=a)) \
            == hash(DataSourceDataFrame(df=b))

    def test_df_in_identity(self):
        a = DataSourceDataFrame(df=self._df())
        b = DataSourceDataFrame(df=self._df())
        assert a == b
        assert hash(a) == hash(b)

    def test_different_df_different_hash(self):
        a = DataSourceDataFrame(df=self._df(n=2))
        b = DataSourceDataFrame(df=self._df(n=3))
        assert a != b
        assert hash(a) != hash(b)

    def test_get_routes_through_from_paths(self, monkeypatch):
        # from_paths reads real nifti, so it must be stubbed; but make the
        # stub's size track the df it receives (one image per subject row) and
        # assert the *produced* exp reflects that, rather than spying on args.
        def fake_from_paths(cls, paths, **kw):
            return _stub_exp_img_only(num_img=len(paths))

        monkeypatch.setattr(ExperimentImageOnly, 'from_paths',
                            classmethod(fake_from_paths))
        df = self._df(n=3)
        exp = DataSourceDataFrame(df=df).exp
        # y is (b, num_img, num_vox); num_img == subject count routed through
        assert exp.y.shape[1] == 3


# ---------------------------------------------------------------------------
# 8. HCP subclass
# ---------------------------------------------------------------------------

class TestDataSourceHCP:
    """HCP fetches the df eagerly in __init__; all tests stub hcp."""

    @pytest.fixture(autouse=True)
    def _hcp_df(self, monkeypatch):
        _patch_hcp_df(monkeypatch)

    def test_hcp_feats_default(self):
        assert DataSourceHCP().hcp_feats == hcp.HCP_FEATS

    def test_hcp_feats_tuple_coercion(self):
        assert DataSourceHCP(hcp_feats=['fa']).hcp_feats == ('fa',)

    def test_hcp_feats_in_identity(self):
        a = DataSourceHCP(hcp_feats=('fa',))
        b = DataSourceHCP(hcp_feats=('fa', 'md'))
        assert a != b
        assert hash(a) != hash(b)

    def test_init_fetches_filtered_df(self):
        assert list(DataSourceHCP(hcp_feats=('fa',)).df.columns) == ['fa']
        assert list(DataSourceHCP(hcp_feats=('fa', 'md')).df.columns) \
            == ['fa', 'md']
