"""Tests for the min_vox size gate (scatter + lookup dropdown + resolution)."""

import sys

import numpy as np
import pytest

from glow.viewer.app import (_create_app, _display_region_ids,
                             _resolve_min_vox, _suggest_min_vox)
from glow.viewer.scatter import build_scatter


def _main_marker_trace(fig):
    """Return the clickable region scatter trace (not the target star)."""
    main = [t for t in fig.data
            if t.mode == 'markers' and t.showlegend is False
            and getattr(t, 'customdata', None) is not None
            and 'target' not in list(t.customdata)]
    assert len(main) == 1
    return main[0]


def _find_component(component, target_id, depth=20):
    """Recursively search a Dash layout for a component by its id."""
    if depth <= 0:
        return None
    if getattr(component, 'id', None) == target_id:
        return component
    children = getattr(component, 'children', None)
    if children is None:
        return None
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is None:
            continue
        found = _find_component(child, target_id, depth - 1)
        if found is not None:
            return found
    return None


class TestBuildScatterMinVox:
    """min_vox drops the smaller regions from the scattered set."""

    def test_zero_scatters_every_region(self, df_with_target, ana):
        num_reg = len(df_with_target)
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr',
                            '__none__', min_vox=0)
        assert len(_main_marker_trace(fig).customdata) == num_reg

    def test_cut_excludes_small_regions(self, df_with_target, ana):
        min_vox = 3
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr',
                            '__none__', min_vox=min_vox)
        shown = list(_main_marker_trace(fig).customdata)
        size_by_reg = dict(zip(df_with_target['region_idx'],
                               df_with_target['n_voxel']))
        assert all(size_by_reg[r] >= min_vox for r in shown)
        # exactly the regions at or above the cut are shown
        expected = int((df_with_target['n_voxel'] >= min_vox).sum())
        assert len(shown) == expected

    def test_cut_reduces_point_count(self, df_with_target, ana):
        n_all = len(_main_marker_trace(
            build_scatter(df_with_target, ana, 'n_voxel', 'llr',
                          '__none__', min_vox=0)).customdata)
        n_cut = len(_main_marker_trace(
            build_scatter(df_with_target, ana, 'n_voxel', 'llr',
                          '__none__', min_vox=2)).customdata)
        # the tree has many size-1 leaves, so a cut at 2 must shrink it
        assert n_cut < n_all


class TestSuggestMinVox:
    """_suggest_min_vox returns the smallest cut keeping <= max_regions."""

    def test_keeps_at_most_max_regions(self):
        size = np.arange(1, 101)  # 100 regions, sizes 1..100
        for max_regions in (1, 7, 10, 50, 99):
            cut = _suggest_min_vox(size, max_regions)
            assert (size >= cut).sum() <= max_regions

    def test_is_tight(self):
        """One voxel smaller would exceed the ceiling."""
        size = np.arange(1, 101)
        cut = _suggest_min_vox(size, 10)
        assert (size >= cut - 1).sum() > 10


class TestResolveMinVox:
    """_resolve_min_vox: int passthrough, gentle None handling."""

    def test_int_used_verbatim(self, ana):
        assert _resolve_min_vox(ana, 7, max_regions=10) == 7

    def test_explicit_zero_never_cuts(self, ana):
        # even a tiny ceiling can't override an explicit 0
        assert _resolve_min_vox(ana, 0, max_regions=1) == 0

    def test_small_tree_returns_zero(self, ana):
        # tree is well under a generous ceiling -> no cut, no warning
        assert _resolve_min_vox(ana, None, max_regions=10_000) == 0

    def test_large_tree_noninteractive_warns(self, ana, monkeypatch):
        monkeypatch.setattr(sys.stdin, 'isatty', lambda: False, raising=False)
        with pytest.warns(UserWarning, match='min_vox'):
            out = _resolve_min_vox(ana, None, max_regions=5)
        # Python callers keep every region unless they opt in
        assert out == 0

    def test_large_tree_interactive_prompts(self, ana, monkeypatch):
        monkeypatch.setattr(sys.stdin, 'isatty', lambda: True, raising=False)
        # Enter (empty) accepts the suggested cut
        monkeypatch.setattr('builtins.input', lambda *a: '')
        suggested = _suggest_min_vox(ana.size, 5)
        assert _resolve_min_vox(ana, None, max_regions=5) == suggested

    def test_interactive_decline_shows_all(self, ana, monkeypatch):
        monkeypatch.setattr(sys.stdin, 'isatty', lambda: True, raising=False)
        monkeypatch.setattr('builtins.input', lambda *a: 'n')
        assert _resolve_min_vox(ana, None, max_regions=5) == 0

    def test_interactive_custom_integer(self, ana, monkeypatch):
        monkeypatch.setattr(sys.stdin, 'isatty', lambda: True, raising=False)
        monkeypatch.setattr('builtins.input', lambda *a: '42')
        assert _resolve_min_vox(ana, None, max_regions=5) == 42


class TestDisplayRegionIds:
    def test_zero_returns_all(self, ana):
        num_reg = ana.exp.y.shape[2] + ana.children.shape[0]
        ids = _display_region_ids(ana, 0)
        assert np.array_equal(ids, np.arange(num_reg))

    def test_cut_matches_size_threshold(self, ana):
        ids = _display_region_ids(ana, 3)
        assert np.array_equal(ids, np.flatnonzero(ana.size >= 3))
        assert (ana.size[ids] >= 3).all()


class TestRegionLookupDropdown:
    """The 'Add by index' dropdown offers exactly the displayed regions."""

    def test_dropdown_restricted_to_displayed(self, ana_2d, mask_target_2d):
        min_vox = 2
        app = _create_app(ana_2d, mask_target=mask_target_2d, min_vox=min_vox)
        dd = _find_component(app.layout, 'dd-region-lookup')
        values = [opt['value'] for opt in dd.options]
        assert all(ana_2d.size[v] >= min_vox for v in values)
        assert len(values) == len(_display_region_ids(ana_2d, min_vox))

    def test_dropdown_default_has_all_regions(self, ana_2d, mask_target_2d):
        num_reg = ana_2d.exp.y.shape[2] + ana_2d.children.shape[0]
        app = _create_app(ana_2d, mask_target=mask_target_2d)
        dd = _find_component(app.layout, 'dd-region-lookup')
        assert len(dd.options) == num_reg
