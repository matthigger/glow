"""resolve_cells: determinism, source filtering, and array sizing.

The driver resolves the cell list and ships it in the run bundle (see
glow._extra.aws.units); determinism keeps the ordering reproducible across
runs so warm-cache hits line up, so it is the key property here.
"""

from glow._extra.aws.units import resolve_cells
from glow._extra.benchmark.config import CONFIG


def test_deterministic():
    a, *_ = resolve_cells('sweep_llr', ('wgn',))
    b, *_ = resolve_cells('sweep_llr', ('wgn',))
    assert a == b


def test_wgn_filter_excludes_hcp():
    cells, *_ = resolve_cells('sweep_b', ('wgn',))
    assert cells and all(c['source'] == 'wgn' for c in cells)


def test_hcp_filter_selects_hcp():
    cells, *_ = resolve_cells('sweep_b', ('hcp',))
    assert cells and all(c['source'] == 'hcp' for c in cells)


def test_both_sources_is_full_grid():
    full = CONFIG['sweep_b'][0]
    cells, *_ = resolve_cells('sweep_b', ('wgn', 'hcp'))
    assert len(cells) == len(full)


def test_passthrough_of_effect_and_fnc_grids():
    name = 'sweep_extent'
    cells, eff, fnc_kw, fnc = resolve_cells(name, ('wgn',))
    _, eff0, fnc_kw0, fnc0 = CONFIG[name]
    assert eff == eff0 and fnc_kw == fnc_kw0 and fnc is fnc0


def test_every_cache_within_array_bound():
    # the data-cell array size must stay under the Batch arrayProperties cap
    from glow._extra.aws.driver import ARRAY_MAX
    for name in CONFIG:
        cells, *_ = resolve_cells(name, ('wgn',))
        assert len(cells) <= ARRAY_MAX
