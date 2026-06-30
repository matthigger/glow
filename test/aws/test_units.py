"""resolve_cells: determinism, full-grid passthrough, and array sizing.

The driver resolves the cell list and ships it in the run bundle (see
glow._extra.aws.units); determinism keeps the ordering reproducible across
runs so warm-cache hits line up, so it is the key property here. Sources are a
CONFIG property (the data grid), not a driver knob -- resolve_cells returns the
grid in full, whatever sources the cache declares.
"""

from glow._extra.aws.units import resolve_cells
from glow._extra.benchmark.config import CONFIG


def test_deterministic():
    a, *_ = resolve_cells('sweep_llr')
    b, *_ = resolve_cells('sweep_llr')
    assert a == b


def test_returns_full_config_grid():
    full = CONFIG['sweep_b'][0]
    cells, *_ = resolve_cells('sweep_b')
    assert cells == list(full)


def test_keeps_every_declared_source():
    # sweep_b spans both sources in CONFIG; resolve_cells keeps them all
    cells, *_ = resolve_cells('sweep_b')
    assert {c['source'] for c in cells} == {'wgn', 'hcp'}


def test_passthrough_of_effect_and_fnc_grids():
    name = 'sweep_extent'
    cells, eff, fnc_kw, fnc = resolve_cells(name)
    _, eff0, fnc_kw0, fnc0 = CONFIG[name]
    assert eff == eff0 and fnc_kw == fnc_kw0 and fnc is fnc0


def test_every_cache_within_array_bound():
    # the data-cell array size must stay under the Batch arrayProperties cap
    from glow._extra.aws.driver import ARRAY_MAX
    for name in CONFIG:
        cells, *_ = resolve_cells(name)
        assert len(cells) <= ARRAY_MAX
