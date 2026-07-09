"""resolve_cells: determinism, data x effect crossing, and array sizing.

The driver resolves the cell list and ships it in the run bundle (see
glow._extra.aws.units); determinism keeps the ordering reproducible across
runs so warm-cache hits line up, so it is the key property here. A cell is one
data cell crossed with one effect (a planted cell), and sources are a CONFIG
property (the data grid), not a driver knob -- resolve_cells crosses the grid
in full, whatever sources the cache declares.
"""

from glow._extra.aws.units import resolve_cells
from glow._extra.benchmark.config import CONFIG


def test_deterministic():
    a, *_ = resolve_cells('sweep_llr')
    b, *_ = resolve_cells('sweep_llr')
    assert a == b


def test_crosses_data_and_effect_grids():
    # a cell is a (kwargs_data, kwargs_effect) pair; resolve_cells is the full
    # data x effect cross in data-major, effect-minor CONFIG order
    data_list, effect_list, *_ = CONFIG['sweep_llr']
    cells, *_ = resolve_cells('sweep_llr')
    assert cells == [(d, e) for d in data_list for e in effect_list]


def test_keeps_every_declared_source():
    # sweep_llr spans both sources in CONFIG; resolve_cells keeps them all
    cells, *_ = resolve_cells('sweep_llr')
    sources = {kwargs_data['source'] for kwargs_data, _ in cells}
    assert sources == {'wgn', 'hcp'}


def test_passthrough_of_fnc_grid_and_effects():
    name = 'sweep_extent'
    cells, fnc_kw, fnc = resolve_cells(name)
    data0, eff0, fnc_kw0, fnc0 = CONFIG[name]
    assert fnc_kw == fnc_kw0 and fnc is fnc0
    # every effect cell rides into the cells, paired with each data cell
    assert cells == [(d, e) for d in data0 for e in eff0]


def test_every_cache_within_array_bound():
    # the planted-cell array size must stay under the Batch arrayProperties cap
    from glow._extra.aws.driver import ARRAY_MAX
    for name in CONFIG:
        cells, *_ = resolve_cells(name)
        assert len(cells) <= ARRAY_MAX
