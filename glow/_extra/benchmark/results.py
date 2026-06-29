"""Slice the shared provenance frame into one tidy CSV per CONFIG entry.

The driver writes every run_ana leaf into one shared RECORDER (data.RECORDER):
``RECORDER.flatten_to_df`` yields one row per leaf, each carrying the data ->
(plant ->) score chain that produced it (see glow._extra.benchmark.recorder /
driver). That frame is config-agnostic -- it holds every cell ever run, across
every figure -- so this module re-attaches the CONFIG catalogue: for one cache
name it recomputes the args-hashes that name's grid produces and keeps only the
matching leaf rows, then writes ``<name>.csv``.

Membership is by content hash, not a stamped column, *because the sweeps share
cells*. The catalogue anchors each structural sweep on the same baseline cell
(the grid-midpoint effect_llr at the default extent, b=1, num_img=100), so the
moderate baseline run is the very same run_ana call -- one args hash, one cached
result, one record -- in sweep_llr, sweep_b, sweep_extent and (WGN) sweep_nimg.
A single record therefore belongs to several caches at once, which a stamped
"which cache" field could not express (the last writer would clobber the rest).
Recomputing each cache's hashes here lets a shared row land in *every* CSV that
asked for it, with no record-time bookkeeping.

The recompute walks the same data x effect x fnc grid the driver does (build the
clean exp, plant each effect, enumerate the fnc kwargs), but instead of running
the leaf it reproduces the *record* key that call writes -- ``Recorder._args_hash``
of (exp, mask_target_list, **kwargs) against the leaf's undecorated signature.
That is the recorder's own keying (``ignore=[]``), so it INCLUDES the run_ana
label; it is NOT the joblib cache id, which ignores label
(``@MEMORY.cache(ignore=['label'])``) and so differs. The frame is keyed by the
record hash (``run_ana.hash``), so matching it needs the record key, label and
all. The builds are disk-memoised, so this is cache hits, not recomputation; the
cost is loading each exp once per cache it appears in.
"""
from pathlib import Path

from .config import CONFIG
from .data import RECORDER, data_factory, effect_factory
from .file import get_path_result
from .recorder import Recorder

# the leaf's flatten_to_df column carrying its own args hash (the record key):
# flatten prefixes every column by the record's short function name, and the
# run_ana score step is the leaf, so its hash is under 'run_ana.hash'.
_LEAF_HASH_COL = 'run_ana.hash'


def config_leaf_hashes(name: str) -> set:
    """The set of leaf args-hashes one CONFIG entry's grid would produce.

    Walks the cache's data x effect x fnc grid exactly as the driver does
    (build the clean exp, plant each effect or skip it for a ``None`` cell,
    enumerate the fnc kwargs), but reproduces the *record* key each leaf call
    writes rather than running it: ``Recorder._args_hash`` of (exp,
    mask_target_list, **kwargs) against the leaf's undecorated signature. That
    is the recorder's keying (``ignore=[]``), so it includes the run_ana label
    -- it is the ``run_ana.hash`` the frame is keyed by, NOT the joblib cache id
    (which ignores label). The builds are cache hits (disk-memoised), so no
    experiment is recomputed; each is loaded once.

    Args:
        name (str): a CONFIG cache name (e.g. 'sweep_llr').

    Returns:
        set[str]: the run_ana record hashes for every cell of this cache's grid.
    """
    kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc = CONFIG[name]

    # the recorder hashes a record by joblib.hash of the call's filtered args
    # (ignore=[]), so it INCLUDES label -- unlike fnc's joblib cache id, which
    # ignores it. fnc.func is the recorder-wrapped fn; its __wrapped__ is the
    # undecorated leaf whose signature _args_hash binds to reproduce that key.
    raw = fnc.func.__wrapped__

    hashes = set()
    for kwargs_data in kwargs_data_list:
        exp = data_factory(**kwargs_data)
        for kwargs_effect in kwargs_effect_list:
            if kwargs_effect is None:
                # null / FWER-calibration cell: clean exp, empty target
                exp_eff, mask_target_list = exp, []
            else:
                exp_eff, mask = effect_factory(exp, **kwargs_effect)
                mask_target_list = [mask]
            for kwargs in kwargs_fnc_list:
                # the record key this leaf call writes (its run_ana.hash): the
                # recorder hashes (exp, mask_target_list, **kwargs) -- label and
                # all -- against the undecorated signature, so we do the same
                hashes.add(Recorder._args_hash(
                    raw, (exp_eff,),
                    dict(mask_target_list=mask_target_list, **kwargs)))
    return hashes


def config_results_df(name: str, df=None):
    """The shared provenance frame, filtered to one cache's leaf rows.

    Loads (or reuses a passed) ``RECORDER.flatten_to_df`` frame and keeps the
    rows whose leaf hash is one this cache's grid produces (config_leaf_hashes)
    -- so a row a sweep shares with another sweep appears in both caches' frames.
    Empty when no row of this cache has been run yet.

    Args:
        name (str): a CONFIG cache name.
        df (pandas.DataFrame | None): a frame from ``RECORDER.flatten_to_df``;
            None loads it fresh (``RECORDER.load`` first, to fold in any records
            other writers / workers left on disk).

    Returns:
        pandas.DataFrame: this cache's rows (empty when none have run).
    """
    if df is None:
        RECORDER.load()
        df = RECORDER.flatten_to_df()
    if df.empty or _LEAF_HASH_COL not in df.columns:
        return df.iloc[0:0]
    return df[df[_LEAF_HASH_COL].isin(config_leaf_hashes(name))]


def write_config_csvs(out_dir=None, names=None) -> dict:
    """Write one ``<name>.csv`` per CONFIG cache that has rows on disk.

    Flattens the shared recorder once, then slices it per cache
    (config_results_df) and writes each non-empty slice to ``out_dir/<name>.csv``
    -- the per-figure results frame the plots consume. A cache with no rows yet
    is skipped (no empty file).

    Args:
        out_dir (str | Path | None): destination directory; defaults to glow's
            per-user results dir (get_path_result).
        names (iterable[str] | None): cache names to export; None exports every
            CONFIG entry.

    Returns:
        dict[str, Path]: the {cache name: written csv path} for caches that had
            rows (those skipped for being empty are omitted).
    """
    out_dir = Path(out_dir) if out_dir is not None else get_path_result()
    out_dir.mkdir(parents=True, exist_ok=True)

    RECORDER.load()
    df = RECORDER.flatten_to_df()

    written = {}
    for name in (names if names is not None else CONFIG):
        sub = config_results_df(name, df=df)
        if sub.empty:
            continue
        path = out_dir / f'{name}.csv'
        sub.to_csv(path, index=False)
        written[name] = path
    return written
