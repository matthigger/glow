"""mv_cache: archive the local cache and records to a dated old/ folder.

archive_caches moves the resolved cache and records dirs into a dated old/
sibling in lock step and leaves fresh empty dirs behind. It runs against tmp
dirs so no real user-data dir is touched.
"""

from glow._extra.benchmark import mv_cache


def _seed(tmp_path, monkeypatch, cache=True, records=True):
    """Point get_path_cache / get_path_records at tmp dirs, seeding each.

    Returns (cache_dir, records_dir); a dir is left empty when its flag is
    False so the empty-source path can be exercised.
    """
    cache_dir = tmp_path / 'cache'
    records_dir = tmp_path / 'records'
    cache_dir.mkdir()
    records_dir.mkdir()
    if cache:
        (cache_dir / 'run_ana' / 'abc').mkdir(parents=True)
        (cache_dir / 'run_ana' / 'abc' / 'output.pkl').write_bytes(b'blob')
    if records:
        (records_dir / 'abc.json').write_bytes(b'{"hash": "abc"}')
    monkeypatch.setattr(mv_cache, 'get_path_cache', lambda: cache_dir)
    monkeypatch.setattr(mv_cache, 'get_path_records', lambda: records_dir)
    return cache_dir, records_dir


def test_archive_moves_both_in_lockstep_and_recreates_empty(
        tmp_path, monkeypatch):
    cache_dir, records_dir = _seed(tmp_path, monkeypatch)

    cache_dest, records_dest = mv_cache.archive_caches(stamp='2026-07-13')

    old = tmp_path / 'old'
    assert cache_dest == old / 'cache-2026-07-13'
    assert records_dest == old / 'records-2026-07-13'
    assert (cache_dest / 'run_ana' / 'abc' / 'output.pkl').read_bytes() \
        == b'blob'
    assert (records_dest / 'abc.json').read_bytes() == b'{"hash": "abc"}'
    # fresh, empty dirs are left in place so the next run starts cold
    for d in (cache_dir, records_dir):
        assert d.is_dir() and not any(d.iterdir())


def test_archive_second_run_same_day_shares_a_bumped_stamp(
        tmp_path, monkeypatch):
    cache_dir, records_dir = _seed(tmp_path, monkeypatch)
    first_cache, first_records = mv_cache.archive_caches(stamp='2026-07-13')

    (cache_dir / 'x').write_bytes(b'more')
    (records_dir / 'y.json').write_bytes(b'{}')
    second_cache, second_records = mv_cache.archive_caches(stamp='2026-07-13')

    # both bump off the taken date stamp, and to the *same* new stamp so
    # cache/records stay in lock step
    assert second_cache != first_cache
    assert second_records != first_records
    assert second_cache.name.split('cache-')[1] \
        == second_records.name.split('records-')[1]
    assert first_cache.is_dir() and first_records.is_dir()


def test_archive_empty_dirs_are_noop(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, cache=False, records=False)

    assert mv_cache.archive_caches() == (None, None)
    assert not (tmp_path / 'old').exists()

