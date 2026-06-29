"""The benchmark CLI's --aws flag routes the sweep to drive_aws.

The unified run CLI is glow._extra.benchmark; --aws hands the resolved names to
glow._extra.aws.drive_aws instead of running locally. These tests pin that
routing (and the precedence of --csv-only) without touching AWS.
"""

from glow._extra.benchmark import __main__ as cli


class _FakeCfg:
    @classmethod
    def from_file(cls, path=None):
        return f'CFG({path})'


def test_aws_flag_routes_to_drive_aws(monkeypatch):
    captured = {}

    def fake_drive(names, cfg, *, sources, write_csv, verbose, out_dir):
        captured.update(names=names, cfg=cfg, sources=sources,
                        write_csv=write_csv, out_dir=out_dir)
        return {'sweep_llr': 'path'}

    monkeypatch.setattr('glow._extra.aws.AWSConfig', _FakeCfg)
    monkeypatch.setattr('glow._extra.aws.drive_aws', fake_drive)

    out = cli.run(names=['sweep_llr'], aws=True, sources=('wgn',),
                  write_csv=False, verbose=False)
    assert out == {'sweep_llr': 'path'}
    assert captured['names'] == ['sweep_llr']
    assert captured['cfg'] == 'CFG(None)'        # default per-user config
    assert captured['sources'] == ('wgn',)
    assert captured['write_csv'] is False


def test_aws_config_path_forwarded(monkeypatch):
    captured = {}
    monkeypatch.setattr('glow._extra.aws.AWSConfig', _FakeCfg)
    monkeypatch.setattr('glow._extra.aws.drive_aws',
                        lambda names, cfg, **kw: captured.update(cfg=cfg))
    cli.run(names=['sweep_llr'], aws=True, aws_config_path='/tmp/c.json',
            verbose=False)
    assert captured['cfg'] == 'CFG(/tmp/c.json)'


def test_csv_only_short_circuits_before_aws(monkeypatch):
    # --csv-only rebuilds from local records; it must not reach drive_aws
    monkeypatch.setattr('glow._extra.benchmark.results.write_config_csvs',
                        lambda out_dir=None, names=None: {})

    def _boom(*a, **k):
        raise AssertionError('drive_aws must not be called under csv_only')
    monkeypatch.setattr('glow._extra.aws.drive_aws', _boom)
    assert cli.run(names=['sweep_llr'], aws=True, csv_only=True,
                   verbose=False) == {}


def test_parse_args_accepts_aws_flags():
    # --sources is one comma-separated value, so it never eats the positional
    args = cli.parse_args(['--aws', '--sources', 'wgn,hcp', 'sweep_llr'])
    assert args.aws and args.sources == ('wgn', 'hcp')
    assert args.names == ['sweep_llr']


def test_parse_args_rejects_bad_source():
    import pytest
    with pytest.raises(SystemExit):
        cli.parse_args(['--sources', 'bogus'])
