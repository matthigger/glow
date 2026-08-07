"""The benchmark CLI's --aws flag routes the sweep to drive_aws.

The unified run CLI is glow._extra.benchmark; --aws hands the resolved names to
glow._extra.aws.drive_aws instead of running locally. These tests pin that
routing without touching AWS, plus the timing-cache prompt: the runtime family
cannot measure wall time on a Spot-backed Batch array, so selecting one warns
and asks before submitting.
"""

from glow._extra.benchmark import __main__ as cli
from glow._extra.benchmark.config import CONFIG


class _FakeCfg:
    @classmethod
    def from_file(cls, path=None):
        return f'CFG({path})'


def test_aws_flag_routes_to_drive_aws(monkeypatch):
    captured = {}

    def fake_drive(names, cfg, *, verbose, methods):
        captured.update(names=names, cfg=cfg, methods=methods)
        return {'sweep_llr': []}

    monkeypatch.setattr('glow._extra.aws.AWSConfig', _FakeCfg)
    monkeypatch.setattr('glow._extra.aws.drive_aws', fake_drive)

    # run reports the caches driven; the driver's per-cache failures are its
    # own return (and are printed there)
    out = cli.run(names=['sweep_llr'], aws=True, verbose=False,
                  methods=['VBA'])
    assert out == ['sweep_llr']
    assert captured['names'] == ['sweep_llr']
    assert captured['cfg'] == 'CFG(None)'        # default per-user config
    # a per-method rerun rides through to the AWS driver, which narrows the
    # shipped fnc grid with it (see glow._extra.aws.driver)
    assert captured['methods'] == ['VBA']


def test_aws_config_path_forwarded(monkeypatch):
    captured = {}
    monkeypatch.setattr('glow._extra.aws.AWSConfig', _FakeCfg)
    monkeypatch.setattr(
        'glow._extra.aws.drive_aws',
        lambda names, cfg, **kw: captured.update(cfg=cfg) or {})
    cli.run(names=['sweep_llr'], aws=True, aws_config_path='/tmp/c.json',
            verbose=False)
    assert captured['cfg'] == 'CFG(/tmp/c.json)'


def test_parse_args_accepts_aws_flag():
    # --aws routes to AWS Batch; the positional name still parses alongside it
    args = cli.parse_args(['--aws', 'sweep_llr'])
    assert args.aws
    assert args.names == ['sweep_llr']


# ---------------------------------------------------------------------------
# a selected timing cache warns and asks before submitting
# ---------------------------------------------------------------------------

def test_catalogue_has_timing_caches():
    # the prompt below is only meaningful if the pattern matches something
    timing = [name for name in CONFIG if name.startswith('runtime')]
    assert timing
    assert cli.AWS_WARN_PATTERN == 'runtime*'


class TestConfirmAws:
    def test_no_timing_cache_never_prompts(self):
        def _boom(prompt):
            raise AssertionError('must not prompt without a timing cache')
        assert cli.confirm_aws(['sweep_llr', 'null'], input_fnc=_boom) is True

    def test_yes_goes_ahead(self, capsys):
        assert cli.confirm_aws(['runtime_num_vox'],
                               input_fnc=lambda p: 'y') is True
        out = capsys.readouterr().out
        assert 'WARNING' in out and 'runtime' in out

    def test_no_aborts(self, capsys):
        assert cli.confirm_aws(['runtime_num_vox'],
                               input_fnc=lambda p: 'n') is False
        assert 'aborted' in capsys.readouterr().out

    def test_empty_reply_declines(self):
        # declining is the default, so a bare Enter aborts
        assert cli.confirm_aws(['runtime_num_vox'],
                               input_fnc=lambda p: '') is False

    def test_yes_is_case_and_space_insensitive(self):
        reply = cli.confirm_aws(['runtime_num_vox'],
                                input_fnc=lambda p: ' YES ')
        assert reply is True

    def test_warns_about_every_matching_cache(self, capsys):
        names = ['sweep_llr', 'runtime_num_vox', 'runtime_1perm_b',
                 'runtime_1perm_n_perm_inner']
        cli.confirm_aws(names, input_fnc=lambda p: 'n')
        out = capsys.readouterr().out
        assert '3 selected cache(s)' in out
        for name in names[1:]:
            assert name in out
        # the eligible cache is not named as a problem
        assert 'sweep_llr' not in out

    def test_no_terminal_declines(self, capsys):
        # pytest's stdin is not a tty, so the default path cannot ask
        assert cli.confirm_aws(['runtime_num_vox']) is False
        assert 'no terminal to ask' in capsys.readouterr().out

    def test_eof_declines(self):
        def _eof(prompt):
            raise EOFError
        assert cli.confirm_aws(['runtime_num_vox'], input_fnc=_eof) is False


class TestRunPromptsBeforeAws:
    def test_decline_submits_nothing(self, monkeypatch, capsys):
        def _boom(*a, **k):
            raise AssertionError('drive_aws must not be reached')
        monkeypatch.setattr('glow._extra.aws.AWSConfig', _FakeCfg)
        monkeypatch.setattr('glow._extra.aws.drive_aws', _boom)
        # no tty under pytest, so the prompt declines and the sweep aborts
        assert cli.run(names=['runtime*'], aws=True) == []
        assert 'nothing submitted' in capsys.readouterr().out

    def test_confirming_submits_the_whole_selection(self, monkeypatch):
        # confirming means confirming: the timing caches go too, unfiltered
        captured = {}
        monkeypatch.setattr('glow._extra.aws.AWSConfig', _FakeCfg)
        monkeypatch.setattr('glow._extra.aws.drive_aws',
                            lambda names, cfg, **kw: captured.update(
                                names=names) or {})
        monkeypatch.setattr(cli, 'confirm_aws', lambda resolved: True)
        cli.run(names=['runtime_num_vox', 'sweep_llr'], aws=True,
                verbose=False)
        assert captured['names'] == ['runtime_num_vox', 'sweep_llr']

    def test_eligible_only_selection_is_unaffected(self, monkeypatch):
        captured = {}
        monkeypatch.setattr('glow._extra.aws.AWSConfig', _FakeCfg)
        monkeypatch.setattr('glow._extra.aws.drive_aws',
                            lambda names, cfg, **kw: captured.update(
                                names=names) or {})
        cli.run(names=['sweep_llr'], aws=True, verbose=False)
        assert captured['names'] == ['sweep_llr']
