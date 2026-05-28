"""infra CLI: interactive .glow_aws_config creation by `bootstrap`."""

import pytest

from glow.aws import infra
from glow.aws.config import AWSConfig


def _answers(*responses):
    """An input() stand-in replaying scripted terminal responses in order."""
    it = iter(responses)
    return lambda _prompt='': next(it)


@pytest.mark.parametrize('answer', ['y', 'Y', 'yes', ''])
def test_init_config_accepts_defaults(tmp_path, monkeypatch, answer):
    """Accepting the prompt (or a blank answer) writes the default names."""
    path = tmp_path / '.glow_aws_config'
    monkeypatch.setattr('builtins.input', _answers(answer))

    infra._init_config(str(path))

    cfg = AWSConfig.from_file(path)
    for key, default in infra.DEFAULT_CONFIG_NAMES.items():
        assert getattr(cfg, key) == default


def test_init_config_prompts_each_on_decline(tmp_path, monkeypatch):
    """Declining the defaults prompts per field; a blank keeps that default."""
    path = tmp_path / '.glow_aws_config'
    monkeypatch.setattr(
        'builtins.input', _answers('n', 'my-bucket', '', 'my-def'))

    infra._init_config(str(path))

    cfg = AWSConfig.from_file(path)
    assert cfg.s3_bucket == 'my-bucket'
    assert cfg.job_queue == infra.DEFAULT_CONFIG_NAMES['job_queue']
    assert cfg.job_definition == 'my-def'


@pytest.mark.parametrize('answer, expected', [
    ('', True), ('y', True), ('Y', True), ('yes', True),
    ('n', False), ('no', False), ('nope', False),
])
def test_prompt_yes(monkeypatch, answer, expected):
    monkeypatch.setattr('builtins.input', _answers(answer))
    assert infra._prompt_yes('ok?') is expected
