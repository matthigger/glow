"""Tests for glow.benchmark.hcp: DUA gating, presence, Zenodo resolution.

No test touches the network or loads NIfTI: downloads are stubbed and the
regex search runs over a synthetic on-disk tree of empty files named like
the real QSIRecon dwimap maps.
"""
import io
import json

import pytest

from glow.benchmark import hcp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lay_down_maps(root, subjects=('100307', '149337'),
                   feats=hcp.HCP_FEATS):
    """Create empty dwimap files under root mirroring the archive layout.

    Names match the QSIRecon convention so SBJ_REGEX and IMG_GLOB_DICT
    select them exactly, including the doubled subject id (sub-<id>/ dir
    and filename) that exercises from_search's dedup.
    """
    model = {'fa': 'dki', 'md': 'dki', 'mk': 'dki',
             'icvf': 'noddi', 'isovf': 'noddi', 'od': 'noddi'}
    for sbj in subjects:
        dwi = root / 'derivatives' / f'sub-{sbj}' / 'dwi'
        dwi.mkdir(parents=True, exist_ok=True)
        for feat in feats:
            name = (f'sub-{sbj}_space-MNI152NLin2009cAsym_'
                    f'model-{model[feat]}_param-{feat}_dwimap.nii.gz')
            (dwi / name).write_bytes(b'')
    return root


def _use_tmp_data_dir(monkeypatch, tmp_path):
    """Point hcp.data_dir at tmp_path (no env override exists)."""
    monkeypatch.setattr(hcp, 'data_dir', lambda: tmp_path)
    return tmp_path


def _force_tty(monkeypatch, replies):
    """Make stdin look like a TTY and feed input() the given reply."""
    monkeypatch.setattr(hcp.sys, 'stdin',
                        type('S', (), {'isatty': staticmethod(
                            lambda: True)})())
    monkeypatch.setattr('builtins.input', lambda _prompt: replies)


# ---------------------------------------------------------------------------
# data_dir / is_present
# ---------------------------------------------------------------------------

class TestDataDir:
    def test_under_user_data_dir(self):
        assert hcp.data_dir().name == 'hcp100_dki_noddi_mni'


class TestIsPresent:
    def test_missing_dir(self, tmp_path):
        assert not hcp.is_present(tmp_path / 'nope')

    def test_complete_tree(self, tmp_path):
        _lay_down_maps(tmp_path)
        assert hcp.is_present(tmp_path)

    def test_partial_tree_not_present(self, tmp_path):
        # all but one feature -> a half-extracted dir must read as absent
        _lay_down_maps(tmp_path, feats=hcp.HCP_FEATS[:-1])
        assert not hcp.is_present(tmp_path)


# ---------------------------------------------------------------------------
# DUA acknowledgment
# ---------------------------------------------------------------------------

class TestAcceptDua:
    def test_non_tty_is_false(self, monkeypatch):
        # pytest captures stdin as a non-tty stream -> no prompt, no hang
        assert not hcp._accept_dua()

    def test_accept_exact(self, monkeypatch):
        _force_tty(monkeypatch, hcp.DUA_ACK_PHRASE)
        assert hcp._accept_dua()

    def test_accept_tolerates_whitespace_period_case(self, monkeypatch):
        _force_tty(monkeypatch, '  ' + hcp.DUA_ACK_PHRASE.upper() + '. ')
        assert hcp._accept_dua()

    def test_decline_other_text(self, monkeypatch):
        _force_tty(monkeypatch, 'no thanks')
        assert not hcp._accept_dua()


# ---------------------------------------------------------------------------
# Zenodo record resolution
# ---------------------------------------------------------------------------

def _fake_urlopen(payload: dict):
    """Return a urlopen stub yielding json.dumps(payload) as a context mgr."""
    body = json.dumps(payload).encode('utf-8')

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def _open(req, timeout=None):
        return _Resp(body)

    return _open


class TestArchiveUrl:
    def test_parses_url_and_md5(self, monkeypatch):
        payload = {'files': [
            {'key': 'maps.zip', 'checksum': 'md5:deadbeef',
             'links': {'self': 'http://x/maps.zip/content'}}]}
        monkeypatch.setattr(hcp.urllib.request, 'urlopen',
                            _fake_urlopen(payload))
        url, md5 = hcp._archive_url()
        assert url == 'http://x/maps.zip/content'
        assert md5 == 'deadbeef'

    def test_no_files_raises(self, monkeypatch):
        monkeypatch.setattr(hcp.urllib.request, 'urlopen',
                            _fake_urlopen({'files': []}))
        with pytest.raises(RuntimeError, match='no downloadable file'):
            hcp._archive_url()


# ---------------------------------------------------------------------------
# ensure_hcp_data
# ---------------------------------------------------------------------------

class TestEnsureHcpData:
    def test_present_returns_without_prompt(self, monkeypatch, tmp_path):
        _use_tmp_data_dir(monkeypatch, tmp_path)
        _lay_down_maps(tmp_path)

        def _boom():
            raise AssertionError('should not prompt when data present')
        monkeypatch.setattr(hcp, '_accept_dua', _boom)

        assert hcp.ensure_hcp_data() == tmp_path

    def test_absent_without_agreement_raises(self, monkeypatch, tmp_path):
        _use_tmp_data_dir(monkeypatch, tmp_path / 'empty')
        monkeypatch.setattr(hcp, '_accept_dua', lambda: False)
        with pytest.raises(RuntimeError, match='Data Use Terms'):
            hcp.ensure_hcp_data()

    def test_absent_downloads_when_agreed(self, monkeypatch, tmp_path):
        dest = _use_tmp_data_dir(monkeypatch, tmp_path / 'data')
        monkeypatch.setattr(hcp, '_accept_dua', lambda: True)

        # stand in for the network: lay the maps where extract would put them
        monkeypatch.setattr(hcp, '_download_and_extract',
                            lambda folder: _lay_down_maps(folder))

        assert hcp.ensure_hcp_data() == dest
        assert hcp.is_present(dest)
