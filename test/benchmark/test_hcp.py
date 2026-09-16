"""Tests for glow._extra.benchmark.hcp: DUA gating, presence, Zenodo resolution.

No test touches the network or loads NIfTI: downloads are stubbed and the
regex search runs over a synthetic on-disk tree of empty files named like
the real QSIRecon dwimap maps.
"""
import io
import json

import numpy as np
import pytest

from glow._extra.benchmark import hcp
from glow.mask import get_mask_idx


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lay_down_maps(root, subjects=('100307', '149337'),
                   feats=hcp.HCP_FEATS, mask=True):
    """Create empty dwimap files under root mirroring the archive layout.

    Names match the QSIRecon convention so SBJ_REGEX and IMG_GLOB_DICT
    select them exactly, including the doubled subject id (sub-<id>/ dir
    and filename) that exercises from_search's dedup.  When mask is True
    the dataset brain-mask file (MASK_GLOB) is laid at the archive root.
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
    if mask:
        root.mkdir(parents=True, exist_ok=True)
        (root / 'brain_mask_space-MNI152NLin2009cAsym.nii.gz').write_bytes(b'')
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

    def test_missing_mask_not_present(self, tmp_path):
        # all six maps but no brain mask (e.g. a pre-mask archive) -> absent
        _lay_down_maps(tmp_path, mask=False)
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


# ---------------------------------------------------------------------------
# atomic bundle writes
# ---------------------------------------------------------------------------

class TestWriteAtomic:
    """A reader sees a bundle file whole or absent, never partly written.

    is_bundle_present tests existence alone, so a parallel sweep converting a
    cold bundle would otherwise have its other workers load a truncated array
    (a 90 MB feature takes a visible while to write).
    """

    def test_destination_holds_the_whole_write(self, tmp_path):
        dest = tmp_path / 'feat.npy'
        hcp._write_atomic(dest, lambda h: h.write(b'0123456789'))
        assert dest.read_bytes() == b'0123456789'
        # nothing staged is left behind
        assert list(tmp_path.iterdir()) == [dest]

    def test_a_failed_write_publishes_nothing(self, tmp_path):
        dest = tmp_path / 'feat.npy'

        def _boom(handle):
            """Write half the file, then fail as a full disk would."""
            handle.write(b'01234')
            raise OSError('no space left on device')

        with pytest.raises(OSError, match='no space'):
            hcp._write_atomic(dest, _boom)
        # the presence check is existence, so a partial file must not exist
        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_second_writer_stages_separately(self, tmp_path, monkeypatch):
        """Two processes converting at once stage under their own pid.

        Both publish the same bytes (one source, one array), so the duplicated
        work is all the collision costs -- but only if neither writes into the
        other's staging file.
        """
        dest = tmp_path / 'feat.npy'
        staged = []

        def _record(handle):
            """Note the staging path this writer opened."""
            staged.append(handle.name)
            handle.write(b'x')

        monkeypatch.setattr(hcp.os, 'getpid', lambda: 111)
        hcp._write_atomic(dest, _record)
        monkeypatch.setattr(hcp.os, 'getpid', lambda: 222)
        hcp._write_atomic(dest, _record)
        assert staged[0] != staged[1]
        assert dest.read_bytes() == b'x'


# ---------------------------------------------------------------------------
# SourceBundle / build_exp_img_from_bundle
# ---------------------------------------------------------------------------

def _lay_down_bundle(monkeypatch, tmp_path, feats=('fa', 'md'), shape=(2, 3, 4),
                     num_img=3):
    """Write a synthetic npy bundle under tmp_path and point hcp at it.

    Every path helper goes through bundle_dir, so patching that one moves
    the whole bundle. Feature arrays hold distinct ramps, so a stack in
    the wrong feature order, or a gather in the wrong voxel order, cannot
    pass for the right one.

    Returns:
        mask (np.array): (X, Y, Z) boolean support, one voxel off
        y_feat (dict): feat -> (num_img, num_vox) array as written
    """
    monkeypatch.setattr(hcp, 'bundle_dir', lambda: tmp_path)
    (tmp_path / 'feat').mkdir(parents=True, exist_ok=True)

    mask = np.ones(shape, dtype=bool)
    mask[0, 0, 0] = False
    num_vox = int(mask.sum())
    np.save(hcp.bundle_mask_path(), mask)
    np.save(hcp.bundle_affine_path(), np.diag([2.0, 2.0, 2.0, 1.0]))
    hcp.bundle_meta_path().write_text(json.dumps(
        {'subjects': [f'sbj{i}' for i in range(num_img)]}))

    y_feat = {}
    for feat_idx, feat in enumerate(feats):
        y = (np.arange(num_img * num_vox, dtype=np.float32)
             .reshape((num_img, num_vox)) + 1000 * feat_idx)
        np.save(hcp.bundle_feat_path(feat), y)
        y_feat[feat] = y
    return mask, y_feat


class TestSourceBundle:
    """The bundle reads back through a source, in the order it was asked."""

    def test_load_stacks_the_features_asked_for(self, monkeypatch, tmp_path):
        """y's feature axis follows the feats argument, not the bundle."""
        _, y_feat = _lay_down_bundle(monkeypatch, tmp_path)
        source = hcp.SourceBundle(('md', 'fa'))
        y = source.load()
        assert source.features == ('md', 'fa')
        assert np.array_equal(y[0], y_feat['md'])
        assert np.array_equal(y[1], y_feat['fa'])

    def test_subset_matches_the_columns_it_stands_for(self, monkeypatch,
                                                      tmp_path):
        """A gather returns the mask's columns in get_mask_idx order."""
        mask, _ = _lay_down_bundle(monkeypatch, tmp_path)
        source = hcp.SourceBundle(('fa',))
        y_all = source.load()

        sub = mask.copy()
        sub.ravel()[1::2] = False
        want = y_all[:, :, get_mask_idx(mask)[sub]]
        assert np.array_equal(source.load(sub), want)

    def test_reaching_past_the_brain_raises(self, monkeypatch, tmp_path):
        """The brain mask is the ceiling; asking past it is refused."""
        mask, _ = _lay_down_bundle(monkeypatch, tmp_path)
        source = hcp.SourceBundle(('fa',))
        with pytest.raises(ValueError, match='outside the source support'):
            source.load(np.ones(mask.shape, dtype=bool))

    def test_the_loader_reads_through_the_source(self, monkeypatch, tmp_path):
        """build_exp_img_from_bundle returns the source's own y, and it."""
        mask, y_feat = _lay_down_bundle(monkeypatch, tmp_path)
        exp = hcp.build_exp_img_from_bundle(('fa', 'md'))

        assert isinstance(exp.source, hcp.SourceBundle)
        assert np.array_equal(exp.y, exp.source.load())
        assert np.array_equal(exp.y[0], y_feat['fa'])
        assert np.array_equal(exp.mask_idx, get_mask_idx(mask))
        assert exp.meta['features'] == ['fa', 'md']
        assert exp.meta['subjects'] == ['sbj0', 'sbj1', 'sbj2']
        assert list(exp.meta) == ['subjects', 'features', 'affine']
