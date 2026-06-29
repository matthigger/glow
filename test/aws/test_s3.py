"""s3: uri parse, dir upload/download round-trip, background uploader."""

import pytest

from glow._extra.aws import s3
from test.aws.fakes import FakeS3


def test_parse_s3_uri():
    assert s3.parse_s3_uri('s3://bkt/a/b/c.json') == ('bkt', 'a/b/c.json')
    assert s3.parse_s3_uri('s3://bkt') == ('bkt', '')
    with pytest.raises(ValueError):
        s3.parse_s3_uri('/not/s3')


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_upload_download_round_trip(tmp_path):
    src = tmp_path / 'src'
    _write(src / 'a.json', 'A')
    _write(src / 'sub' / 'b.json', 'B')
    fake = FakeS3()

    n = s3.upload_dir(fake, 'bkt', src, 'glow/records')
    assert n == 2
    assert ('bkt', 'glow/records/a.json') in fake.store
    assert ('bkt', 'glow/records/sub/b.json') in fake.store

    dst = tmp_path / 'dst'
    n = s3.download_prefix(fake, 'bkt', 'glow/records', dst)
    assert n == 2
    assert (dst / 'a.json').read_text() == 'A'
    assert (dst / 'sub' / 'b.json').read_text() == 'B'


def test_upload_skips_existing(tmp_path):
    src = tmp_path / 'src'
    _write(src / 'a.json', 'A')
    fake = FakeS3()
    assert s3.upload_dir(fake, 'bkt', src, 'p') == 1
    # second call: object already there -> head probe, no re-upload
    assert s3.upload_dir(fake, 'bkt', src, 'p') == 0


def test_download_skips_existing_local(tmp_path):
    src = tmp_path / 'src'
    _write(src / 'a.json', 'A')
    fake = FakeS3()
    s3.upload_dir(fake, 'bkt', src, 'p')
    dst = tmp_path / 'dst'
    assert s3.download_prefix(fake, 'bkt', 'p', dst) == 1
    assert s3.download_prefix(fake, 'bkt', 'p', dst) == 0


def test_missing_dir_uploads_nothing(tmp_path):
    fake = FakeS3()
    assert s3.upload_dir(fake, 'bkt', tmp_path / 'nope', 'p') == 0


def test_background_uploader_flush(tmp_path):
    src = tmp_path / 'src'
    _write(src / 'a.json', 'A')
    fake = FakeS3()
    up = s3.BackgroundUploader(fake, 'bkt', [(src, 'p')], interval=999)
    assert up.flush() == 1
    # a new file appears; the seen-set means only it ships on the next flush
    _write(src / 'b.json', 'B')
    assert up.flush() == 1
    assert up.flush() == 0


def test_background_uploader_thread(tmp_path):
    src = tmp_path / 'src'
    _write(src / 'a.json', 'A')
    fake = FakeS3()
    with s3.BackgroundUploader(fake, 'bkt', [(src, 'p')], interval=0.01):
        pass  # __exit__ stops + final flush
    assert ('bkt', 'p/a.json') in fake.store
