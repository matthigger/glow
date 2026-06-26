"""Tests for the Zenodo fetch-and-mount path of the web viewer.

Unit-covers the zenodo client (file-list parsing across both API shapes,
the size cap, MD5 verification) and end-to-end-covers the server flow
(picker -> load -> dynamic mount -> redirect) against a local HTTP stub
standing in for the Zenodo API, so nothing here touches the network.
"""

import gzip
import hashlib
import http.server
import pickle
import threading

import pytest
from werkzeug.test import Client

from glow._extra.viewer.web import zenodo


# ---------- zenodo client units ---------------------------------------------


def test_normalize_files_list_shape():
    """Parse the top-level `files` list shape (current zenodo.org)."""
    record = {'files': [
        {'key': 'a.p.gz', 'size': 10, 'checksum': 'md5:abc',
         'links': {'self': 'http://h/a/content'}},
        {'key': 'no-link', 'size': 5, 'checksum': 'md5:def', 'links': {}},
    ]}
    files = zenodo._normalize_files(record)
    assert [f.key for f in files] == ['a.p.gz']
    assert files[0].md5 == 'abc' and files[0].size == 10
    assert files[0].url == 'http://h/a/content'


def test_normalize_files_entries_shape():
    """Parse the InvenioRDM `files.entries` dict shape."""
    record = {'files': {'entries': {
        'b.pkl': {'key': 'b.pkl', 'size': 7, 'checksum': 'md5:111',
                  'links': {'content': 'http://h/b/content'}}}}}
    files = zenodo._normalize_files(record)
    assert len(files) == 1 and files[0].key == 'b.pkl'
    assert files[0].url == 'http://h/b/content' and files[0].md5 == '111'


def test_fetch_file_md5_mismatch(tmp_path, monkeypatch):
    """fetch_file raises when the downloaded bytes fail the MD5 check."""
    payload = b'hello world'
    zf = zenodo.ZenodoFile(key='x.bin', size=len(payload),
                           md5='deadbeef', url='http://stub/x')

    class _Resp:
        def __init__(self, data): self._data = data
        def read(self, n=-1):
            out, self._data = self._data, b''
            return out
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(zenodo.urllib.request, 'urlopen',
                        lambda *a, **k: _Resp(payload))
    with pytest.raises(ValueError, match='MD5 mismatch'):
        zenodo.fetch_file(zf, dest_dir=tmp_path)


def test_fetch_file_size_cap(tmp_path):
    """fetch_file rejects a file whose advertised size is over the cap."""
    zf = zenodo.ZenodoFile(key='big.bin', size=10_000, md5='', url='http://x')
    with pytest.raises(ValueError, match='over the'):
        zenodo.fetch_file(zf, max_bytes=1_000, dest_dir=tmp_path)


# ---------- end-to-end server flow against a local API stub -----------------


@pytest.fixture
def zenodo_space(ana_2d, exp_2d, mask_target_2d, tmp_path, monkeypatch):
    """Serve one baked pickle from a local Zenodo-API stub; build the app.

    Yields (client, slug, filename) where client drives the dispatcher-
    wrapped server with GLOW_ZENODO_* pointed at the stub.
    """
    blob = gzip.compress(pickle.dumps(
        {'ana': ana_2d, 'exp': exp_2d, 'mask_target': mask_target_2d,
         'combo': {}}))
    md5 = hashlib.md5(blob).hexdigest()
    filename = 'demo_wgn2d.p.gz'
    record_id = '424242'

    state = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == f'/api/records/{record_id}':
                body = (
                    b'{"files": [{"key": "%b", "size": %d, '
                    b'"checksum": "md5:%b", "links": {"self": '
                    b'"http://%b/api/records/%b/files/%b/content"}}]}'
                ) % (filename.encode(), len(blob), md5.encode(),
                     state['host'].encode(), record_id.encode(),
                     filename.encode())
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
            elif self.path.endswith(f'/files/{filename}/content'):
                body = blob
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    host = f'127.0.0.1:{httpd.server_address[1]}'
    state['host'] = host
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    monkeypatch.setenv('GLOW_ZENODO_API_BASE', f'http://{host}/api')
    monkeypatch.setenv('GLOW_ZENODO_RECORD_ID', record_id)
    monkeypatch.setenv('GLOW_ZENODO_CACHE_DIR', str(tmp_path / 'cache'))

    # Import here so the module exists; build a fresh app under the env above.
    from glow._extra.viewer.web import server
    app = server.build_application(pickle_dir=str(tmp_path / 'no_demos'))
    client = Client(app)
    slug = 'demo_wgn2d'
    try:
        yield client, slug, filename, app
    finally:
        httpd.shutdown()


def test_picker_lists_the_file(zenodo_space):
    """The /zenodo/ picker lists the record's file with a load link."""
    client, slug, filename, _ = zenodo_space
    r = client.get('/zenodo/')
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert filename in body
    assert f'/zenodo/load/{slug}' in body


def test_load_mounts_and_serves_viewer(zenodo_space):
    """Loading a file mounts a viewer and 302s to a working Dash app."""
    client, slug, _, _ = zenodo_space
    r = client.get(f'/zenodo/load/{slug}')
    assert r.status_code == 302
    assert r.headers['Location'].endswith(f'/zenodo/view/{slug}/')

    layout = client.get(f'/zenodo/view/{slug}/_dash-layout')
    assert layout.status_code == 200


def test_load_is_idempotent(zenodo_space):
    """Loading the same slug twice reuses the one mount."""
    client, slug, _, app = zenodo_space
    client.get(f'/zenodo/load/{slug}')
    client.get(f'/zenodo/load/{slug}')
    mounter = app.app.zenodo_mounter
    assert list(mounter.mounted_) == [slug]


def test_unknown_slug_404(zenodo_space):
    """A slug not in the record 404s rather than fetching anything."""
    client, _, _, _ = zenodo_space
    assert client.get('/zenodo/load/not-a-real-file').status_code == 404
