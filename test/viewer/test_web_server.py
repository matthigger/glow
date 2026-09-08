"""Tests for the hosted multi-demo web server.

Covers the manifest read and the request flow the Space runs on: landing
page -> load -> dynamic mount -> redirect -> a working Dash app, plus the
LRU that bounds how many viewers stay resident. The bundle is fit with
AnalysisGLOW, the per-perm arm the baked set ships, so the hosted path is
exercised on the arm it actually serves.
"""

import gzip
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest
from werkzeug.test import Client

from glow.analysis import AnalysisGLOW
from glow.effect import EffectSynthetic
from glow.experiment.exper import Experiment
from glow._extra.viewer.web import server


@pytest.fixture(scope='module')
def bundle():
    """Build one tiny per-perm GLOW bundle, as bake_demos would write it."""
    shape = (5, 5, 5)
    center = np.array([s // 2 for s in shape])
    coords = np.indices(shape).reshape(3, -1).T
    dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
    mask = (dist <= 2.0).reshape(shape)

    exp = Experiment.from_gauss(b=1, num_img=20, shape=shape, seed=0, a=2)
    exp_eff = EffectSynthetic(mask=mask, effect_llr=2.0, seed=0).fit(exp)[0]
    ana = AnalysisGLOW(n_perm_fwer=5, n_perm_inner=5).fit(exp_eff)
    return {'ana': ana, 'exp': exp_eff, 'mask_target': mask}


@pytest.fixture(scope='module')
def pickle_dir(tmp_path_factory, bundle):
    """Write two copies of the bundle plus a manifest naming both."""
    out = tmp_path_factory.mktemp('pickles')
    keys = ['llr_moderate', 'null']
    for key in keys:
        with gzip.open(out / f'{key}.p.gz', 'wb') as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    entries = [{'key': k, 'blurb': f'blurb for {k}', 'cache': 'sweep_llr',
                'source': 'wgn',
                'size_bytes': (out / f'{k}.p.gz').stat().st_size}
               for k in keys]
    (out / server.MANIFEST_NAME).write_text(json.dumps({'demos': entries}))
    return out


@pytest.fixture
def client(pickle_dir):
    """Return (Client, application) for a server over pickle_dir."""
    app = server.build_application(pickle_dir=pickle_dir)
    return Client(app), app


def test_manifest_lists_bundles(pickle_dir):
    """read_manifest returns one entry per bundle named in the manifest."""
    entries = server.read_manifest(pickle_dir)
    assert [e['key'] for e in entries] == ['llr_moderate', 'null']


def test_manifest_skips_missing_bundle(pickle_dir):
    """A manifest entry with no file on disk is dropped."""
    manifest = json.loads((pickle_dir / server.MANIFEST_NAME).read_text())
    manifest['demos'].append({'key': 'gone', 'blurb': '', 'cache': '',
                              'source': 'wgn', 'size_bytes': 0})
    path = pickle_dir / server.MANIFEST_NAME
    original = path.read_text()
    path.write_text(json.dumps(manifest))
    try:
        assert 'gone' not in [e['key'] for e in server.read_manifest(
            pickle_dir)]
    finally:
        path.write_text(original)


def test_manifest_falls_back_to_listing(tmp_path, bundle):
    """With no manifest, the bundles on disk are listed unlabelled."""
    with gzip.open(tmp_path / 'solo.p.gz', 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    entries = server.read_manifest(tmp_path)
    assert [e['key'] for e in entries] == ['solo']


def test_missing_dir_serves_empty(tmp_path):
    """A build with no bundle dir still boots and says so."""
    app = server.build_application(pickle_dir=tmp_path / 'nothing')
    body = Client(app).get('/').get_data(as_text=True)
    assert 'no bundles baked' in body


def test_landing_lists_every_bundle(client):
    """The landing page names each bundle with a load link."""
    c, _ = client
    body = c.get('/').get_data(as_text=True)
    assert 'blurb for llr_moderate' in body
    assert '/load/llr_moderate' in body
    assert '/load/null' in body


def test_boot_unpickles_nothing(client):
    """Serving the landing page mounts no viewer."""
    c, app = client
    c.get('/')
    assert list(app.app.mounter.mounted_) == []


def test_load_mounts_and_serves_viewer(client):
    """Loading a bundle mounts a viewer and 302s to a working Dash app."""
    c, _ = client
    r = c.get('/load/llr_moderate')
    assert r.status_code == 302
    assert r.headers['Location'].endswith('/view/llr_moderate/')
    assert c.get('/view/llr_moderate/_dash-layout').status_code == 200


def test_load_is_idempotent(client):
    """Loading the same key twice reuses the one mount."""
    c, app = client
    c.get('/load/llr_moderate')
    c.get('/load/llr_moderate')
    assert list(app.app.mounter.mounted_) == ['llr_moderate']


def test_lru_evicts_past_the_cap(client):
    """Past max_mounts the least-recently-loaded viewer is unmounted."""
    c, app = client
    app.app.mounter.max_mounts = 1
    c.get('/load/llr_moderate')
    c.get('/load/null')
    assert list(app.app.mounter.mounted_) == ['null']
    assert '/view/llr_moderate' not in app.mounts


def test_unknown_key_404(client):
    """A key naming no bundle 404s rather than touching the disk."""
    c, _ = client
    assert c.get('/load/not-a-real-bundle').status_code == 404


def test_health(client):
    """The liveness probe answers without loading anything.

    /health, not /healthz: Cloud Run's frontend answers /healthz itself
    and never forwards it to the container.
    """
    c, _ = client
    assert c.get('/health').status_code == 200
    assert c.get('/healthz').status_code == 404


def test_boot_does_not_import_benchmark():
    """Importing the server reaches no benchmark module.

    deploy_hf.sh rsyncs glow/ into the Space with --exclude 'benchmark/',
    so anything the server imports at boot has to stay clear of
    glow._extra.benchmark or the Space fails to start. Run in a
    subprocess: the parent test session has already imported it.
    """
    code = ('import sys;'
            'from glow._extra.viewer.web import server;'
            "print([m for m in sys.modules"
            " if m.startswith('glow._extra.benchmark')])")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip().endswith('[]'), out.stdout


def test_direct_view_url_self_mounts(client):
    """A /view/<key>/ URL works even when nothing is mounted yet.

    The host scales to zero between visits, so a bookmarked or shared
    link routinely lands on a process that holds no mount.
    """
    c, app = client
    assert list(app.app.mounter.mounted_) == []
    r = c.get('/view/llr_moderate/')
    assert r.status_code == 302
    assert r.headers['Location'].endswith('/view/llr_moderate/')
    assert list(app.app.mounter.mounted_) == ['llr_moderate']
    assert c.get('/view/llr_moderate/_dash-layout').status_code == 200


def test_direct_view_url_unknown_key_404(client):
    """A /view/ URL naming no bundle 404s rather than mounting."""
    c, _ = client
    assert c.get('/view/not-a-real-bundle/').status_code == 404


def test_large_tree_is_capped(pickle_dir, monkeypatch):
    """A tree over max_regions is cut before the viewer is built.

    launch() resolves this ceiling for a local caller; the server has to
    do it itself, or a full-brain bundle scatters every region and the
    layout payload grows with it.
    """
    seen = {}
    real = server._resolve_min_vox

    def spy(ana, min_vox, max_regions):
        seen['max_regions'] = max_regions
        return real(ana, min_vox, max_regions)

    monkeypatch.setattr(server, '_resolve_min_vox', spy)
    app = server.build_application(pickle_dir=pickle_dir)
    app.app.mounter.max_regions = 4
    Client(app).get('/load/llr_moderate')
    assert seen['max_regions'] == 4


def test_dash_apps_compress_when_flask_compress_is_present():
    """Responses are gzipped, which is most of what the page weighs.

    Dash raises rather than degrades if compress=True without
    flask-compress, so app.py switches on presence; this pins that the
    switch reaches Dash instead of silently staying off.
    """
    from glow._extra.viewer import app as viewer_app
    if not viewer_app._HAS_COMPRESS:
        pytest.skip('flask-compress not installed')

    seen = {}
    real = viewer_app.Dash

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    viewer_app.Dash = spy
    try:
        exp = Experiment.from_gauss(b=1, num_img=20, shape=(5, 5, 5),
                                    seed=0, a=2)
        ana = AnalysisGLOW(n_perm_fwer=5, n_perm_inner=5).fit(exp)
        viewer_app._create_app(ana, exp)
    finally:
        viewer_app.Dash = real
    assert seen.get('compress') is True


def test_error_responses_are_not_cacheable(client):
    """A 404 is marked no-store so the CDN cannot pin it.

    Firebase Hosting caches a rewritten response with no Cache-Control
    for ten minutes, which would hold a 404 for a bundle that is merely
    unmounted long after the mount exists.
    """
    c, _ = client
    r = c.get('/load/not-a-real-bundle')
    assert r.status_code == 404
    assert r.headers['Cache-Control'] == 'no-store'


def test_self_redirect_is_not_cacheable(client):
    """The /view/ self-redirect is no-store, or it loops.

    It redirects to its own URL, so an edge-cached copy would answer the
    request that was supposed to reach the process and mount the bundle,
    sending the browser back to the same cached redirect.
    """
    c, _ = client
    r = c.get('/view/llr_moderate/')
    assert r.status_code == 302
    assert r.headers['Cache-Control'] == 'no-store'


def test_success_stays_cacheable(client):
    """A 2xx is left alone, so the edge can still serve it."""
    c, _ = client
    r = c.get('/')
    assert r.status_code == 200
    assert 'no-store' not in r.headers.get('Cache-Control', '')


def test_mounted_viewer_errors_are_not_cacheable(client):
    """A 404 from inside a mounted viewer is no-store too.

    The dispatcher hands these to the Dash app's own Flask server, which
    never sees the outer server's after_request hook. A missing asset is
    the error Dash does raise: it answers an unknown page route with the
    app index, to leave client-side routing to the callbacks.
    """
    c, _ = client
    c.get('/load/llr_moderate')
    r = c.get('/view/llr_moderate/assets/no-such-file.js')
    assert r.status_code == 404
    assert r.headers['Cache-Control'] == 'no-store'


def test_hcp_source_withholds_per_image(pickle_dir, monkeypatch):
    """A gated source is mounted with per_image off.

    The HCP data use terms let the derived maps be shared only with
    recipients bound by those same terms, which a public visitor is not,
    so the hosted viewer serves the group mean.
    """
    manifest = json.loads((pickle_dir / server.MANIFEST_NAME).read_text())
    for entry in manifest['demos']:
        entry['source'] = 'hcp'
    path = pickle_dir / server.MANIFEST_NAME
    original = path.read_text()
    path.write_text(json.dumps(manifest))

    seen = {}
    real = server._create_app

    def spy(*args, **kwargs):
        seen['per_image'] = kwargs.get('per_image')
        return real(*args, **kwargs)

    monkeypatch.setattr(server, '_create_app', spy)
    try:
        app = server.build_application(pickle_dir=pickle_dir)
        Client(app).get('/load/llr_moderate')
    finally:
        path.write_text(original)
    assert seen['per_image'] is False


def test_wgn_source_keeps_per_image(client):
    """An ungated source still offers the individual images."""
    c, app = client
    assert 'wgn' not in server.GATED_PER_IMAGE_SOURCES
    c.get('/load/llr_moderate')
    layout = c.get('/view/llr_moderate/_dash-layout').get_data(as_text=True)
    assert 'mean only' not in layout


def test_per_image_off_refuses_any_image_index():
    """Asking for a subject's volume by index still resolves to the mean.

    The dropdown is only a control; its value reaches the server from the
    client, so disabling it withholds nothing on its own. This pins the
    refusal in the resolver both background callbacks go through.
    """
    from glow._extra.viewer.app import _resolve_image_idx

    for val in ('0', '7', 0, 7, 'mean', None):
        assert _resolve_image_idx(val, per_image=False) is None, val

    assert _resolve_image_idx('mean', per_image=True) is None
    assert _resolve_image_idx(None, per_image=True) is None
    assert _resolve_image_idx('7', per_image=True) == 7
