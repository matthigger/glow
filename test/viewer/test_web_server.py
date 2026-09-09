"""Tests for the hosted multi-demo web server.

Covers the manifest read and the request flow the Space runs on: landing
page -> load -> dynamic mount -> redirect -> a working Dash app, plus the
LRU that bounds how many viewers stay resident. The bundle is fit with
AnalysisGLOW, the per-perm arm the baked set ships, so the hosted path is
exercised on the arm it actually serves.
"""

import gzip
import inspect
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


def test_gated_source_serves_per_image_once_accepted(pickle_dir,
                                                     monkeypatch):
    """Nothing is withheld behind the terms screen.

    The screen is the condition the HCP redistribution clause sets, so a
    visitor past it gets the individual images like any other viewer.
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
        seen['per_image'] = kwargs.get('per_image', True)
        seen['source'] = kwargs.get('source')
        return real(*args, **kwargs)

    monkeypatch.setattr(server, '_create_app', spy)
    try:
        app = server.build_application(pickle_dir=pickle_dir)
        c = Client(app)
        c.post('/terms', data={'next': '/'})
        c.get('/load/llr_moderate')
    finally:
        path.write_text(original)
    assert seen['per_image'] is True
    assert seen['source'] == 'hcp'


def test_wgn_source_keeps_per_image(client):
    """An ungated source still offers the individual images."""
    c, app = client
    assert 'wgn' not in server.GATED_SOURCES
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


@pytest.fixture
def picker_dir(tmp_path, bundle):
    """Write a three-bundle set whose manifest carries picker parameters.

    The moderate cell is filed under two figures at once, which is what
    the hub cell does in the real baked set.
    """
    entries = []
    for key, caches, params in (
            ('llr_weak', ['sweep_llr'], dict(effect_llr=0.003)),
            ('llr_moderate', ['sweep_llr', 'prune'],
             dict(effect_llr=0.03, prune_rule='greedy')),
            ('prune_dp', ['prune'], dict(prune_rule='dp'))):
        with gzip.open(tmp_path / f'{key}.p.gz', 'wb') as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        full = dict(source='wgn', b=1, num_img=20, num_vox=125,
                    effect_llr=0.03, n_vox_frac=0.1, cluster_mode='FOCUS',
                    prune_rule='greedy', keep_stat=False)
        full.update(params)
        entries.append({
            'key': key, 'blurb': f'blurb for {key}', 'cache': caches[0],
            'caches': caches, 'source': 'wgn', 'params': full,
            'size_bytes': (tmp_path / f'{key}.p.gz').stat().st_size})
    (tmp_path / server.MANIFEST_NAME).write_text(
        json.dumps({'demos': entries}))
    return tmp_path


def test_landing_builds_a_picker_from_params(picker_dir):
    """A manifest carrying params renders the figure picker."""
    c = Client(server.build_application(pickle_dir=picker_dir))
    body = c.get('/').get_data(as_text=True)
    assert '<select id="figure">' in body
    assert 'value="sweep_llr"' in body
    assert 'value="prune"' in body


def test_picker_files_the_hub_cell_under_every_figure(picker_dir):
    """A cell on several axes is offered by each figure it sits on.

    The moderate fit is the point the prune sweep varies the rule away
    from, so a picker that filed it under sweep_llr alone would offer
    the prune figure no greedy option at all.
    """
    manifest = server.read_manifest(picker_dir)
    demos = server._picker_demos(manifest)
    hub = [d for d in demos if d['key'] == 'llr_moderate'][0]
    assert set(hub['caches']) == {'sweep_llr', 'prune'}

    for cache in ('sweep_llr', 'prune'):
        reachable = [d['key'] for d in demos if cache in d['caches']]
        assert 'llr_moderate' in reachable


def test_figure_list_orders_by_paper_figure(picker_dir):
    """Figures come out in the paper's numbering, not manifest order."""
    figures = server._figure_list(server.read_manifest(picker_dir))
    assert [f['cache'] for f in figures] == ['prune', 'sweep_llr']
    assert all(f['heading'] and f['blurb'] for f in figures)


def test_figure_label_carries_the_paper_index(picker_dir):
    """A cache the paper reports is labelled with its figure number."""
    figures = server._figure_list(server.read_manifest(picker_dir))
    label = {f['cache']: f['label'] for f in figures}
    assert label['prune'].startswith('Figure 8 --')
    assert label['sweep_llr'].startswith('Figure 10 --')
    assert 'sweep_llr' in label['sweep_llr']


def test_figure_label_says_when_it_is_not_in_the_paper():
    """A catalogue cache no figure reports says so, rather than looking
    like an omission. sweep_extent is one: CONFIG builds it and the
    paper's figures do not read it."""
    manifest = [{'key': 'k', 'cache': 'sweep_extent', 'size_bytes': 1,
                 'params': {}}]
    label = server._figure_list(manifest)[0]['label']
    assert label.endswith('not in the paper')
    assert 'Figure' not in label


def test_shortcuts_skip_what_was_not_baked():
    """A shortcut naming an absent bundle is dropped, not shown broken."""
    manifest = [{'key': 'hcp_llr_strong', 'source': 'hcp'}]
    shortcuts = server._shortcuts_html(manifest)
    assert '/load/hcp_llr_strong"' in shortcuts
    assert 'hcp_null' not in shortcuts
    assert 'hcp_vox_full_brain' not in shortcuts


def test_shortcuts_absent_when_none_are_baked():
    """A set holding none of the shortcut bundles renders no row."""
    assert server._shortcuts_html([{'key': 'nothing_named_this'}]) == ''


def test_seed_is_a_picker_axis():
    """The seed a bundle was built at is offered like any other axis."""
    assert 'seed' in [name for name, _, _ in server._PARAM_SPEC]


def test_picker_keeps_a_noscript_link_to_every_bundle(picker_dir):
    """Every bundle stays reachable without javascript."""
    c = Client(server.build_application(pickle_dir=picker_dir))
    body = c.get('/').get_data(as_text=True)
    assert '<noscript>' in body
    for key in ('llr_weak', 'llr_moderate', 'prune_dp'):
        assert f'/load/{key}' in body


def test_landing_falls_back_when_params_are_absent(client):
    """A manifest with no params renders the plain list, no picker."""
    c, _ = client
    body = c.get('/').get_data(as_text=True)
    assert '<select id="figure">' not in body
    assert '/load/llr_moderate' in body


def test_json_for_script_cannot_close_the_script_element(picker_dir):
    """An embedded value carrying </script> is neutralised."""
    blob = server._json_for_script({'k': '</script><b>'})
    assert '</script>' not in blob
    assert '<\\/script>' in blob


def test_value_order_covers_every_labelled_code():
    """An ordered parameter lists every code it has a label for.

    The script ranks by index, so a code the order forgot would fall to
    the end -- correct, but silent. Nothing should reach that path.
    """
    for name, order in server._VALUE_ORDER.items():
        labelled = set(server._VALUE_LABELS.get(name, {}))
        assert labelled.issubset(set(order)), name


def test_reported_arm_leads_its_dropdown():
    """Each ordered axis opens on the value the paper reports."""
    assert server._VALUE_ORDER['cluster_mode'][0] == 'FOCUS'
    assert server._VALUE_ORDER['prune_rule'][0] == 'greedy'


def test_shortcuts_offer_hcp_only():
    """The synthetic cells are reached through the picker, not here.

    Someone arriving cold wants to see the method on real images; the
    WGN cells answer a different question and stay one dropdown away.
    """
    manifest = [{'key': 'llr_strong', 'source': 'wgn'},
                {'key': 'hcp_llr_strong', 'source': 'hcp'}]
    out = server._shortcuts_html(manifest)
    assert 'Real diffusion maps (HCP)' in out
    assert '/load/hcp_llr_strong' in out
    assert '/load/llr_strong"' not in out


def test_shortcuts_drop_an_empty_group(picker_dir):
    """A WGN-only build shows no shortcut row at all."""
    assert server._shortcuts_html(server.read_manifest(picker_dir)) == ''


def test_picker_carries_the_stats():
    """A bundle's scores reach the picker's script as its s field."""
    manifest = [{'key': 'k', 'cache': 'sweep_llr', 'size_bytes': 1,
                 'params': {}, 'stats': {'n_pred': 1, 'dice': 0.5}}]
    assert server._picker_demos(manifest)[0]['s'] == {'n_pred': 1,
                                                      'dice': 0.5}


def test_picker_tolerates_a_bundle_with_no_stats():
    """An unscored bundle gets an empty stats dict, not a KeyError."""
    manifest = [{'key': 'k', 'cache': 'sweep_llr', 'size_bytes': 1,
                 'params': {}}]
    assert server._picker_demos(manifest)[0]['s'] == {}


def test_demo_stats_scores_a_planted_fit(bundle):
    """demo_stats reports the figures' own metrics for a planted cell."""
    bake = pytest.importorskip('glow._extra.viewer.web.bake_demos')
    stats = bake.demo_stats(bundle['ana'], bundle['mask_target'],
                            bundle['exp'].mask_idx > -1)
    assert set(stats) >= {'n_pred', 'min_pval', 'dice', 'sens', 'ppv',
                          'hom', 'com'}
    for key in ('dice', 'sens', 'ppv', 'hom', 'com'):
        assert stats[key] is None or 0.0 <= stats[key] <= 1.0


def test_demo_stats_omits_overlap_on_the_null_path(bundle):
    """With nothing planted there is no overlap worth reporting.

    Reporting dice 0 there would read as a failure to find something,
    when there was nothing to find.
    """
    bake = pytest.importorskip('glow._extra.viewer.web.bake_demos')
    stats = bake.demo_stats(bundle['ana'], None,
                            bundle['exp'].mask_idx > -1)
    assert 'dice' not in stats
    assert 'hom' not in stats
    assert stats['n_pred'] >= 0


def test_detail_panel_names_the_image_source(bundle):
    """The experiment detail panel says which dataset the images are."""
    from glow._extra.viewer import layout
    rendered = str(layout._detail_panels(bundle['ana'], bundle['exp'],
                                         source='hcp'))
    assert 'image source' in rendered
    assert layout.SOURCE_LABELS['hcp'] in rendered


def test_detail_panel_omits_an_unknown_source(bundle):
    """An unlabelled experiment gets no row, rather than one reading None."""
    from glow._extra.viewer import layout
    rendered = str(layout._detail_panels(bundle['ana'], bundle['exp']))
    assert 'image source' not in rendered


def test_every_shortcut_is_a_gated_source():
    """Only real-data entry points are offered up front."""
    keys = [key for _, entries in server._SHORTCUTS for key, _ in entries]
    assert keys
    assert all(k.startswith('hcp_') for k in keys)


def test_hcp_group_offers_the_volume_shortcuts():
    """The histogram and whole-brain entry points exist for HCP too."""
    manifest = [{'key': 'hcp_vox_1k'}, {'key': 'hcp_vox_full_brain'}]
    out = server._shortcuts_html(manifest)
    assert '/load/hcp_vox_1k' in out
    assert '/load/hcp_vox_full_brain' in out
    assert 'Synthetic images (WGN)' not in out


@pytest.fixture
def gated_dir(tmp_path, bundle):
    """A one-bundle set whose manifest calls its source gated."""
    with gzip.open(tmp_path / 'hcp_llr_moderate.p.gz', 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    (tmp_path / server.MANIFEST_NAME).write_text(json.dumps({'demos': [
        {'key': 'hcp_llr_moderate', 'blurb': 'gated', 'cache': 'sweep_llr',
         'source': 'hcp', 'size_bytes': 1, 'params': {'source': 'hcp'},
         'stats': {'n_pred': 0}}]}))
    return tmp_path


def test_gated_load_asks_for_the_terms_first(gated_dir):
    """A gated bundle is not even unpickled before its terms are met."""
    app = server.build_application(pickle_dir=gated_dir)
    response = Client(app).get('/load/hcp_llr_moderate')
    assert response.status_code == 302
    assert '/terms' in response.headers['Location']
    assert list(app.app.mounter.mounted_) == []


def test_gated_load_proceeds_once_accepted(gated_dir):
    """Accepting lets the same request through to the viewer."""
    c = Client(server.build_application(pickle_dir=gated_dir))
    c.post('/terms', data={'next': '/'})
    response = c.get('/load/hcp_llr_moderate')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/view/hcp_llr_moderate/')


def test_ungated_load_needs_no_terms(client):
    """A synthetic bundle is nobody's data and asks for nothing."""
    c, _ = client
    response = c.get('/load/llr_moderate')
    assert response.status_code == 302
    assert '/terms' not in response.headers['Location']


def test_terms_post_records_the_acceptance(gated_dir):
    """Accepting sets the cookie and returns the visitor where they were."""
    c = Client(server.build_application(pickle_dir=gated_dir))
    response = c.post('/terms', data={'next': '/load/hcp_llr_moderate'})
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/load/hcp_llr_moderate')
    assert server.TERMS_COOKIE in response.headers.get('Set-Cookie', '')


def test_terms_refuses_to_redirect_off_site():
    """next arrives from the query string, so a crafted link could
    otherwise carry a visitor elsewhere through our own redirect."""
    for bad in ('https://example.com/x', '//example.com/x', 'evil', ''):
        assert server._safe_next(bad) == '/'
    assert server._safe_next('/load/k') == '/load/k'


def test_a_mounted_gated_viewer_still_asks(gated_dir):
    """The check has to live on the mount, not only on the route.

    The dispatcher hands a mounted viewer the request before the outer
    Flask app sees it, so a guard on /view alone would fire on a cold
    mount and never again.
    """
    app = server.build_application(pickle_dir=gated_dir)
    accepted = Client(app)
    accepted.post('/terms', data={'next': '/'})
    accepted.get('/load/hcp_llr_moderate')
    assert 'hcp_llr_moderate' in app.app.mounter.mounted_

    stranger = Client(app)
    response = stranger.get('/view/hcp_llr_moderate/')
    assert response.status_code == 302
    assert '/terms' in response.headers['Location']


def test_gated_viewer_is_never_shared_cached(gated_dir):
    """Its responses turn on a cookie the CDN does not vary by."""
    c = Client(server.build_application(pickle_dir=gated_dir))
    c.post('/terms', data={'next': '/'})
    c.get('/load/hcp_llr_moderate')
    response = c.get('/view/hcp_llr_moderate/')
    assert response.status_code == 200
    assert response.headers['Cache-Control'] == 'no-store'


def test_terms_page_links_the_consortium_document(gated_dir):
    """A reader accepts the HCP document, not this project's summary."""
    c = Client(server.build_application(pickle_dir=gated_dir))
    body = c.get('/terms?next=/load/hcp_llr_moderate').get_data(as_text=True)
    assert server.HCP_TERMS_URL in body
    assert body.count('humanconnectome.org') == 1
    assert '/load/hcp_llr_moderate' in body

    # one source of truth: the page points at the document and does not
    # paraphrase it, so our wording cannot drift from the consortium's
    for restated in ('identify or contact', 'acknowledge', 'cite'):
        assert restated not in body


def test_landing_says_the_terms_are_coming(gated_dir):
    """A reader learns the terms are ahead before they click, not after."""
    c = Client(server.build_application(pickle_dir=gated_dir))
    body = c.get('/').get_data(as_text=True)
    assert 'Data Use Terms' in body
    assert server.HCP_TERMS_URL in body


def test_landing_stays_quiet_for_an_ungated_set(picker_dir):
    """A synthetic-only build mentions no terms at all."""
    c = Client(server.build_application(pickle_dir=picker_dir))
    body = c.get('/').get_data(as_text=True)
    assert 'Data Use Terms' not in body


def test_the_cookie_is_named_for_what_the_cdn_forwards():
    """Firebase Hosting strips every request cookie except __session.

    Under any other name the cookie never reaches Cloud Run, so an
    accepted visitor is asked again on every click -- which is what
    happened, and only showed up through the CDN, never against the
    origin.
    """
    assert server.TERMS_COOKIE == '__session'


def test_the_mandrill_set_is_not_gated():
    """The terms page promises this set is reachable without accepting.

    It says a visitor can see the Gaussian noise and mandrill examples
    without agreeing to anything, so gating either one would make the
    page lie about what it is asking for.
    """
    from glow._extra.viewer import web
    assert 'mandrill' not in web.GATED_SOURCES
    assert 'wgn' not in web.GATED_SOURCES


def test_mandrill_mirrors_the_strength_sweep():
    """The photograph set differs from sweep_llr in its images alone."""
    bake = pytest.importorskip('glow._extra.viewer.web.bake_demos')
    llr = {d.key: d for d in bake.DEMOS if d.cache == 'sweep_llr'
           and d.source == 'wgn'}
    mand = {d.key.removeprefix('mandrill_'): d for d in bake.DEMOS
            if d.cache == 'mandrill'}
    assert mand and set(mand) == set(llr)
    for key, demo in mand.items():
        assert demo.source == 'mandrill'
        assert demo.effect == llr[key].effect
        assert demo.ana == llr[key].ana


def test_mandrill_params_describe_the_photograph():
    """Its axes report the image's own shape, not a benchmark cell's."""
    bake = pytest.importorskip('glow._extra.viewer.web.bake_demos')
    p = bake.demo_params(bake._DEMO_BY_KEY['mandrill_llr_moderate'], 0)
    wgn = bake.demo_params(bake._DEMO_BY_KEY['llr_moderate'], 0)
    assert p['b'] == len(bake.MANDRILL_CHANNELS)
    assert p['num_vox'] == bake.mandrill_num_vox()
    assert p['num_img'] == wgn['num_img']
    assert p['effect_llr'] == wgn['effect_llr']


def test_every_baked_cache_has_a_heading():
    """A set with no heading falls back to its bare cache name."""
    bake = pytest.importorskip('glow._extra.viewer.web.bake_demos')
    caches = {c for d in bake.DEMOS for c in (d.cache, *d.also)}
    assert caches <= set(server._CACHE_HEADINGS)
    assert caches <= set(server._CACHE_BLURBS)


def test_the_top_dropdown_is_not_only_paper_figures():
    """It carries sets the paper never reports, so it says so."""
    out = server._picker_html([{'cache': 'mandrill', 'heading': 'x',
                                'blurb': '', 'label': 'x'}])
    assert 'Analysis set' in out
    assert 'Paper figure' not in out


def test_terms_page_names_the_cohort(gated_dir):
    """A reader should know which HCP release these maps come from."""
    c = Client(server.build_application(pickle_dir=gated_dir))
    body = c.get('/terms').get_data(as_text=True)
    assert 'Open Access' in body
    assert '100 unrelated' in body


def test_mandrill_images_are_numbered():
    """Every draw is the same photograph, so the labels need indices.

    bootstrap_img carries the one source label through to every image,
    which leaves the viewer's picker offering num_img entries a reader
    cannot tell apart.
    """
    bake = pytest.importorskip('glow._extra.viewer.web.bake_demos')
    names = bake.mandrill_subjects()
    assert len(names) == bake.MANDRILL_NUM_IMG
    assert len(set(names)) == len(names)
    assert all(n.startswith('mandrill_') for n in names)


def test_call_uncached_reaches_past_the_dispatchers():
    """data_factory and effect_factory dispatch; they are not wrapped.

    Unwrapping a dispatcher is a no-op and the builder it selects still
    carries @MEMORY.cache(ignore=['exp']) -- keyed on parent_uid with
    the experiment ignored -- so an unresolved dispatch answers a
    changed cohort out of the cache built for the old one.
    """
    bake = pytest.importorskip('glow._extra.viewer.web.bake_demos')
    data = pytest.importorskip('glow._extra.benchmark.data')

    assert set(bake._DISPATCH) == {data.data_factory, data.effect_factory}
    for dispatcher, (_, table, _default) in bake._DISPATCH.items():
        assert inspect.unwrap(dispatcher) is dispatcher
        assert table
        for builder in table.values():
            assert inspect.unwrap(builder) is not builder


def test_dash_subpaths_survive_a_lost_mount(client):
    """A viewer's own URLs must work when the mount is gone.

    Only /view/<key>/ used to self-mount, so once the LRU evicted a
    viewer -- or the worker recycled, or the host scaled to zero -- an
    open page kept drawing plotly's client-side hover text while every
    callback POST 404'd behind it.
    """
    c, app = client
    c.get('/load/llr_moderate')
    assert list(app.app.mounter.mounted_) == ['llr_moderate']

    app.app.mounter.mounted_.clear()
    app.mounts.clear()

    assert c.get('/view/llr_moderate/_dash-dependencies').status_code == 200
    assert list(app.app.mounter.mounted_) == ['llr_moderate']


def test_a_callback_post_is_never_redirected(client):
    """A redirected POST is retried as a GET, losing the callback."""
    c, app = client
    c.get('/load/llr_moderate')
    app.app.mounter.mounted_.clear()
    app.mounts.clear()

    r = c.post('/view/llr_moderate/_dash-update-component',
               json={'output': 'nope.figure', 'outputs': [], 'inputs': [],
                     'changedPropIds': []})
    assert r.status_code not in (301, 302, 307, 308, 404, 405)


def test_view_subpath_unknown_key_404(client):
    """A sub-path of a bundle this build never baked is still a 404."""
    c, _ = client
    assert c.get('/view/nope/_dash-layout').status_code == 404


def test_gated_subpath_refuses_before_the_terms(gated_dir):
    """The gate covers a viewer's callbacks, not just its page."""
    c = Client(server.build_application(pickle_dir=gated_dir))
    r = c.get('/view/hcp_llr_moderate/_dash-layout')
    assert r.status_code == 403
