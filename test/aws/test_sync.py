"""sync: the HCP bundle S3 layout (stage dir + selective per-feature keys)."""

from glow._extra.aws import sync
from glow._extra.benchmark import hcp


def test_hcp_bundle_pair_is_whole_dir():
    local_dir, key = sync.hcp_bundle_pair('glow')
    assert local_dir == hcp.bundle_dir()
    assert key == 'glow/hcp_bundle'


def test_hcp_bundle_keys_are_per_feature():
    feats = ('fa', 'mk')
    pairs = sync.hcp_bundle_keys('glow', feats)
    keys = [k for k, _ in pairs]
    # shared mask/affine/meta + one per feature, in pull order
    assert keys == ['glow/hcp_bundle/mask.npy',
                    'glow/hcp_bundle/affine.npy',
                    'glow/hcp_bundle/meta.json',
                    'glow/hcp_bundle/feat/fa.npy',
                    'glow/hcp_bundle/feat/mk.npy']
    # local paths line up with hcp's bundle layout
    fa_key = 'glow/hcp_bundle/feat/fa.npy'
    assert dict(pairs)[fa_key] == hcp.bundle_feat_path('fa')


def test_hcp_bundle_keys_match_staging_layout():
    # the worker's pull keys must be exactly what stage_hcp's upload produces:
    # {prefix}/hcp_bundle/<path relative to bundle_dir>
    _, base = sync.hcp_bundle_pair('p')
    for key, local in sync.hcp_bundle_keys('p', ('fa',)):
        rel = local.relative_to(hcp.bundle_dir()).as_posix()
        assert key == f'{base}/{rel}'
