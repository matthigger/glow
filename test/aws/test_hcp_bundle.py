"""The HCP npy bundle: glue correctness and hash-equivalence to from_search.

The glue (hcp.build_exp_img_from_bundle) is the single HCP build path, used
locally and on an AWS worker. The synthetic test pins its mechanics without
any HCP data; the guarded test proves the headline property -- an experiment
built from the bundle hashes identically to one built from the niftis -- on a
box that has the data (skips otherwise).
"""

import json

import joblib
import numpy as np
import pytest

from glow._extra.benchmark import hcp


def _write_bundle(bundle_dir, feats, num_img, mask):
    """Write a minimal bundle (mask / affine / meta + per-feature arrays)."""
    (bundle_dir / 'feat').mkdir(parents=True, exist_ok=True)
    np.save(bundle_dir / 'mask.npy', mask)
    np.save(bundle_dir / 'affine.npy', np.eye(4))
    (bundle_dir / 'meta.json').write_text(
        json.dumps({'subjects': [f's{i}' for i in range(num_img)]}))
    num_vox = int(mask.sum())
    arrays = {}
    for fi, feat in enumerate(feats):
        arr = (np.arange(num_img * num_vox, dtype=np.float32)
               .reshape(num_img, num_vox) + fi * 100)
        np.save(bundle_dir / 'feat' / f'{feat}.npy', arr)
        arrays[feat] = arr
    return arrays


def test_glue_builds_expected_exp(monkeypatch, tmp_path):
    monkeypatch.setattr(hcp, 'bundle_dir', lambda: tmp_path)
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 1] = mask[0, 1, 0] = True  # 3 in-brain voxels
    arrays = _write_bundle(tmp_path, ('fa', 'mk'), num_img=4, mask=mask)

    exp_img = hcp.build_exp_img_from_bundle(('fa', 'mk'))

    assert exp_img.y.shape == (2, 4, 3)            # (b, num_img, num_vox)
    assert exp_img.y.dtype == np.float32
    # features stacked in the requested order
    np.testing.assert_array_equal(exp_img.y[0], arrays['fa'])
    np.testing.assert_array_equal(exp_img.y[1], arrays['mk'])
    # mask_idx is get_mask_idx(mask); meta mirrors from_paths
    import glow.mask
    np.testing.assert_array_equal(
        exp_img.mask_idx, glow.mask.get_mask_idx(mask))
    assert exp_img.meta['features'] == ['fa', 'mk']
    assert exp_img.meta['subjects'] == ['s0', 's1', 's2', 's3']


def test_feature_order_follows_request(monkeypatch, tmp_path):
    monkeypatch.setattr(hcp, 'bundle_dir', lambda: tmp_path)
    mask = np.ones((1, 1, 2), dtype=bool)
    arrays = _write_bundle(tmp_path, ('fa', 'mk', 'od'), num_img=2, mask=mask)
    exp_img = hcp.build_exp_img_from_bundle(('od', 'fa'))
    assert exp_img.meta['features'] == ['od', 'fa']
    np.testing.assert_array_equal(exp_img.y[0], arrays['od'])
    np.testing.assert_array_equal(exp_img.y[1], arrays['fa'])


@pytest.mark.skipif(not hcp.is_present(),
                    reason='HCP niftis not present (set XDG_DATA_HOME)')
def test_bundle_exp_hashes_like_from_search():
    # the headline property: same data -> same hash. Build a 2-feature exp via
    # from_search (niftis) and via the bundle; their joblib.hash must match.
    from glow.experiment import ExperimentImageOnly
    feats = ('fa', 'mk')

    folder = hcp.ensure_hcp_data()
    from_nii = ExperimentImageOnly.from_search(
        folder=folder, sbj_regex=hcp.SBJ_REGEX,
        img_glob_dict={f: hcp.IMG_GLOB_DICT[f] for f in feats},
        mask=next(folder.glob(hcp.MASK_GLOB)))

    from_bundle = hcp.build_exp_img_from_bundle(feats)

    assert joblib.hash(from_bundle.y) == joblib.hash(from_nii.y)
    assert joblib.hash(from_bundle.mask_idx) == joblib.hash(from_nii.mask_idx)
    assert joblib.hash(from_bundle) == joblib.hash(from_nii)
