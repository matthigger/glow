from glow.vba import *


def test_apply_tfce():
    # get random 5x5x5 image
    np.random.seed(0)
    x = np.random.normal(size=(5, 5, 5))

    # run tfce
    apply_tfce_img(x)


def test_single_cluster_f1_1():
    mask_true = np.zeros((4, 4), dtype=int)
    mask_true[:2, :2] = 1
    mask_est = mask_true.copy()

    thresh, mask_est_prune, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh == 4
    assert np.allclose(mask_est_prune, mask_true)


def test_single_cluster_f1_0():
    mask_true = np.zeros((4, 4), dtype=int)
    mask_true[:2, :2] = 1
    mask_est = np.zeros((4, 4), dtype=int)
    mask_est[-2:, -2:] = 1

    thresh, mask_est_prune, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh == np.inf
    assert np.allclose(mask_est_prune, np.zeros((4, 4), dtype=bool))


def test_single_cluster_no_overlap():
    mask_true = np.zeros((4, 4), dtype=int)
    mask_true[0, 0] = 1
    mask_est = np.ones((4, 4), dtype=int)

    thresh, mask_est_prune, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh > 0
    assert np.allclose(mask_est_prune, mask_est)


def test_two_clusters_small_one_matches():
    mask_true = np.zeros((6, 6), dtype=int)
    mask_true[:1, :] = 1

    mask_est = np.zeros((6, 6), dtype=int)
    mask_est[:1, :] = 1  # cluster 1 (true overlap)
    mask_est[-2:, :] = 2  # cluster 2 (false positive)

    thresh, mask_est_prune, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh == 6
    assert np.allclose(mask_est_prune, mask_est.astype(bool))
    assert 0 < f1 < 1

def test_two_clusters_big_one_matches():
    mask_true = np.zeros((6, 6), dtype=int)
    mask_true[-2:, :] = 1

    mask_est = np.zeros((6, 6), dtype=int)
    mask_est[:1, :] = 1  # cluster 1 (true overlap)
    mask_est[-2:, :] = 2  # cluster 2 (false positive)

    thresh, mask_est_prune, f1 = optimize_cluster_thresh(mask_est,
                                                         mask_true)

    assert thresh == 12
    assert np.allclose(mask_est_prune, mask_est == 2)
    assert np.isclose(f1, 1)


def test_mask_active_limits_analysis():
    mask_true   = np.array([1, 1, 1, 1, 0, 0, 0, 0]).astype(bool)
    mask_est    = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    mask_active = np.array([1, 1, 1, 1, 0, 0, 0, 0]).astype(bool)

    thresh, mask_est_prune, f1 = optimize_cluster_thresh(mask_est, mask_true, mask_active)

    assert thresh == 4
    assert np.allclose(mask_est_prune, mask_active)
    assert f1 == 1


def test_same_size_clusters():
    mask_true   = np.array([1, 1, 0, 0, 0, 0, 0, 0]).astype(bool)
    mask_est    = np.array([1, 1, 2, 2, 0, 0, 0, 0])

    thresh, mask_est_prune, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh == 2
    assert np.allclose(mask_est_prune, mask_est.astype(bool))
