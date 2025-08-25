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

    thresh, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh == 4


def test_single_cluster_f1_0():
    mask_true = np.zeros((4, 4), dtype=int)
    mask_true[:2, :2] = 1
    mask_est = np.zeros((4, 4), dtype=int)
    mask_est[-2:, -2:] = 1

    thresh, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh == np.inf


def test_single_cluster_no_overlap():
    mask_true = np.zeros((4, 4), dtype=int)
    mask_true[0, 0] = 1
    mask_est = np.ones((4, 4), dtype=int)

    thresh, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh > 0


def test_two_clusters_small_one_matches():
    mask_true = np.zeros((6, 6), dtype=int)
    mask_true[:1, :] = 1

    mask_est = np.zeros((6, 6), dtype=int)
    mask_est[:1, :] = 1  # cluster 1 (true overlap)
    mask_est[-2:, :] = 2  # cluster 2 (false positive)

    thresh, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh == 6
    assert 0 < f1 < 1

def test_two_clusters_big_one_matches():
    mask_true = np.zeros((6, 6), dtype=int)
    mask_true[-2:, :] = 1

    mask_est = np.zeros((6, 6), dtype=int)
    mask_est[:1, :] = 1  # cluster 1 (true overlap)
    mask_est[-2:, :] = 2  # cluster 2 (false positive)

    thresh, f1 = optimize_cluster_thresh(mask_est,
                                                         mask_true)

    assert thresh == 12
    assert np.isclose(f1, 1)


def test_mask_active_limits_analysis():
    mask_true   = np.array([1, 1, 1, 1, 0, 0, 0, 0]).astype(bool)
    mask_est    = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    mask_active = np.array([1, 1, 1, 1, 0, 0, 0, 0]).astype(bool)

    thresh, f1 = optimize_cluster_thresh(mask_est, mask_true, mask_active)

    assert thresh == 4
    assert f1 == 1


def test_same_size_clusters():
    mask_true   = np.array([1, 1, 0, 0, 0, 0, 0, 0]).astype(bool)
    mask_est    = np.array([1, 1, 2, 2, 0, 0, 0, 0])

    thresh, f1 = optimize_cluster_thresh(mask_est, mask_true)

    assert thresh == 2
