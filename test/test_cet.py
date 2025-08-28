from collections import Counter
from collections import namedtuple

from scipy.ndimage import label

from glow.cet import *
from glow.mask import conn_dict


def test_align_curves():
    curves = [(np.array([0, 2, 4, 5]),
               np.array([10, 20, 40, 50])),
              (np.array([1, 2, 4]),
               np.array([5, 25, 45]))
              ]
    x_exp = np.array([0, 1, 2, 4, 5])
    y0_exp = np.array([10, 10, 20, 40, 50])
    y1_exp = np.array([np.nan, 5, 25, 45, 45])

    x, y = align_curves(curves)
    assert np.array_equal(x, x_exp)
    assert np.array_equal(y[0, :], y0_exp, equal_nan=True)
    assert np.array_equal(y[1, :], y1_exp, equal_nan=True)


def test_align_curves_random():
    rng = np.random.default_rng(0)

    # generate two random curves of different lengths
    x1 = np.sort(rng.choice(100, size=10, replace=False))
    y1 = rng.integers(0, 50, size=10)
    x2 = np.sort(rng.choice(100, size=15, replace=False))
    y2 = rng.integers(50, 100, size=15)

    curves = [(x1, y1), (x2, y2)]
    x, y = align_curves(curves)

    for x_orig, y_orig, row in zip([x1, x2], [y1, y2], y):
        # everything before first observed x must be NaN
        first_idx = np.where(x == x_orig[0])[0][0]
        assert np.all(np.isnan(row[:first_idx]))

        # check fidelity of y's, must carry last seen value forward
        for j in range(first_idx + 1, len(x)):
            idx_orig = bisect_left(x_orig, x[j])
            if (idx_orig >= len(x_orig)) or (x[j] != x_orig[idx_orig]):
                idx_orig -= 1
            assert row[j] == y_orig[idx_orig]


def test_iter_label_map_thresh():
    # approach: given a random image, run through all thresholds and ensure
    # same label map is given (up to region index re-mappings)
    Case = namedtuple('Case', ['shape', 'seed', 'p_mask', 'conn'])
    conn2d = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    conn3d = conn_dict[26]
    case_list = [Case(seed=0, shape=(3, 3), p_mask=1, conn=conn2d),
                 Case(seed=0, shape=(3, 3), p_mask=.6, conn=conn2d),
                 Case(seed=0, shape=(5, 5, 5), p_mask=1, conn=conn3d),
                 Case(seed=0, shape=(5, 5, 5), p_mask=.6, conn=conn3d)]

    for case_idx, case in enumerate(case_list):
        # generate image
        rng = np.random.default_rng(seed=case.seed)
        img = rng.standard_normal(size=case.shape)
        mask_active = rng.random(size=case.shape) <= case.p_mask

        _iter = iter_label_map_thresh(img,
                                      conn=case.conn,
                                      mask_active=mask_active)

        thresh = np.inf
        for iter_idx, (ijk, label_map, reg_size) in enumerate(_iter):
            # ensure we're decreasing in ijk
            assert thresh > img[*ijk]
            thresh = img[*ijk]

            mask = (img >= thresh) & mask_active
            label_map_exp = label(mask, structure=case.conn)[0]
            reg_size_exp = Counter(label_map_exp.ravel())
            if 0 in reg_size_exp.keys():
                del reg_size_exp[0]

            # ensure label maps align (labels themselves may differ)
            assert np.array_equal(label_map > 0, label_map_exp > 0)

            # reg_size may have different labels, so we just test on values
            assert sorted(reg_size.values()) == sorted(reg_size_exp.values())