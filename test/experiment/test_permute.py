from hrba.experiment.permute import *


def test_get_perm_matrix():
    perm = get_perm_matrix(num_img=5, seed=0)
    np.testing.assert_array_almost_equal(perm, np.eye(5))

    perm = get_perm_matrix(num_img=5, seed=1)
    perm_expect = np.array([[0., 0., 0., 0., 1.],
                            [1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.]])

    np.testing.assert_array_almost_equal(perm, perm_expect)
