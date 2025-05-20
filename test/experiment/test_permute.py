from hglm.experiment.exper import Experiment
from hglm.experiment.permute import *


class TestPermuter:
    def test_get_perm_matrix(self):
        perm = get_perm_matrix(num_img=5, seed=0)
        np.testing.assert_array_almost_equal(perm, np.eye(5))

        perm = get_perm_matrix(num_img=5, seed=1)
        perm_expect = np.array([[0., 0., 0., 0., 1.],
                                [1., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 1., 0., 0.],
                                [0., 0., 0., 1., 0.]])

        np.testing.assert_array_almost_equal(perm, perm_expect)

    def test_call(self):
        perm_idx = 1
        shape = 10, 10
        num_img = 5
        exp = Experiment.from_gauss(shape=shape, seed=0, num_img=num_img)

        # prep
        perm = Permuter(x=exp.x[~exp.contrast, :])
