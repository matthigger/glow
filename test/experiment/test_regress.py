from hglm.experiment import Experiment
from hglm.experiment.regress import *
from hglm.experiment.sigma import get_sigma
from hglm.graph import iter_size_yout_ymean, iter_topo


def test_all():
    """ test that regress stats computed properly for random data

    it is convenient here to test all region in an arbitrary hierarchy (
    allows us to use the iter_size_yout_ymean() generator), though these
    outputs generated another way would be just as valid
    """
    exp = Experiment.from_gauss(b=3, shape=(10, 10), seed=0)
    b, num_img, num_vox = exp.y.shape
    children = np.arange(2 * num_vox - 2).reshape((-1, 2))

    for reg_idx, size, yout, ymean in iter_size_yout_ymean(exp.y, children):
        yout = yout[:, :, 0]
        ymean = ymean[:, :, 0]

        # build y corresponding to region
        vox_idx = list(iter_topo(children=children,
                                 num_leaf=num_vox,
                                 node_start=reg_idx,
                                 only_leaf=True))
        _y = np.atleast_3d(exp.y[:, :, vox_idx])

        # compute sigma (slow & reliable)
        _y_center = _y - _y.mean(axis=2)[:, :, np.newaxis]
        sigma_exp = np.cov(_y_center.reshape((b, -1), order='F'), bias=True)

        sigma = get_sigma(size, yout, ymean)
        assert np.allclose(sigma_exp, sigma)
