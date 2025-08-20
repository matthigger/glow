from hglm.experiment import Experiment
from hglm.experiment.tailor import *


def test_tailor():
    # region 12 (covering regions 0, 1, 2, 3) is one effect
    # region 10 and 11 are identical effects, but we force region 13,
    # their union to be insignificant here to avoid their being merged
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5],
                         [6, 7],
                         [8, 9],
                         [10, 11],
                         [12, 13]])
    sig_reg_list = [5, 6, 7, 8, 9, 10, 11, 12, 14]
    exp = Experiment.from_gauss(shape=(8,), num_img=100)

    # ensure region 0, 1, 2, 3 have sufficiently different stats
    exp.y[:, :, :4] += 100

    reg_out, homo_pval_dict = tailor(sig_reg_list=sig_reg_list,
                                     children=children,
                                     exp=exp,
                                     alpha_tailor=.05,
                                     n_perm=100)

    assert np.allclose(reg_out, [10, 11, 12])
