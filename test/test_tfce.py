from hglm.tfce import *


def test_apply_tfce():
    # get random 5x5x5 image
    np.random.seed(0)
    x = np.random.normal(size=(5, 5, 5))

    # run tfce
    apply_tfce_img(x)
