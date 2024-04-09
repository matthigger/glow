import numpy as np


def generate_dummy_data(a=2, b=3, num_img=5, num_vox=10, bias=True,
                        seed=None, add_effect=False):
    # generate dummy data
    rng = np.random.default_rng(seed=seed)
    x = rng.standard_normal((a, num_img))
    y = rng.standard_normal((b, num_img, num_vox))
    contrast = np.ones(a, dtype=bool)

    if bias:
        contrast[0] = False
        x[0, :] = 1

    if add_effect:
        # add strong effect
        beta = rng.standard_normal((b, a)) * 100
        y += (beta @ x)[..., np.newaxis]

    return x, y, contrast
