import numpy as np
from numpy.polynomial.polynomial import Polynomial

from ..helper import generate_dummy_data


def test_validate_eps():
    """ validate formula for residual sum of squares

    N|r| \mathcal{E} = \sum_v Y_v Y_v^T - |r| \bar{Y}_r H \bar{Y}^T

    and H = Q @ Q.T
    """
    a, b, num_img, reg_size = 2, 3, 5, 10

    for seed in range(10):
        for add_effect in range(2):
            x, y, _ = generate_dummy_data(bias=True, add_effect=add_effect)

            # compute residual sum of squares (explicit, slow & reliable)
            y_mean = y.mean(axis=2)
            h = np.linalg.pinv(x) @ x
            y_hat = y_mean @ h

            # test h = q
            q, r = np.linalg.qr(x.T, mode='reduced')
            q = q.T
            assert np.allclose(h, q.T  @ q)

            y_resid = y - y_hat[..., np.newaxis]
            eps_exp = np.cov(y_resid.reshape((b, -1)), ddof=0)

            # compute residual sum of squares (quick formula)
            yv = y.reshape((b, -1))
            eps_obs = yv @ yv.T - reg_size * y_mean @ h @ y_mean.T
            eps_obs = eps_obs / (num_img * reg_size)
            assert np.allclose(eps_exp, eps_obs)


def test_validate_tr_eps_delta():
    """ validates eps w/ add constant (may differ per image, constant in space)

    suppose we add \Delta \in \mathcal{R}^{b, num_img} to all voxels within a
    region, how does this change the trace of the error covariance?

    N|r| tr \mathcal{E} = tr \sum_v Y_v Y_v^T +
                            |r| (-\bar{Y}_r H \bar{Y}^T
                                 + 2c \Delta (I - H) \bar{Y}_r^T
                                 + c^2 \Delta (I - H) \Delta^T)
    """
    for seed in range(10):
        for add_effect in range(2):
            x, y, _ = generate_dummy_data(bias=True, add_effect=add_effect)

            # sample delta
            b, num_img, reg_size = y.shape
            rng = np.random.default_rng(seed=seed)
            delta = rng.standard_normal((b, num_img))

            # compute residual sum of squares (polynomial in c)
            yv_squared = (y ** 2).sum()
            h = np.linalg.pinv(x) @ x
            i_minus_h = np.eye(num_img) - h
            y_mean = y.mean(axis=2)
            eps_obs_poly = Polynomial([
                yv_squared - reg_size * np.trace(y_mean @ h @ y_mean.T),
                reg_size * 2 * np.trace(delta @ i_minus_h @ y_mean.T),
                reg_size * np.trace(delta @ i_minus_h @ delta.T)])
            eps_obs_poly /= num_img * reg_size

            for c in np.linspace(-10, 10, 8):
                # compute residual sum of squares (explicit, slow & reliable)
                y_offset = y + c * delta[..., np.newaxis]
                y_mean = y_offset.mean(axis=2)
                y_hat = y_mean @ h
                y_resid = y_offset - y_hat[..., np.newaxis]
                eps_exp = np.cov(y_resid.reshape((b, -1)), ddof=0)

                assert np.isclose(np.trace(eps_exp), eps_obs_poly(c))
