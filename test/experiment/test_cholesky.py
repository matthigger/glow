from math import isclose

import pytest

from hrba.experiment import *

# maindonald statistical computation page 1
x = np.array([[1, 1, 1, 1], [-2, -1, 2, 7]])
y = np.array([[0, 2, 5, 3]])
r = np.array([10, 29])[:, np.newaxis, np.newaxis]

# maindonald statistical computation page 6 (3 in bottom right entry
# squared is 9, the sum of squared residuals)
ssr = 9
y2 = np.atleast_2d((y ** 2).sum())


class TestCholeskyRegress:
    def test_get_r_y2(self):
        chol_regr = CholeskyRegress(x=x)

        # test varying dimensions of y
        y_tuple = y, y[:, :, np.newaxis]
        for _y in y_tuple:
            r_obs, y2_obs = chol_regr.compute_r_y2(y=y)
            np.testing.assert_array_almost_equal(r, r_obs)
            np.testing.assert_array_almost_equal(y2, y2_obs)

        with pytest.raises(AttributeError):
            chol_regr.compute_r(y=np.squeeze(y))

    def test_math_maindonald(self):
        chol_regr = CholeskyRegress(x=x)
        r, y2 = chol_regr.get_r_y2(y=y)
        chol_col = np.squeeze(r) @ chol_regr.chol_factor

        assert isclose(ssr, y2 - (chol_col ** 2).sum())

    def test_compute_ssr(self):
        num_reg = 7
        n = 5
        a = 2
        b = 3

        rng = np.random.default_rng(seed=0)
        x = rng.standard_normal((a, n))
        y = rng.standard_normal((b, n, num_reg))

        # expected (reliable)
        to_y_hat = np.linalg.pinv(x) @ x
        y_hat = np.stack([y[:, :, reg_idx] @ to_y_hat
                          for reg_idx in range(num_reg)], axis=2)
        ssr_exp = ((y - y_hat) ** 2).sum(axis=1)

        # observed
        chol_regr = CholeskyRegress(x=x)
        r, y2 = chol_regr.get_r_y2(y)
        ssr_obs = chol_regr.get_ssr(r, y2)

        np.testing.assert_array_almost_equal(ssr_exp, ssr_obs)
