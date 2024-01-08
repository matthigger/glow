import pytest

from hrba.experiment import *

# maindonald statistical computation page 1
x = np.array([[1, 1, 1, 1], [-2, -1, 2, 7]])
y = np.array([[0, 2, 5, 3]])

c = np.array([[4, 6, 10],
              [6, 58, 29],
              [10, 29, 38]])
r_exp = np.linalg.cholesky(c)[-1, :-1][:, np.newaxis, np.newaxis]

# maindonald statistical computation page 6 (3 in bottom right entry
# squared is 9, the sum of squared residuals)
ssr = 9
y2_exp = np.atleast_2d((y ** 2).sum())


class TestCholeskyRegress:
    def test_init(self):
        chol_regr = CholeskyRegress(x=x)

        # test orthonormal
        a = x.shape[0]
        i_exp = chol_regr.x_prime @ chol_regr.x_prime.T
        np.testing.assert_array_almost_equal(i_exp, np.eye(a))

        # test that x and x_prime have same span (there is a transform from one
        # to the other)
        m = np.linalg.pinv(chol_regr.x) @ chol_regr.x_prime
        np.testing.assert_array_almost_equal(chol_regr.x @ m,
                                             chol_regr.x_prime)

    def test_get_r(self):
        chol_regr = CholeskyRegress(x=x)

        # test varying dimensions of y
        y_tuple = y, y[:, :, np.newaxis]
        for _y in y_tuple:
            r = chol_regr.get_r(y=y)
            r_obs = r[:-1, ...]
            y2_obs = r[-1, ...]

            # note that we may swap sign in r matrix without changing its
            # meaning
            np.testing.assert_array_almost_equal(np.abs(r_exp), np.abs(r_obs))
            np.testing.assert_array_almost_equal(y2_exp, y2_obs)

            # test to/from x prime
            np.testing.assert_array_almost_equal(x,
                                                 chol_regr.from_x_prime @ chol_regr.x_prime)
            np.testing.assert_array_almost_equal(chol_regr.x_prime,
                                                 chol_regr.to_x_prime @ x)

            with pytest.raises(AttributeError):
                chol_regr.get_r(y=np.squeeze(y))

    def test_get_ssr_get_beta(self):
        # build example
        num_reg = 7
        n = 5
        a = 2
        b = 3
        rng = np.random.default_rng(seed=0)
        x = rng.standard_normal((a, n))
        y = rng.standard_normal((b, n, num_reg))

        # compute expected (reliable, kind of clunky)
        beta_exp = np.stack([(y[:, :, reg_idx] @ np.linalg.pinv(x)).T
                             for reg_idx in range(num_reg)], axis=2)
        y_hat = np.stack([beta_exp[:, :, reg_idx].T @ x
                          for reg_idx in range(num_reg)], axis=2)
        ssr_exp = ((y - y_hat) ** 2).sum(axis=1)

        # compute observed
        chol_regr = CholeskyRegress(x=x)
        r = chol_regr.get_r(y)
        beta_obs = chol_regr.get_beta(r)
        ssr_obs = chol_regr.get_ssr(r)

        np.testing.assert_array_almost_equal(ssr_exp, ssr_obs)
        np.testing.assert_array_almost_equal(beta_exp, beta_obs)
