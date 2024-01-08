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
        np.testing.assert_array_almost_equal(x,
                                             chol_regr.from_x_prime @ chol_regr.x_prime)
        np.testing.assert_array_almost_equal(chol_regr.x_prime,
                                             chol_regr.to_x_prime @ x)

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

            with pytest.raises(AttributeError):
                chol_regr.get_r(y=np.squeeze(y))

    def test_get_ssr_get_beta(self):
        # build example
        num_reg = 7
        n = 5
        rng = np.random.default_rng(seed=0)
        b = 3

        for a in range(3):
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


class TestCholeskyRegressCovariate:
    def test_init(self):
        # build example
        n = 100
        a = 5
        rng = np.random.default_rng(seed=0)
        x = rng.standard_normal((a, n))

        for n_covariate in range(a - 1):
            contrast = np.ones(a)
            contrast[:n_covariate] = 0

            for shuffle_idx in range(4):
                chol_regr = CholeskyRegressCovariate(x, contrast)

                # check that to_sorted sorts as required (increasing contrast)
                exp = chol_regr.to_sorted @ np.arange(a).reshape(
                    (a, 1)).flatten()
                np.testing.assert_array_almost_equal(np.argsort(contrast), exp)

                # check to and from x_prime
                np.testing.assert_array_almost_equal(x,
                                                     chol_regr.from_x_prime @ chol_regr.x_prime)
                np.testing.assert_array_almost_equal(chol_regr.x_prime,
                                                     chol_regr.to_x_prime @ x)

                assert chol_regr.n_covariate == n_covariate

                # create new shuffling of contrast for second run (doing after
                # allows us to ensure we test contrast already sorted)
                rng.shuffle(contrast)

    def test_get_f_ratio(self):
        # build example
        num_reg = 11
        n = 10
        b = 3
        rng = np.random.default_rng(seed=0)

        for a in range(3):
            x = rng.standard_normal((a, n))
            y = rng.standard_normal((b, n, num_reg))

            for n_covariate in range(a - 1):
                contrast = np.ones(a, dtype=bool)
                contrast[:n_covariate] = 0

                for shuffle_idx in range(4):
                    # expected (build reduced & full model explicitly)
                    chol_regr = (CholeskyRegress(x[~contrast, :]),
                                 CholeskyRegress(x))
                    ssr = [_chol_reg.get_ssr(r=_chol_reg.get_r(y)).sum(axis=0)
                           for _chol_reg in chol_regr]
                    f_ratio_exp = (ssr[0] - ssr[1]) / ssr[1]

                    # observed
                    chol_regr = CholeskyRegressCovariate(x, contrast)
                    r = chol_regr.get_r(y)
                    f_ratio_obs = chol_regr.get_f_ratio(r)

                    np.testing.assert_array_almost_equal(f_ratio_obs,
                                                         f_ratio_exp)

                    # create new shuffling of contrast for second run (doing after
                    # allows us to ensure we test contrast already sorted)
                    rng.shuffle(contrast)
