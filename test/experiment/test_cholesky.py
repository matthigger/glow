from itertools import product

import pytest

from hrba.experiment import *

# maindonald "statistical computation" page 1
x = np.array([[1, 1, 1, 1], [-2, -1, 2, 7]])
y = np.array([[0, 2, 5, 3]])

# (see last column on bottom of page 6 maindonald)
qyt_exp = np.array([5, 2])[:, np.newaxis, np.newaxis]
y2_exp = np.array([38])


class TestCholeskyRegress:
    def test_get_r_maindonald(self):
        """ sanity check: example from maindonald"""
        chol_regr = CholeskyRegress(x=x)

        # test varying dimensions of y
        for _y in y, y[:, :, np.newaxis]:
            qyt_obs, y2_obs = chol_regr.get_qyt_y2(y=y)

            np.testing.assert_array_almost_equal(np.abs(qyt_exp),
                                                 np.abs(qyt_obs))
            np.testing.assert_array_almost_equal(y2_obs, y2_exp)

            with pytest.raises(AttributeError):
                chol_regr.get_qyt_y2(y=np.squeeze(y))

    def test_get_qyt_y2_mse_beta_rand(self):
        for x, y, mse, beta, mse_whole, beta_whole in case_iter():
            # get qyt, y2
            chol_regr = CholeskyRegress(x=x)
            qyt_obs, y2_obs = chol_regr.get_qyt_y2(y)

            # test mse (each individual region)
            a = x.shape[0]
            for _a in range(1, a + 1):
                mse_obs = chol_regr.get_mse(qyt_obs, y2_obs, a=_a)
                np.testing.assert_allclose(mse_obs, mse[_a - 1, :], atol=1e-8)

            # test beta (each individual region)
            beta_obs = chol_regr.get_beta(qyt_obs)
            np.testing.assert_allclose(beta_obs, beta)

            # compute qyt & y2 for whole region
            qyt_obs_whole = qyt_obs.mean(axis=2)
            y2_obs_whole = y2_obs.mean()

            # test mse_whole
            for _a in range(1, a + 1):
                mse_obs = chol_regr.get_mse(qyt_obs_whole, y2_obs_whole, a=_a)
                np.testing.assert_allclose(mse_obs, mse_whole[_a - 1, :],
                                           atol=1e-8)

            # test beta_whole
            beta_obs = chol_regr.get_beta(qyt_obs_whole)
            np.testing.assert_allclose(beta_obs, beta_whole)


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
                argsort = chol_regr.to_sorted @ np.arange(a).reshape((a, 1))
                np.testing.assert_array_almost_equal(np.argsort(contrast),
                                                     argsort.flatten())

                # validate x = rq
                np.testing.assert_array_almost_equal(x,
                                                     chol_regr.r @ chol_regr.q)
                assert chol_regr.n_covariate == n_covariate

                # create new shuffling of contrast for second run (doing after
                # allows us to ensure we test contrast already sorted)
                rng.shuffle(contrast)

    def test_get_f_ratio(self):
        rng = np.random.default_rng(seed=0)
        for x, y, mse, beta, mse_whole, beta_whole in case_iter():
            a = x.shape[0]
            for n_covariate in range(a - 1):
                # create "sorted" contrast (covariates up front)
                contrast = np.ones(a, dtype=bool)
                contrast[:n_covariate] = 0

                for shuffle_idx in range(4):
                    # expected (build reduced & full model explicitly)
                    chol_regr = CholeskyRegress(x)
                    qyt, y2 = chol_regr.get_qyt_y2(y)
                    mse1 = chol_regr.get_mse(qyt, y2)
                    mse0 = chol_regr.get_mse(qyt, y2, a=n_covariate)
                    f_ratio_exp = (mse0 - mse1) / mse1

                    # observed
                    chol_regr = CholeskyRegressCovariate(x, contrast)
                    qyt, y2 = chol_regr.get_qyt_y2(y)
                    f_ratio_obs = chol_regr.get_f_ratio(qyt, y2)

                    np.testing.assert_array_almost_equal(f_ratio_obs,
                                                         f_ratio_exp)

                    # create new shuffling of contrast (first ordering is
                    # always sorted)
                    rng.shuffle(contrast)


def case_iter(a=range(1, 3), b=range(1, 3), num_reg=(1, 10), seed=range(4),
              num_img=(11,)):
    for _a, _b, _num_reg, _seed, _num_img in product(a, b, num_reg, seed,
                                                     num_img):
        yield case(a=_a, b=_b, num_reg=_num_reg, seed=_seed, num_img=_num_img)


def case(a=3, b=3, num_reg=11, num_img=10, seed=0):
    rng = np.random.default_rng(seed=seed)

    x = rng.standard_normal((a, num_img))
    y = rng.standard_normal((b, num_img, num_reg))

    mse, beta = get_mse_beta(x, y)

    # flatten x and y (union of all regions)
    xr = np.tile(x, (1, num_reg))
    yr = y.reshape((y.shape[0], -1), order='F')
    # validate yr reshaping
    np.testing.assert_array_almost_equal(y[:, :, 0], yr[:, :y.shape[1]])
    mse_whole, beta_whole = get_mse_beta(xr, yr)

    return x, y, mse, beta, mse_whole, beta_whole


def get_mse_beta(x, y):
    """ slow and reliable computing of mse and beta

    returns:
        mse (np.array): (a, num_reg) mse[i, ...] gives the mean of
            squared residuals if we use only i of the a total features.
        beta (np.array): (a, b, num_reg) the mmse mapping from x to y for each
            region
    """
    a = x.shape[0]
    y = np.atleast_3d(y)
    b, num_img, num_reg = y.shape

    # compute beta (explicit, slow and reliable)
    pinv_x = np.linalg.pinv(x)
    beta = np.empty((a, b, num_reg))
    for reg_idx in range(num_reg):
        _y = y[:, :, reg_idx]
        beta[:, :, reg_idx] = (_y @ pinv_x).T

    # compute ssr (explicit, slow and reliable)
    ssr = np.empty((a, num_reg))
    for a_idx in range(a):
        _x = x[:a_idx + 1, :]
        resid_form = np.eye(num_img) - np.linalg.pinv(_x) @ _x
        for reg_idx in range(num_reg):
            resid = y[:, :, reg_idx] @ resid_form
            ssr[a_idx, reg_idx] = (resid ** 2).sum()

    mse = ssr / x.shape[1]

    return mse, beta
