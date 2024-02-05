from itertools import product

import pytest

from hrba.experiment import *

# maindonald statistical computation page 1
x = np.array([[1, 1, 1, 1], [-2, -1, 2, 7]])
y = np.array([[0, 2, 5, 3]])

c = np.array([[4, 6, 10],
              [6, 58, 29],
              [10, 29, 38]])

# (see first array on page 6 maindonald, we store square of last column)
r_exp = np.array([[[25]],
                  [[4]],
                  [[9]]])


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

    def test_get_r_maindonald(self):
        chol_regr = CholeskyRegress(x=x)

        # test varying dimensions of y
        y_tuple = y, y[:, :, np.newaxis]
        for _y in y_tuple:
            r_obs = chol_regr.get_r(y=y)

            # note that we may swap sign in r matrix without changing its
            # meaning
            np.testing.assert_array_almost_equal(np.abs(r_exp), np.abs(r_obs))

            with pytest.raises(AttributeError):
                chol_regr.get_r(y=np.squeeze(y))

    def test_get_r_beta_rand(self):
        num_img = 11
        for a, b, num_reg, seed in product(range(1, 3),
                                           range(1, 3),
                                           [1, 10],
                                           range(4)):
            x, y, ssr, beta = case(a=3, b=b, num_reg=num_reg, seed=seed,
                                   num_img=num_img)

            # test get_r
            chol_regr = CholeskyRegress(x=x)
            r_obs = chol_regr.get_r(y)

            # test y2 term
            y2 = (y ** 2).sum(axis=1)
            np.testing.assert_array_almost_equal(y2, r_obs.sum(axis=0))

            # test ssr (remaining terms in r)
            ssr_obs = chol_regr.get_ssr(r_obs)
            np.testing.assert_array_almost_equal(ssr, ssr_obs)


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
        rng = np.random.default_rng(seed=0)
        num_img = 11
        for a, b, num_reg, seed in product(range(1, 3),
                                           range(1, 3),
                                           [1, 10],
                                           range(4)):
            x, y, ssr, beta = case(a=a, b=b, num_reg=num_reg, seed=seed,
                                   num_img=num_img)

            for n_covariate in range(a - 1):
                # create "sorted" contrast (covariates up front)
                contrast = np.ones(a, dtype=bool)
                contrast[:n_covariate] = 0

                for shuffle_idx in range(4):
                    # expected (build reduced & full model explicitly)
                    chol_regr = CholeskyRegress(x)
                    ssr = chol_regr.get_ssr(chol_regr.get_r(y))
                    ssr0 = ssr[n_covariate - 1, :, :].sum(axis=0)
                    ssr1 = ssr[-2, :, :].sum(axis=0)
                    f_ratio_exp = (ssr0 - ssr1) / ssr1

                    # observed
                    chol_regr = CholeskyRegressCovariate(x, contrast)
                    r = chol_regr.get_r(y)
                    f_ratio_obs = chol_regr.get_f_ratio(r)

                    np.testing.assert_array_almost_equal(f_ratio_obs,
                                                         f_ratio_exp)

                    # create new shuffling of contrast (first ordering is
                    # always sorted)
                    rng.shuffle(contrast)


def case(a=3, b=3, num_reg=11, num_img=10, seed=0):
    rng = np.random.default_rng(seed=seed)

    x = rng.standard_normal((a, num_img))
    y = rng.standard_normal((b, num_img, num_reg))

    ssr, beta = get_ssr_beta(x, y)

    return x, y, ssr, beta


def case_multi_vox(*args, num_vox=2, **kwargs):
    x, y, ssr, beta = case(*args, num_reg=num_vox, **kwargs)

    # flatten x and y
    xr = np.tile(x, (1, num_vox))
    yr = y.reshape((y.shape[0], -1), order='F')
    # validate yr reshaping
    np.testing.assert_array_almost_equal(y[:, :, 0], yr[:, :y.shape[1]])

    ssr, beta = get_ssr_beta(xr, yr)

    return xr, yr, ssr, beta


def get_ssr_beta(x, y):
    """ slow and reliable computing of ssr and beta

    returns:
        ssr (np.array): (a + 1, b, num_reg) ssr[i, ...] gives the sum of
            squared residuals if we use only i of the a total.  the final row
            ssr[-1, ...] represents the sum of square of y (not a residual)
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
    ssr = np.empty((a + 1, b, num_reg))
    for a_idx in range(a):
        _x = x[:a_idx + 1, :]
        resid_form = np.eye(num_img) - np.linalg.pinv(_x) @ _x
        for reg_idx in range(num_reg):
            resid = y[:, :, reg_idx] @ resid_form
            ssr[a_idx, :, reg_idx] = np.diag(resid @ resid.T)
    ssr[-1, :, :] = (y ** 2).sum(axis=1)

    return ssr, beta


def test_eqn():
    num_vox = 3
    kwargs = dict(a=2, b=4, num_img=10, seed=0)
    # build case
    x, y, ssr, beta = case(num_reg=num_vox, **kwargs)
    # build multi-voxel case (corresponds to union of all regions above)
    xr, yr, ssr_r, beta_r = case_multi_vox(num_vox=num_vox, **kwargs)

    # build q
    q, r = np.linalg.qr(x.T, mode='reduced')
    q = q.T

    for a_idx in range(kwargs['a']):
        # reduce to only the first few features in x
        _q = q[:a_idx + 1, :]

        # compute ssr expected per formulation
        # N|r| \Tr \mathcal{E} = ||Y_r||_F^2 - |r| ||Q \bar{Y}_r^T||_F^2
        ssr_exp = (yr ** 2).sum()
        ssr_exp -= num_vox * ((_q @ y.mean(axis=2).T) ** 2).sum()
        assert np.isclose(ssr_exp, ssr_r[a_idx, :, 0].sum())

    # build c_r by summing in paper
