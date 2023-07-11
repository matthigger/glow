import numpy as np
from scipy.optimize import minimize


class LLRModel:
    """ model of llr under h0 (no effect present)

    llr = size * m + b + error

    where:

    error ~ N(mu=0, var=size * m' + b')

    Attributes:
        theta (np.array): (4) [m, b, m', b'] parameters
    """

    def __init__(self):
        self.theta = None
        self.res = None

    def fit(self, llr, size, **kwargs):
        """ fits model to observed llr and size"""

        def obj(theta):
            return -LLRModel.log_like(theta=theta, size=size, llr=llr)

        def jac(theta):
            return LLRModel.jac(theta=theta, size=size, llr=llr)

        # optimize
        self.res = minimize(obj, x0=np.ones(4), bounds=[(0, None),
                                                        (0, None),
                                                        (0, None),
                                                        (0, None)],
                            method='Nelder-Mead', options={'maxiter': 1e5})
        assert self.res.success, f'optimization failed'

        self.theta = self.res.x

    def predict(self, size):
        """ gets mean and std of llr given size

        Args:
            size (np.array): size of each region

        Returns:
            mu (np.array): expected llr of each region under h0
            var (np.array): expected var of each region under h0
        """
        assert self.theta is not None, 'model must be .fit() first'
        m, b, mp, bp = tuple(self.theta.flatten())

        mu = size * m + b
        var = size * mp + bp

        return mu, var

    @classmethod
    def log_like(cls, theta, size, llr):
        """ log likelihood of model

        Args:
            theta (np.array): (4) [m, b, m', b'] parameters
            size (np.array): (n) size of each region
            llr (np.array): log likelihood ratio of each region

        Returns:
            log_like (float): log likelihood
        """
        assert size.shape == llr.shape

        m, b, mp, bp = tuple(theta.flatten())

        error = llr - m * size + b
        std = (mp * size + bp) ** .5
        const = np.log(2 * np.pi) * llr.size

        return -.5 * (const +
                      np.log(std).sum() +
                      ((error / std) ** 2).sum())

    @classmethod
    def jac(cls, theta, size, llr):
        """ computes derivative of llr w/ respect to each item in theta

        Args:
            theta (np.array): (4) [m, b, m', b'] parameters
            size (np.array): (n) size of each region
            llr (np.array): log likelihood ratio of each region

        Returns:
            jac (np.array): (4) [dL/dm, dL/db, dL/dm', dL/db']
        """
        m, b, mp, bp = tuple(theta.flatten())

        # compute common numerator and denominator terms
        num = (llr - m * size + b)
        den = mp * size + bp

        # and some powers
        num2 = num ** 2
        den2 = den ** 2
        den3 = den ** 3

        # compute partial derivatives
        dldm = (size * num / den2).sum()
        dldb = - (num / den2).sum()
        dlbmp = (size * num2 / den3 - .5 * size / den).sum()
        dldbp = (num2 / den3 - .5 / den).sum()

        return np.array([dldm, dldb, dlbmp, dldbp])
