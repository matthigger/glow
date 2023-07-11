import numpy as np


class LLRModel:
    """ model of llr under h0 (no effect present)

    llr = size * m + b + error where error ~N(0, var=size^2*var)

    llr = size * m + b + error * size where error ~N(0, var)

    Attributes:
        m (float): see definition above
        b (float): see definition above
        var (float): see definition above
    """

    def __init__(self):
        self.m = None
        self.b = None
        self.var = None

    def fit(self, llr, size):
        """ fits model to observed llr and size

        we actually fit the following model

        (llr/size) = m + b/size + error

        error is consistent across observations so OLS works just fine for ML
        """
        # transform to homoskedastic error
        llr_over_size = llr.flatten() / size.flatten()
        one_over_size = 1 / size.flatten()

        # fit model
        x = np.vstack([np.ones(size.size), one_over_size])
        coef = llr_over_size @ np.linalg.pinv(x)

        # should look backwards, "intercept" of the fitted model is m
        self.m = coef[0]
        self.b = coef[1]
        error = llr_over_size - self.m + self.b * one_over_size
        self.var = np.var(error)

    def predict(self, size):
        """ gets mean and std of llr given size

        Args:
            size (np.array): size of each region

        Returns:
            mu (np.array): expected llr of each region under h0
            var (np.array): expected var of each region under h0
        """
        mu = size * self.m + self.b
        var = size ** 2 * self.var

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

        m, b, mp = tuple(theta.flatten())

        error = llr - m * size + b
        std = (mp * size) ** .5
        const = np.log(2 * np.pi) * llr.size

        return -.5 * (const +
                      np.log(std).sum() +
                      ((error / std) ** 2).sum())
