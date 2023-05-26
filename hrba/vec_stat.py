import numpy as np


class VecStat:
    """ n, mean and cov of a set of vectors.  has union method for 2 VecStat

    Attributes:
        n (int): number of vectors observed (vectors of length b)
        mu (np.array): (b,) mean vector
        cov (np.array): (b, b) covariance (no bessel's correction, simple
            average)
        mean_outer (np.array): (b, b) mean outer product (simple average) of
            all vectors
    """

    @classmethod
    def from_array(cls, x):
        """ VecStat from raw data

        Args:
            x (np.array): (b, n) a set of (column) vectors
        """
        assert x.ndim == 2, f'2d input required (passed {x.ndim}d array)'
        _, n = x.shape

        return cls(n=n, mu=x.mean(axis=1), mean_outer=x @ x.T / n)

    def __init__(self, n, mu, cov=None, mean_outer=None):
        assert (cov is None) != (mean_outer is None), \
            'either cov xor mean_outer required'
        self.n = n
        self.mu = mu

        if cov is None:
            # store mean_outer, compute cov
            self.mean_outer = mean_outer
            self.cov = mean_outer - np.outer(mu, mu)
        else:
            # store cov, compute mean_outer
            self.cov = cov
            self.mean_outer = cov + np.outer(mu, mu)

    def __or__(self, othr):
        """ returns VecStat representing union of self and other

        Args:
            othr (VecStat): another vector
        """
        if not isinstance(othr, VecStat):
            raise TypeError

        if self.mu.size != othr.mu.size:
            raise ValueError('dimension mismatch')

        # compute new n & weights of each addend
        n = self.n + othr.n
        lam0 = self.n / n
        lam1 = othr.n / n

        # compute new mean
        mean = self.mu * lam0 + \
               othr.mu * lam1

        # compute new mean_outer
        mean_outer = self.mean_outer * lam0 + \
                     othr.mean_outer * lam1

        return VecStat(n=n, mu=mean, mean_outer=mean_outer)
