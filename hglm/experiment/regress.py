import numpy as np


def to_3d(x):
    if x.ndim == 1:
        raise AttributeError('y must be 2d or 3d')
    elif x.ndim == 2:
        return x[:, :, np.newaxis]
    assert x.ndim == 3
    return x


class QRRegress:
    """ computes sum of squared residuals from regressions with common x

    see paper for detail

    Attributes:
            x (np.array): (a, n) explanatory variables
            q (np.array): (a, n) orthonormal explanatory variables (same
                span as original explanatory variables)
            r (np.array): (a, a) x = rq.  maps from q to x (its invertible, if
                x is full rank, if you wanted to go backwards too)
    """

    def __init__(self, x):
        if x.shape[0]:
            assert np.linalg.matrix_rank(x) == x.shape[0], \
                'dependent x observations'

        self.x = x

        # compute x_prime & transforms
        q, r = np.linalg.qr(x.T, mode='reduced')
        self.q = q.T
        self.r = r.T
        self.r_inv = np.linalg.inv(self.r)

    def get_qyt_yout(self, y):
        """ computes qyt vector per region

        Args:
            y (np.array): (b, n, num_reg) image intensities (spatial average)

        Returns:
            qyt (np.array): (a, b, num_reg) alternate beta
            yout (np.array): (b, b, num_reg) sum of outer product of all y
                values for each image (n), but averaged across voxels in the
                region (in this method, each region is 1 voxel so this isn't as
                relevant here)
        """
        y = to_3d(y)

        # compute qyt
        qyt = np.einsum('an,bnr->abr', self.q, y)

        # compute residual
        yout = np.einsum('abc,xbc->axc', y, y)

        return qyt, yout

    def get_eps(self, qyt, yout, a=None):
        """ computes sum of squared residual

        Args:
            qyt (np.array): (a, b, num_reg) alternate beta
            yout (np.array): (b, b, num_reg) sum of outer product of all y
                values for each image (n), but averaged across voxels in the
                region
            a (int): considers only the first a features (default to all)

        Returns:
            eps (np.array): (b, b, num_reg) covariance matrix of error (
                averaged over voxels in region and number of images n)
        """
        qyt = to_3d(qyt)

        if a is not None:
            qyt = qyt[:a, :, :]

        offset = np.einsum('abc,ayc->byc', qyt, qyt)

        return (np.atleast_3d(yout) - offset) / self.x.shape[1]

    def get_mse(self, *args, **kwargs):
        """ gets mean squared error (diag sum of eps)

        Args:
            see get_eps()

        Returns:
            mse (np.array): (num_reg) mean squared error
        """
        eps = self.get_eps(*args, **kwargs)
        return np.einsum('aab', eps)

    def get_beta(self, qyt):
        """ computes minimum mse beta

        Args:
            qyt (np.array): (a, b, num_reg) alternate beta

        Returns:
            beta (np.array): (a, b, num_reg) minimum MSE mapping from x to y
        """
        qyt = to_3d(qyt)

        return np.einsum('abr,an->nbr', qyt, self.r_inv)


class QRRegressCovariate(QRRegress):
    """ considers covariates (f stat computation via contrast vector)

    this object will swap the order of x features in qr decomposition so that
    all covariates come first.  this allows us to compute residual sum of
    squares from the reduced and full models from same matrix.

    Attributes:
        n_covariate (int): number of covariates
    """

    def __init__(self, x, contrast):
        """
        Args:
            contrast (np.array): (a) True for each corresponding feature in
                x which is "of interest" (other x features form the reduced
                model in computing f statistic)
        """
        # build matrix, which when left multiplied by x, produced x_sorted
        # which shuffles rows so that all reduced features come first
        n = x.shape[0]
        self.to_sorted = np.eye(n)[np.argsort(contrast), :]

        super().__init__(x=self.to_sorted @ x)

        self.n_covariate = contrast.size - contrast.sum()

        # replace shuffled x w/ orig, modify r as needed so that x = rq holds
        self.x = x
        self.r = np.linalg.inv(self.to_sorted) @ self.r
        self.r_inv = np.linalg.inv(self.r)

    def get_f_ratio(self, qyt, yout):
        """ (mse0 - mse1) / mse1 where mse0 is mean square error reduced model

        Args:
            qyt (np.array): (a, b, num_reg) alternate beta
            yout (np.array): (b, b, num_reg) sum of outer product of all y
                values for each image (n), but averaged across voxels in the
                region
        Returns:
            f_ratio (np.array): (num_reg) f ratio per region
        """
        qyt = to_3d(qyt)

        # compute numerator
        mse0_minus_mse1 = (qyt[self.n_covariate:, :, :] ** 2).sum(axis=(0, 1))
        mse0_minus_mse1 /= self.x.shape[1]

        # compute denominator
        mse1 = self.get_mse(qyt, yout)

        return mse0_minus_mse1 / mse1

    def get_llr(self, qyt, yout, size):
        """ log likelihood ratio

        Args:
            qyt (np.array): (a, b, num_reg) alternate beta
            yout (np.array): (b, b, num_reg) sum of outer product of all y
                values for each image (n), but averaged across voxels in the
                region
            size (np.array): (num_reg) size of region (in voxels)

        Returns:
            llr (np.array): (num_reg) log likelihood ratio per region
        """
        eps_full = self.get_eps(qyt, yout)
        eps_reduced = self.get_eps(qyt, yout, a=self.n_covariate)

        def log_det(eps):
            # det operates on last 2 dims by default
            eps = np.moveaxis(eps, -1, 0)
            return np.log(np.linalg.det(eps))

        return size / 2 * (log_det(eps_reduced) - log_det(eps_full))