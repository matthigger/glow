import numpy as np


class CholeskyRegress:
    """ cholesky compute: regression beta and residual for multiple regression

    goal: quickly compute sum of squared residuals from many regression
    problems which share a common design matrix x

    let X (a, n) be a design matrix and Y (b, n) be observations of target
    variable.

    todo: doc needs update given removal of y**2 row funk
    two observations (let us assume b=1 below):
        (1) maindonald's cholesky ("statistical computation" 1984 ch 1)
        observe the (a+1, n) matrix [X.T Y.T].T made from appending Y to the
        bottom of X has:

            ssr = cholesky([X.T, Y.T].T @ [X.T, Y.T])[-1, -1]

        that is, its lower right entry is the squared sum of residuals.

        note that regardless of what y is, we apply the same row operations to
        yield the cholesky factor of the matrix X @ X.T (this matrix is all but
        last row / col of the matrix we find cholesky factor of above).  That
        is:

            X.T @ X @ row_ops_to_chol = cholesky(X.T @ X)

            so that:

            row_ops_to_chol = inv(X.T @ X) @ cholesky(X.T @ X)

        taking two results together, we are motivated to represent each region
        by a region vector r = X @ Y.T and its sum of y squared
        y2 = (Y ** 2).sum(axis=0)

            ssr = Y @ Y.T - ((r @ row_ops_to_chol) ** 2).sum()

        (2) note that any invertible transform of the design matrix X yields a
        regression problem whose ssr is unchanged.  Here, we choose to
        transform x to an orthonormal basis with identical span (gram-schmidt).
        the advantage is that X.T @ X = I so that:

        row_ops_to_chol = inv(X.T @ X) @ cholesky(X.T @ X)
                        = I @ I = I

        with our new basis, X_prime, we have a corresponding x_prime with:

            ssr = y2 - (r_prime ** 2).sum()

    The representation via r & y2 has a few advantages:
        - they're small (memory & compute savings, doesn't scale with
            observations)
        - may be summed across regions (r0 + r1 is the r matrix of the union of
            all voxels in region 0 and region 1, a big deal given how many of
            our regions are made of others)
        - r captures the variance due to each feature step by step (
            gram-schmidt style).  if all the reduced models variables are first
            (F-statistic) then we can subtract only these square sums to get
            the ssr_reduced

    In practice, we append y2 as a final row of r.

    Attributes:
            x (np.array): (a, n) explanatory variables
            x_prime (np.array): (a, n) orthonormal explanatory variables (same
                span as original explanatory variables)
            from_x_prime (np.array): (a, a) from_x_prime @ x_prime = x
            to_x_prime (np.array): (a, a) to_x_prime @ x = x_prime
    """

    def __init__(self, x):
        if x.shape[0]:
            assert np.linalg.matrix_rank(x) == x.shape[0], \
                'dependent x observations'

        self.x = x

        # compute x_prime & transforms
        q, r = np.linalg.qr(x.T, mode='reduced')
        self.x_prime = q.T
        self.from_x_prime = r.T
        self.to_x_prime = np.linalg.inv(r.T)

    def get_r(self, y):
        """ computes r vector per region

        (Assumes all inputs are of the same size)

        Args:
            y (np.array): (b, n, num_reg) image intensities

        Returns:
            r (np.array): (a + 1, b, num_reg) region arrays (see doc above)
        """
        # add dimensions to y as needed
        if y.ndim == 1:
            raise AttributeError('y must be 2d or 3d')
        elif y.ndim == 2:
            y = y[:, :, np.newaxis]
        assert y.ndim == 3

        # compute r
        a = self.x.shape[0]
        b, n, num_reg = y.shape
        r = np.empty((a + 1, b, num_reg), dtype=y.dtype)
        r[:-1, :, :] = np.einsum('an,bnr->abr', self.x_prime, y) ** 2

        # compute residual
        r[-1, :, :] = (y ** 2).sum(axis=1) - r[:-1, :, :].sum(axis=0)

        return r

    @staticmethod
    def get_ssr(r):
        """ computes sum of squared residual
        Args:
            r (np.array): (a + 1, b, num_reg) region arrays (see doc above),
                note last row is the y2 array while remaining rows above are r

        Returns:
            ssr (np.array): (a + 1, b, num_reg) sum of squared residual per region
                ssr[i, j, k] is the sum of squared residuals after accounting
                for covariates 0, 1, 2, 3, ..., i in imaging feature j for
                region k.  the last row corresponds to no covariates accounted
                for (y^2)
        """

        ssr = np.empty_like(r)
        ssr[-1, :, :] = r.sum(axis=0)
        ssr[:-1, :, :] = ssr[-1, :, :] - r[:-1, :, :].cumsum(axis=0)

        return ssr


class CholeskyRegressCovariate(CholeskyRegress):
    """ considers covariates (f stat computation via contrast vector)

    this object will swap the order of x features so that all covariates come
    first (necessary given cholesky iterative approach above)

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

        # apply transforms back to original x (and replace stored x w/ orig)
        self.x = x
        self.to_x_prime = self.to_x_prime @ self.to_sorted
        self.from_x_prime = self.to_sorted.T @ self.from_x_prime

    def get_f_ratio(self, r):
        """ f_ratio = (rss0 - rss1) / rss1 where rss0 is sum squares reduced

        Args:
            r (np.array): (a + 1, b, num_reg) region arrays (see doc above)

        Returns:
            f_ratio (np.array): (num_reg) f ratio per region
        """

        # add dimensions to r as needed
        if r.ndim == 1:
            raise AttributeError('r must be 2d or 3d')
        elif r.ndim == 2:
            r = r[:, :, np.newaxis]
        assert r.ndim == 3

        ssr0 = r[self.n_covariate:, :, :].sum(axis=(0, 1))
        ssr1 = r[-1, :, :].sum(axis=0)

        f_ratio = (ssr0 - ssr1) / ssr1

        assert (f_ratio >= 0).all(), 'negative f ratio'

        return f_ratio
