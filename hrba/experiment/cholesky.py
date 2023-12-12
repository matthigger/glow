import numpy as np


class CholeskyRegress:
    """ cholesky compute: regression beta and residual for multiple regression

    goal: quickly compute sum of squared residuals from many regression
    problems which share a common design matrix x

    let X (a, n) be a design matrix and Y (b, n) be observations of target
    variable.

    two observations (let us assume b=1 below):
        (1) maindonald's cholesky ("statistical computation" 1984 ch 1)
        observe the (a+1, n) matrix [X.T Y.T].T made from appending Y to the
        bottom of X:

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

        (2) note that any linear transform of the design matrix X yields a
        regression problem whose ssr is unchanged.  Here, we choose to
        transform x to an orthonormal basis with identical span.  the advantage
        is that X.T @ X = I so that:

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

    Attributes:
            x (np.array): (a, n) explanatory variables
            x_prime (np.array): (a, n) orthonormal explanatory variables (same
                span as original explanatory variables)
            from_x_prime (np.array): (a, a) from_x_prime @ x_prime = x
            to_x_prime (np.array): (a, a) to_x_prime @ x = x_prime
    """

    def __init__(self, x):
        assert np.linalg.matrix_rank(x) == x.shape[0], \
            'dependent x observations'

        self.x = x

        # compute x_prime & transforms
        x_prime, from_x_prime = np.linalg.qr(x.T, mode='reduced')
        self.x_prime = x_prime.T
        self.from_x_prime = from_x_prime.T
        self.to_x_prime = np.linalg.inv(from_x_prime)

    def get_r_y2(self, y):
        """ computes r vector per region

        (Assumes all inputs are of the same size)

        Args:
            y (np.array): (b, n, num_reg) image intensities

        Returns:
            r (np.array): (a, b, num_reg) region arrays (see doc above)
            y2 (np.array): (b, num_reg) sum of squared intensities
        """
        # add dimensions to y as needed
        if y.ndim == 1:
            raise AttributeError('y must be 2d or 3d')
        elif y.ndim == 2:
            y = y[:, :, np.newaxis]
        assert y.ndim == 3

        # compute r
        r = np.einsum('an,bnr->abr', self.x_prime, y)

        # compute y2
        y2 = (y ** 2).sum(axis=1)

        return r, y2

    @staticmethod
    def get_ssr(r, y2):
        """ computes sum of squared residual
        Args:
            r (np.array): (a, b, num_reg) region arrays (see doc above)
            y2 (np.array): (b, num_reg) sum of squared intensities

        Returns:
            ssr (np.array): (num_reg) sum of squared residual per region
        """

        return y2 - (r ** 2).sum(axis=0)

    def get_beta(self, r):
        """ returns beta, the minimum MSE mapping from x to y

        Args:
            r (np.array): (a, b, num_reg) region arrays (see doc above)

        Returns:
            beta (np.array): (a, b, num_reg) min mse mapping (in original x
                space, not using the orthonormal x within this object)
        """
        # add dimensions to r as needed
        if r.ndim == 1:
            raise AttributeError('r must be 2d or 3d')
        elif r.ndim == 2:
            r = r[:, :, np.newaxis]
        assert r.ndim == 3

        # compute r
        beta = np.einsum('xa,abr->xbr', self.to_x_prime, r)

        return beta
