import numpy as np


class CholeskyRegress:
    """ cholesky compute: regression beta and residual for multiple regression

    goal: quickly compute sum of squared residuals from many regression
    problems which share a common design matrix x

    let X (n, a) be a design matrix and Y (n, b) be observations of target
    variable.  We represent each region with two values matrices:
        - the dot product of each input / output feature pair:
            r = X @ Y.T
        - the sum of each y feature squared, across observation:
            y2 = np.diag(Y @ Y.T)

    since x is common to all regions:
        - min MSE mapping  = np.linalg.inv(X @ X.T) @ X @ Y.T
                           = np.linalg.inv(X @ X.T) @ r
        - ssr (sum of squared residuals) may be computed as

            chol_last = r @ chol_factor
            y2 - (chol_last ** 2).sum(axis=0)

            where chol_factor is the matrix equivalent to the row operations
            which reduce X @ X.T to its cholesky factor (see __init__)

    The representation via r & y2 has a few advantages:
        - they're small (memory & compute savings, doesn't scale with
            observations)
        - may be summed across regions (r0 + r1 is the r matrix of the union of
            all voxels in region 0 and region 1)
        - the resulting cholesky ssr removes variance due to each y feature
            one by one.  if all "reduced model" covariates before "full model"
            covariates then one small modification gets us rss for both models
            (F-stat)

    Attributes:
            x (np.array): (a, n) explanatory variables
            x_xt (np.array): (a, a) X @ X.T.  dot product of each x with itself
                across all n observations
            x_xt_inv (np.array): (a, a) inverse of x_xt
            chol_factor (np.array): (a, a) chol_factor @ r will yield the final
                column in the cholesky decomposition of C above (useful to get
                ssr)
    """

    def __init__(self, x):
        self.x = x
        self.x_xt = x @ x.T
        assert np.linalg.det(self.x_xt) != 0, 'dependent x observations'
        self.x_xt_inv = np.linalg.inv(self.x_xt)
        self.chol_factor = self.x_xt_inv @ np.linalg.cholesky(self.x_xt)

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

        # compute r
        r = np.einsum('an,bnr->abr', self.x, y)

        # compute y2
        y2 = (y ** 2).sum(axis=1)

        return r, y2

    def get_ssr(self, r, y2):
        """ computes sum of squared residual
        Args:
            r (np.array): (a, b, num_reg) region arrays (see doc above)
            y2 (np.array): (b, num_reg) sum of squared intensities

        Returns:
            ssr (np.array): (num_reg) sum of squared residual per region
        """
        # get cholesky last columns
        chol_col = np.einsum('ax,abr->xbr', self.chol_factor, r)

        return y2 - (chol_col ** 2).sum(axis=0)

