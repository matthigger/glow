import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages

if not rpackages.isinstalled('dglm'):
    utils = rpackages.importr('utils')
    utils.install_packages('dglm')

dglm = rpackages.importr('dglm')


class LLRModel:
    """ model of llr under h0 (no effect present)

    llr = size * m + b + error

    where

    error ~N(mu=0, var=exp(size*m' + b'))

    Attributes:
        m (float): see definition above
        b (float): see definition above
        mp (float): see definition above
        bp (float): see definition above
    """

    def __init__(self):
        self.m = None
        self.b = None
        self.mp = None
        self.bp = None

    def fit(self, llr, size):
        """ fits model to observed llr and size

        we actually fit the following model

        (llr/size) = m + b/size + error

        error is consistent across observations so OLS works just fine for ML
        """
        llr_rvec = ro.FloatVector(llr.flatten())
        size_rvec = ro.FloatVector(size.flatten())

        formula_mean = ro.Formula('llr~size')
        formula_mean.environment['llr'] = llr_rvec
        formula_mean.environment['size'] = size_rvec

        formula_disp = ro.Formula('~size')
        formula_disp.environment['size'] = size_rvec

        dglm = ro.r['dglm']
        fit = dglm(formula_mean, dformula=formula_disp, dlink='log',
                   family='gaussian')

        self.b, self.m = tuple(fit.rx2('coefficients'))
        self.bp, self.mp = tuple(fit.rx2('dispersion.fit').rx2('coefficients'))

    def predict(self, size):
        """ gets mean and std of llr given size

        Args:
            size (np.array): size of each region

        Returns:
            mu (np.array): expected llr of each region under h0
            var (np.array): expected var of each region under h0
        """
        mu = size * self.m + self.b
        var = np.exp(size * self.mp + self.bp)

        return mu, var
