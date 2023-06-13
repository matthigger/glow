from _bisect import bisect_left

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import grid_to_graph
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from hrba.graph import iter_reg_stat_exp
from .permute import get_perm_matrix


class Epoch:
    """ a single round of region discovery in HRBA, computes FWER p-val

    Attributes:
        exp (Experiment): the source data to run experiment on
        dendro_dict (dict): keys are permutation indices, values are
            (2, n) dendrogram arrays (equiv to sklearn.cluster.Ward.children_)
        size (np.array): (n_permute + 1, num_reg) size of each region
        f_stat (np.array): (n_permute + 1, num_reg) raw f-stat of each region
        z_stat (np.array): (n_permute + 1, num_reg)  "z-score" of each f-stat
        p_val (np.array): (n_permute + 1, num_reg) FWER controlled pval
        model_f_mu (LinearRegression): the mean f stat as a function of
            region size (log10 F = m * log10 num_vox + b)
        model_f_var (LinearRegression): the var of f stat as a function
            of region size (log10 F_var = m * log10 num_vox + b)
    """

    def __init__(self, exp, n_permute, **kwargs):
        self.exp = exp

        # build hierarchy
        self.dendro_dict = self.cluster(n_permute=n_permute, **kwargs)

        # compute f stat per every region in hierarchy (across all permutes)
        self.size, self.f_stat = self.get_f_stat(exp=exp,
                                                 dendro_dict=self.dendro_dict)

        # model f mu & var as a function of region size
        self.model_f_mu, self.model_f_var, self.z_stat = \
            self.model_adjust_f(self.size, self.f_stat)

        # compute p-values
        self.p_val = self.get_pval(self.z_stat)

    @classmethod
    def cluster(cls, exp, n_permute, verbose=True):
        """ build dendro_dict """
        # prep
        b, num_img, num_vox = exp.y.shape
        x = exp.x[~exp.contrast, :], exp.x
        h = [np.linalg.pinv(_x) @ _x for _x in x]
        h_diff = h[1] - h[0]
        i = np.eye(num_img)

        # get connectivity (ensures only neighboring voxels joined)
        mask = exp.mask_idx >= 0
        if mask.ndim == 3:
            shape = mask.shape
        elif mask.ndim == 2:
            shape = (*mask.shape, 1)

        # prep ward clustering object
        dendro_dict = dict()
        connectivity = grid_to_graph(*shape, mask=mask)
        ward = AgglomerativeClustering(connectivity=connectivity,
                                       linkage='ward')
        tqdm_dict = dict(desc='clustering per permutation',
                         disable=not verbose)
        for perm_idx in tqdm(range(n_permute + 1), **tqdm_dict):
            # permute data residuals under reduced model (freedman lane)
            p = get_perm_matrix(perm_idx, num_img)
            freed_lane = (i - h[0]) @ p + h[0]

            # prepare y
            y = np.einsum('ijk,jm->imk', exp.y, freed_lane @ h_diff)
            y = y.reshape((-1, num_vox))

            # cluster & store
            ward.fit(y.T)
            dendro_dict[perm_idx] = ward.children_

        return dendro_dict

    @classmethod
    def get_f_stat(cls, exp, dendro_dict, verbose=True):
        # compute f-stat per region in all permutations
        num_vox = exp.y.shape[2]
        n_permute = len(dendro_dict)
        shape = (n_permute + 1, 2 * num_vox - 1)
        size = np.full(shape, fill_value=-1, dtype=int)
        f_stat = np.full(shape, fill_value=-1, dtype=float)

        tqdm_dict = dict(desc='compute stats per permutation',
                         disable=not verbose)
        for perm_idx, dendro in tqdm(dendro_dict.items(),
                                     **tqdm_dict):
            for reg_idx, _size, _f_stat in iter_reg_stat_exp(dendro=dendro,
                                                             exp=exp):
                size[perm_idx, reg_idx] = _size
                f_stat[perm_idx, reg_idx] = _f_stat

        return size, f_stat

    @classmethod
    def model_adjust_f(cls, size, f_stat):
        """ models f stats as function of region size

        motivation: we need a size agnostic stat per region (i.e. z-stat)
        """
        # init
        model_f_mu = LinearRegression(fit_intercept=True)
        model_f_var = LinearRegression(fit_intercept=True)

        # fit model_f_mu
        log_size = np.log10(size).reshape(-1, 1)
        log_f = np.log10(f_stat).flatten()
        model_f_mu.fit(X=log_size, y=log_f)

        # fit model_f_var
        error = log_f - model_f_mu.predict(log_size)
        model_f_var.fit(X=log_size, y=error ** 2)

        # adjust f stats per region size
        mu = model_f_mu.predict(log_size).reshape(size.shape)
        var = model_f_var.predict(log_size).reshape(size.shape)
        z_stat = (f_stat - mu) / (var ** .5)

        return model_f_mu, model_f_var, z_stat

    @classmethod
    def get_pval(cls, z_stat):
        """ computes FWER adjusted pval (percentile within max per permute)
        """
        # max z_stat per permutation (sorted from low to high)
        z_stat_max = np.sort(z_stat.max(axis=1))

        num_perm, num_reg = z_stat.shape
        pval = np.full(num_reg, fill_value=-1, dtype=float)
        for reg_idx, z in enumerate(z_stat[0, :]):
            pval[reg_idx] = 1 - bisect_left(z_stat_max, z) / num_perm

        return pval

    def discover(self, alpha):
        raise NotImplementedError
