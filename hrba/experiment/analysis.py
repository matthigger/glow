import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import grid_to_graph
from tqdm import tqdm

from .permute import get_perm_matrix


class AnalysisHRBA:
    """
    Attributes:
        perm_dendro_dict (dict): keys are permutation indices, values are
            (2, n) dendrogram arrays (equiv to sklearn.cluster.Ward.children_)
    """

    def __init__(self, exp, alpha=.05, n_permute=100):
        self.exp = exp
        self.alpha = alpha
        self.n_permute = n_permute
        self.perm_dendro_dict = dict()

    def run(self):
        """ runs analysis to find all significant regions in experiment
        """
        # ward per permuatation
        self.cluster()

        # compute f-stat per region in all permutations

        #

    def cluster(self, verbose=True):
        """ build perm_dendro_dict """
        # prep
        b, num_img, num_vox = self.exp.y.shape
        x = self.exp.x[~self.exp.contrast, :], self.exp.x
        h = [np.linalg.pinv(_x) @ _x for _x in x]
        h_diff = h[1] - h[0]
        i = np.eye(num_img)

        # get connectivity (ensures only neighboring voxels joined)
        mask = self.exp.mask_idx >= 0
        if mask.ndim == 3:
            shape = mask.shape
        elif mask.ndim == 2:
            shape = (*mask.shape, 1)

        # prep ward clustering object
        connectivity = grid_to_graph(*shape, mask=mask)
        ward = AgglomerativeClustering(connectivity=connectivity,
                                       linkage='ward')
        tqdm_dict = dict(desc='clustering per permutation',
                         disable=not verbose)
        for perm_idx in tqdm(range(self.n_permute + 1), **tqdm_dict):
            # permute data residuals under reduced model (freedman lane)
            p = get_perm_matrix(perm_idx, num_img)
            freed_lane = (i - h[0]) @ p + h[0]

            # prepare y
            y = np.einsum('ijk,jm->imk', self.exp.y, freed_lane @ h_diff)
            y = y.reshape((-1, num_vox))

            # cluster & store
            ward.fit(y.T)
            self.perm_dendro_dict[perm_idx] = ward.children_
