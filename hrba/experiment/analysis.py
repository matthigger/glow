from .epoch import Epoch


class AnalysisHRBA:
    """
    Attributes:
        alpha (float): upper bound on family wise error rate
        n_permute (int): number of permutations to run (note that the
            unpermuted stats are assigned "permutation" index 0 so that
            n_permute + 1 "permutations" are run)
        epoch_list (list): Epoch of region discovery
    """

    def __init__(self, exp, alpha=.05, n_permute=100):
        self.exp = exp
        self.alpha = alpha
        self.n_permute = n_permute
        self.epoch_list = list()

    def run(self, **kwargs):
        """ runs analysis to find all significant regions in experiment
        """
        while True:
            # build new epoch
            epoch = Epoch(exp=self.exp, n_permute=self.n_permute,
                          alpha=self.alpha, **kwargs)
            self.epoch_list.append(epoch)

            if not self.epoch_list[-1].discovered:
                break

            # build new experiment, adjust for previously discovered regions
            raise NotImplementedError
