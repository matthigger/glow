from itertools import chain
from warnings import warn

from .epoch import EpochHRBA, EpochTFCE


class Analysis:
    """
    Attributes:
        alpha (float): upper bound on family wise error rate
        n_permute (int): number of permutations to run (note that the
            unpermuted stats are assigned "permutation" index 0 so that
            n_permute + 1 "permutations" are run in an epoch)
        epoch_list (list): Epoch of region discovery
        effect_tup (tuple): tuple of effects discovered
    """

    def __init__(self, exp, alpha=.05, n_permute=100):
        self.exp = exp
        self.alpha = alpha
        self.n_permute = n_permute
        self.epoch_list = list()
        self.effect_tup = None

    def run(self, verbose=True, max_epoch=1e8):
        """ runs analysis to find all significant regions in experiment
        """
        assert not self.epoch_list, 'Analysis may only be run once'

        exp = self.exp
        for epoch_idx in range(int(max_epoch)):
            if verbose:
                print(f'begin epoch {epoch_idx}')

            # run & store new epoch
            epoch = self.Epoch(exp=exp, n_permute=self.n_permute,
                               alpha=self.alpha,
                               verbose=verbose)
            self.epoch_list.append(epoch)

            if verbose:
                n = len(epoch.effect_list)
                print(f'{n} significant regions discovered in this epoch')
            if not self.another_epoch_needed():
                # no new regions discovered, analysis complete
                break

            for effect in epoch.effect_list:
                # build new experiment which removes impact of this effect
                exp = exp.rm_effect(effect)
        else:
            warn(f'stopping.  epoch limit reached: {max_epoch} epochs')

        # collect all effects in one place
        self.effect_tup = tuple(chain.from_iterable(e.effect_list for e in
                                                    self.epoch_list))


class AnalysisHRBA(Analysis):
    Epoch = EpochHRBA

    def another_epoch_needed(self):
        # more epochs needed so long as effects are found
        return bool(self.epoch_list[-1].effect_list)


class AnalysisTFCE(Analysis):
    Epoch = EpochTFCE

    def another_epoch_needed(self):
        # tfce only needs a single epoch
        return False
