class ValidationCase:
    """
    Attributes:
        seed:
        mask (np.array):
        perc (float):
        beta (np.array):
        f_stat (float):
    """

class ValidationCaseBuilder:
    """ builds ExperimentValidation, and experiment with known, imposed effect

    Attributes:
        effect_size (int): number of voxels contained in effect region
        perc (float): effect severity, measured as a p-value under a typical
            f-test
        extenter (Extenter): builds the effect's extent
        no_effect_dilate (int): if a voxel is within this range of an effect
            voxel, it is included as a "no effect" voxel in the experiment.  by
            default this is None, and all voxels are included in experiment
            (useful for speedup)
    """

    def __init__(self, effect_size, perc, extenter, no_effect_dilate):
        self.effect_size = effect_size
        self.perc = perc
        self.extenter = extenter
        self.no_effect_dilate = no_effect_dilate

    def impose(self, experiment, seed):
        # build extent
        mask = self.extenter(mask_idx=experiment.mask_idx,
                             y=experiment.y,
                             seed=seed)

        # build offset to y to impose effect within mask

        # construction ExperimentValidation
        pass
