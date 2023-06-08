import pathlib
import re
from copy import copy

import numpy as np
import pandas as pd

from hrba.sample_effect import compute_offset
from .effect import Effect
from .load_image import load_image_color, load_image_nii


class Experiment:
    """ contains source data to identify regions with some effect

    an effect is a linear mapping from x to y within some contiguous set of
    voxels:

    y = a_0 + a_1 x_1 + a_2 x_2 + ...

    Attributes:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        mask_idx (np.array): same shape as image.  -1 where voxel not
            included in analysis, otherwise contains voxel index
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        x_names  (list): (a) names of each x feature (defaults to None)
        y_names (list): (b) names of each y feature (defaults to None)
    """

    def __init__(self, *, x=None, y, mask_idx, contrast=None, x_names=None,
                 y_names=None):
        self.x = x
        self.y = y
        self.mask_idx = mask_idx
        self.contrast = contrast
        self.x_names = x_names
        self.y_names = y_names

    @classmethod
    def from_search(cls, folder, sbj_regex, img_glob_dict, **kwargs):
        """ searches path for matching image files

        Args:
            folder (str): folder to search (recursively) for image files
            sbj_regex (str): regex string which matches on subject names,
                must match once on each image's full path
            img_glob_dict (dict): keys are str names of each image type
                (e.g. 'FA', 'MD'), values are glob strs (e.g. '*_FA.nii.gz')

        Returns:
            experiment (Experiment): experiment generated from all images found
        """
        # find all images
        folder = pathlib.Path(folder)
        df = pd.DataFrame()
        for y_feat, y_glob in img_glob_dict.items():
            for file in folder.glob(y_glob):
                # extract sbj from full path of file
                sbj_list = re.findall(sbj_regex, str(file))
                assert len(sbj_list) == 1, \
                    f'unique sbj not found in file: {file}'
                sbj = sbj_list[0]

                # todo: check that duplicate file doesn't already exist
                df.loc[sbj, y_feat] = file

        assert df.size, 'no images found'

        # check if any subject is missing any imaging feature
        s_missing = df.isna().mean(axis=1)
        if s_missing.any():
            print('some sbj missing files:')
            print(df.loc[s_missing, :].notnull().astype('int'))

        nii_in_file = ['.nii' in str(file) for file in df.values.flatten()]
        if all(nii_in_file):
            feat_sbj_img, mask_idx = load_image_nii(df)
        elif not any(nii_in_file):
            feat_sbj_img, mask_idx = load_image_color(df)
        else:
            raise TypeError('may not mix nifti and color images in input')

        # mask into each image, store as y
        mask = mask_idx >= 0
        y_names = sorted(feat_sbj_img.keys())
        y = np.empty((len(y_names), df.shape[0], mask.sum()))
        for sbj_idx, sbj in enumerate(sorted(df.index)):
            for feat_idx, feat in enumerate(y_names):
                y[feat_idx, sbj_idx, :] = feat_sbj_img[feat][sbj][mask]

        return cls(y=y, y_names=y_names, mask_idx=mask_idx, **kwargs)

    def bootstrap_img(self, n, seed=None, noise_scale=1):
        """ returns a new Experiment which bootstrap resamples images

        Args:
            n (int): number of images in the resulting Experiment
            seed: used for random number generator
            noise_scale (float): scales noise std_dev.  noise is drawn
                independently, per voxel, from a normal distribution whose
                covariance is the (b, b) sample covariance of self.y

        Returns:
            exp_out (Experiment): y has been bootstrap resampled from self
        """
        # bootstrap sample images
        rng = np.random.default_rng(seed=seed)
        b, num_img_init, num_vox = self.y.shape
        img_idx = rng.choice(num_img_init, n, replace=True)
        self.y = self.y[:, img_idx, :]

        # add noise
        assert noise_scale >= 0, 'snr cannot be negative'
        if noise_scale > 0:
            cov = np.cov(self.y.reshape((b, -1)))
            noise = rng.multivariate_normal(mean=np.zeros(b),
                                            cov=cov * (noise_scale ** 2),
                                            size=num_vox * n)
            self.y += noise.T.reshape((b, n, num_vox))

    def load_x(self):
        """ loads a csv of real x data (as opposed to sampled x data) """
        raise NotImplementedError

    def sample_x(self, a=None, contrast=None, seed=None):
        """ generates (or replaces) x with an arbitrary std normal noise

        Args:
            a (int): number of x features in output
            contrast (np.array): (a) True for each corresponding feature in x
                which is "of interest" (other x features form the reduced
                model in computing f statistic)
            seed: used for random number generator

        Returns:
            exp_out (Experiment): x has been replaced with noise from self
        """
        assert (a is None) != (contrast is None), 'a xor contrast required'

        if a is None:
            # contrast specified, extract a from it
            a = contrast.size
            self.contrast = contrast
        else:
            # default contrast: all x of interest but bias term
            self.contrast = np.ones(a, dtype=bool)
            self.contrast[0] = False

        # sample x
        num_img = self.y.shape[1]
        rng = np.random.default_rng(seed=seed)
        self.x = rng.standard_normal(size=(a, num_img))

    def impose_effect(self, extenter, seed=None, **kwargs):
        """ builds experiment with effect imposed

        Args:
            extenter (ExtenterSphere or ExtenterMinVar): identifies volume to
                impose effect on
            seed: seed of random number generator (for extent)
            **kwargs: passed to compute_offset(), either p_val or f_stat

        Returns:
            exp (Experiment): an experiment
            effect (Effect): encapsulates
        """
        # sample effect space
        mask = extenter(y=self.y, mask_idx=self.mask_idx, seed=seed)

        # get offset which imposes desired effect strength
        effect_idx = self.mask_idx[mask]
        y_effect = self.y[:, :, effect_idx]
        offset = compute_offset(x=self.x, y=y_effect, contrast=self.contrast,
                                **kwargs)

        # impose effect on y, build new experiment
        y = copy(self.y)
        y[:, :, effect_idx] += offset[..., np.newaxis]
        exp = type(self)(x=self.x, y=y, contrast=self.contrast,
                         mask_idx=self.mask_idx)

        # store effect
        y_effect = y[:, :, effect_idx]
        effect = Effect.from_x_y_contrast(x=self.x, y=y_effect,
                                          contrast=self.contrast,
                                          mask=mask)
        return exp, effect
