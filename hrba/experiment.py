import pathlib
import re
from collections import defaultdict
from copy import copy

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from hrba.sample_effect.extent import get_mask_idx


class Experiment:
    """ identifies regions which show some statistically significant effect

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
        # todo: support passing x as a csv, reconcile with files found and
        #  error on missing

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

        # load images (check affine is consistent, if present)
        affine = None
        feat_sbj_img = defaultdict(dict)
        for feat in df.columns:
            for sbj in df.index:
                file = df.loc[sbj, feat]

                # load image
                if '.nii' in str(file):
                    # load img
                    img = nib.load(file)
                    if affine is None:
                        affine = img.affine
                    assert np.array_equal(img.affine,
                                          affine), 'affine mismatch'
                    feat_sbj_img[feat][sbj] = img.get_fdata()
                else:
                    x = np.array(Image.open(file))
                    assert x.ndim == 2, 'only 2d non-nii supported'
                    # todo: support for rgb split into 3 features here
                    feat_sbj_img[feat][sbj] = x

        # count nonzero voxels per position (also check images have same shape)
        vox_count = None
        for feat, sbj_img in feat_sbj_img.items():
            for sbj, img in sbj_img.items():
                if vox_count is None:
                    vox_count = np.zeros(img.shape)
                vox_count += img != 0

        # build mask_idx
        mask = vox_count == df.size
        mask_idx = get_mask_idx(mask)

        # mask into each image, store as y
        y = np.empty((len(df.columns), df.shape[0], mask.sum()))
        for sbj_idx, sbj in enumerate(sorted(df.index)):
            for feat_idx, feat in enumerate(sorted(df.columns)):
                y[feat_idx, sbj_idx, :] = feat_sbj_img[feat][sbj][mask]

        return cls(y=y, y_names=df.columns, mask_idx=mask_idx, **kwargs)

    def bootstrap_y(self, n, seed=None, snr=1):
        """ returns a new Experiment which bootstrap resamples images

        Args:
            n (int): number of images in the resulting Experiment
            seed: used for random number generator
            snr (float): signal-to-noise ratio, at a snr of 2 the noise std
                deviation is twice the standard deviation of a y feature (
                across all voxels and images)

        Returns:
            exp_out (Experiment): y has been bootstrap resampled from self
        """

        raise NotImplementedError

    def sample_x(self, a, seed=None):
        """ generates (or replaces) x with an arbitrary std normal noise

        Args:
            a (int): number of x features in output
            seed: used for random number generator

        Returns:
            exp_out (Experiment): x has been replaced with noise from self
        """
        # sample x
        num_img = self.y.shape[1]
        rng = np.random.default_rng(seed=seed)
        x = rng.standard_normal(size=(a, num_img))

        # build a new instance which "copies" self, replacing x
        d = copy(self.__dict__)
        d['x'] = x
        return type(self)(**d)
