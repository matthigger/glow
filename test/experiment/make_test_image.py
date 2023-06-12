import pathlib
from collections import defaultdict
from itertools import product

import nibabel as nib
import numpy as np
from PIL import Image

from hrba import __file__ as hrba_file

folder_hrba = pathlib.Path(hrba_file).resolve().parents[1]
folder_test_data = folder_hrba / 'test' / 'data'

# build some dummy data to load
folder_test_data.mkdir(exist_ok=True)

shape = (10, 11)
affine = np.eye(4)
img_feat_intensity = defaultdict(dict)
for intensity, (img_idx, feat_idx) in enumerate(product(range(3),
                                                        range(2))):
    # add 1 to intensity (zero values are considered background)
    intensity += 1
    file = folder_test_data / f'img{img_idx}_feat{feat_idx}.nii.gz'
    x = np.full((shape), fill_value=intensity).astype(float)
    if not file.exists():
        img = nib.Nifti2Image(dataobj=x, affine=affine)
        img.to_filename(file)

    file = folder_test_data / f'img{img_idx}_feat{feat_idx}.jpg'
    if not file.exists():
        Image.fromarray(x.astype(np.uint8)).save(file)

    img_feat_intensity[img_idx][feat_idx] = intensity
