import pathlib
from pprint import pformat

import numpy as np

from hrba import __file__ as hrba_file
from hrba.plot import image_iter

folder_hrba = pathlib.Path(hrba_file).resolve().parents[1]
folder_test_data = folder_hrba / 'test' / 'data'

case = dict(mask_idx=np.arange(4).reshape((2, 2)),
            children=np.arange(6).reshape((3, 2)),
            num_vox=4), \
    dict(mask_idx=np.arange(4).reshape((2, 2)),
         children=np.array(
             [[0, 1], [100, 101], [4, 1000], [1001, 5], [3, 2], [8, 6]]),
         num_vox=4)


def build_case():
    """ be sure to check output of txt files before relying on test cases!

    we choose to output human-readable txt to document expected behavior"""

    for idx, _case in enumerate(case):
        # run image_iter, get string
        l = list()
        for image, mask_idx_current, color_dict in image_iter(**_case):
            l.append(pformat(dict(image=image,
                                  mask_idx_current=mask_idx_current,
                                  color_dict=color_dict)))

        # write to file
        file = folder_test_data / f'image_iter_case{idx}.txt'
        with open(file, 'w') as f:
            print('\n'.join(l), end='', file=f)


def test_image_iter():
    for idx, _case in enumerate(case):
        # run image_iter, get string
        l = list()
        for image, mask_idx_current, color_dict in image_iter(**_case):
            l.append(pformat(dict(image=image,
                                  mask_idx_current=mask_idx_current,
                                  color_dict=color_dict)))
        s_obs = '\n'.join(l)

        # compare to file
        file = folder_test_data / f'image_iter_case{idx}.txt'
        with open(file) as f:
            s_exp = f.read()

        assert s_obs == s_exp, f'case{idx}'
