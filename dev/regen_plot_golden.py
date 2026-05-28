"""Regenerate the golden text dumps for ``glow.plot.image_iter``.

This is a developer utility, NOT a test.  ``test/test_plot.py`` used to
carry this function (as ``build_case``); running it would silently
overwrite the expected output, so it lives here instead.

``test_plot.py`` now asserts observable invariants directly and no longer
depends on these .txt files, but the human-readable dumps are kept as
documentation of expected behavior.  Regenerate them with::

    python dev/regen_plot_golden.py

and review the diff before relying on the new output.
"""
import pathlib
from pprint import pformat

import numpy as np

from glow import __file__ as glow_file
from glow.plot import image_iter

folder_glow = pathlib.Path(glow_file).resolve().parents[1]
folder_test_data = folder_glow / 'test' / 'data'

# the same two cases exercised by test/test_plot.py
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


if __name__ == '__main__':
    build_case()
