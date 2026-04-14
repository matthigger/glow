import filecmp
import os
import tempfile
from pprint import pformat

from glow import __file__ as glow_file
from glow.experiment import *
from glow.analysis import *
from glow.analysis.cluster import cluster
from glow.plot import image_iter, make_gif

folder_glow = pathlib.Path(glow_file).resolve().parents[1]
folder_test_data = folder_glow / 'test' / 'data'

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


def test_make_gif():
    # load single image, bootstrap a few more (no noise), sample rand x
    exp = ExperimentImageOnly.from_search(folder=folder_test_data,
                                          sbj_regex='squares_test.png',
                                          img_glob_dict={'color': '*test.png'})
    exp = exp.bootstrap_img(n=10, noise_scale=0, seed=0)
    exp = exp.sample_x(a=2, seed=0, add_bias=True)
    children = cluster(exp=exp)

    file_obs = tempfile.NamedTemporaryFile(suffix='.gif').name
    file_exp = folder_test_data / 'squares_test_cluster.gif'

    make_gif(file_out=file_obs,
             n_list=30, min_n=5, fps=10, mask_idx=exp.mask_idx,
             children=children, num_vox=np.prod(exp.mask_idx.shape))

    assert filecmp.cmp(file_obs, file_exp)

    os.remove(file_obs)
