import pytest
from scipy.ndimage import label

from glow.experiment import *
from glow.analysis.cluster import *
from glow.graph import get_dice_sens_spec
from test.experiment.make_test_image import folder_test_data


def test_cluster():
    # in a population of identical test images, clustering should
    # segment based on color

    # load single image, bootstrap a few more (no noise), sample rand x
    exp = ExperimentImageOnly.from_search(folder=folder_test_data,
                                          sbj_regex='squares',
                                          img_glob_dict={'color': '*test.png'})

    exp = exp.bootstrap_img(n=10, noise_scale=0)
    exp = exp.sample_x(a=2, add_bias=True)

    # cleave mask into many pieces (forest clustering)
    mask = exp.mask_idx > -1
    mask[:, mask.shape[1] // 2] = False
    mask[mask.shape[0] // 2, :] = False
    exp_cleave = exp.apply_mask(mask)
    children_forest = cluster(exp=exp_cleave)
    num_vox_cleave = int((exp_cleave.mask_idx > -1).sum())
    assert children_forest.shape[0] < num_vox_cleave, \
        'forest should have fewer internal nodes than a single tree'

    # cluster the full mask (should collect all areas of consistent color)
    children = cluster(exp=exp)

    dice_list = list()
    # cast to uint8 to avoid floating point precision comparison error
    y0 = exp.y[:, 0, :].astype(np.uint8)
    unique_colors = np.unique(np.unique(y0, axis=-1), axis=-1)
    for color in unique_colors.T:
        # build img_color, True at every voxel which given color
        color_mask = (y0.T == color).all(axis=-1)
        img_color = np.zeros(exp.mask_idx.shape, dtype=bool)
        for idx in np.where(color_mask)[0]:
            img_color[exp.mask_idx == idx] = True

        # each contiguous uniformly-colored region should appear as a node
        _label, n_regions = label(img_color)
        for idx in range(1, n_regions + 1):
            mask = _label == idx
            dice = get_dice_sens_spec(mask=mask, mask_idx=exp.mask_idx,
                                  children=children)[0]
            dice_list.append(max(dice))

    # every contiguous color region is recovered exactly by some tree node
    assert dice_list, 'no color regions found'
    np.testing.assert_allclose(dice_list, 1.0)
