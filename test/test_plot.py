import numpy as np

from glow.plot import image_iter

# Two hierarchies over a 2x2 image (mask_idx == [[0, 1], [2, 3]], num_vox=4).
# case0: merges only touch in-mask regions; case1 mixes in some merge events
# whose endpoints are outside the mask (large indices) which image_iter must
# ignore.  Both should drive the image from one-color-per-voxel down to a
# single root color.
case = dict(mask_idx=np.arange(4).reshape((2, 2)),
            children=np.arange(6).reshape((3, 2)),
            num_vox=4), \
    dict(mask_idx=np.arange(4).reshape((2, 2)),
         children=np.array(
             [[0, 1], [100, 101], [4, 1000], [1001, 5], [3, 2], [8, 6]]),
         num_vox=4)


def test_image_iter():
    """image_iter walks a Ward hierarchy and re-colours the image as
    regions merge.  Rather than diff against a golden dump, assert the
    invariants the docstring promises:

      * the first frame has one colour per voxel (num_vox colours);
      * the final frame has a single colour (one tree root here);
      * the number of distinct regions never increases between frames
        (merges only ever reduce the colour count);
      * at every frame the colour count, the distinct colours actually
        painted into the image, and the distinct labels in mask_idx_current
        all agree.

    NB: image_iter yields the *same* mutated image/dict objects each step,
    so each frame must be inspected inline (not collected into a list first).
    """
    for idx, _case in enumerate(case):
        num_vox = _case['num_vox']
        counts = []
        for fi, (image, mask_idx_current, color_dict) in enumerate(
                image_iter(**_case)):
            # image is an rgb uint8 array shaped like the mask
            assert image.shape == (*_case['mask_idx'].shape, 3), f'case{idx}'
            assert image.dtype == np.uint8, f'case{idx}'

            n_regions = len(color_dict)
            # the colours painted into the image and the labels in the
            # current mask both track color_dict one-for-one
            painted = set(map(tuple, image.reshape(-1, 3).tolist()))
            labels = set(int(x) for x in mask_idx_current.ravel().tolist())
            assert len(painted) == n_regions, f'case{idx} frame{fi}'
            assert len(labels) == n_regions, f'case{idx} frame{fi}'

            if fi == 0:
                # first frame: one colour per voxel
                assert n_regions == num_vox, f'case{idx}'
            counts.append(n_regions)

        # final frame collapses to the tree roots (a single region here)
        assert counts[-1] == 1, f'case{idx}'
        # region count is monotonically non-increasing as regions merge
        assert all(counts[i] >= counts[i + 1]
                   for i in range(len(counts) - 1)), f'case{idx} {counts}'
