from collections import namedtuple

from glow.mask import *


def test_get_mask_idx():
    mask = np.ones((4, 4))
    mask[0, :] = 0
    mask_idx = get_mask_idx(mask)

    mask_idx_expect = np.array([[-1, -1, -1, -1],
                                [0, 1, 2, 3],
                                [4, 5, 6, 7],
                                [8, 9, 10, 11]])

    assert np.array_equal(mask_idx, mask_idx_expect)


def test_get_entropy():
    case_list = [
        # all negatives -> no valid voxels -> entropy = 0
        (np.array([-1, -1, -1]), 0.0),

        # single label only -> no uncertainty -> entropy = 0
        (np.zeros(6, dtype=int), 0.0),

        # two labels, equal counts -> max entropy for 2 labels -> 1 bit
        (np.array([0, 0, 1, 1]), 1.0),

        # negatives ignored, remaining equal counts of 2 labels -> 1 bit
        (np.array([-1, 0, 1, -1, 0, 1]), 1.0),

        # complex distribution: 3xlabel 0, 1xlabel 1 -> H ~ 0.8113 bits
        (np.array([[0, -1, 0],
                   [1, -1, 0]]),
         -(3 / 4 * np.log2(3 / 4) + 1 / 4 * np.log2(1 / 4))),
    ]

    for label_map, h_exp in case_list:
        assert np.isclose(get_entropy(label_map), h_exp)


def test_get_score():
    # validates the same dice/sens/spec metrics as
    # test_graph.py::test_get_dice_sens_spec, but via get_score's sklearn
    # voxel-level path (the other exercises the node_sum tree path)
    Case = namedtuple('Case', ['y_true', 'y_pred', 'dice', 'sens', 'spec',
                               'mask_active'])

    cases = [
        # Normal mix of positives/negatives
        Case(
            y_true=np.array([0, 0, 1, 1], dtype=int),
            y_pred=np.array([0, 1, 1, 0], dtype=int),
            dice=0.5,
            sens=0.5,
            spec=0.5,
            mask_active=None
        ),
        # No actual positives → sens=0, spec computed normally
        Case(
            y_true=np.array([0, 0, 0, 0], dtype=int),
            y_pred=np.array([0, 1, 0, 1], dtype=int),
            dice=0.0,
            sens=0.0,
            spec=0.5,
            mask_active=None
        ),
        # No actual negatives → spec=1
        Case(
            y_true=np.array([1, 1, 1, 1], dtype=int),
            y_pred=np.array([1, 0, 1, 0], dtype=int),
            dice=4 / 6,
            sens=0.5,
            spec=1.0,
            mask_active=None
        ),
        # With mask_active filtering
        Case(
            y_true=np.array([0, 0, 1, 1], dtype=int),
            y_pred=np.array([0, 1, 1, 0], dtype=int),
            dice=0.0,
            sens=0.0,
            spec=0.5,
            mask_active=np.array([True, True, False, False])
        )
    ]

    for i, case in enumerate(cases, start=1):
        dice, sens, spec = get_score(
            mask_pred=case.y_pred,
            mask_target=case.y_true,
            mask_active=case.mask_active
        )
        assert np.isclose(dice, case.dice), f'Case {i} failed Dice'
        assert np.isclose(sens, case.sens), f'Case {i} failed Sensitivity'
        assert np.isclose(spec, case.spec), f'Case {i} failed Specificity'


Case = namedtuple("Case", ["conn", "not_reflexive", "offset_exp"])


def test_get_neighbor_offsets():
    case_list = [
        Case(conn=np.array([1, 1, 1]),
             not_reflexive=False,
             offset_exp=np.array([[-1],
                                  [0],
                                  [1]])),
        Case(conn=np.array([1, 1, 1]),
             not_reflexive=True,
             offset_exp=np.array([[-1],
                                  [1]])),
        Case(conn=np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]]),
             not_reflexive=True,
             offset_exp=np.array([[-1, 0],
                                  [1, 0],
                                  [0, -1],
                                  [0, 1]])),
        Case(conn=np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]),
             not_reflexive=True,
             offset_exp=np.array([[-1, -1],
                                  [-1, 0],
                                  [-1, 1],
                                  [0, -1],
                                  [0, 1],
                                  [1, -1],
                                  [1, 0],
                                  [1, 1]])),
        Case(conn=6,
             not_reflexive=True,
             offset_exp=np.array([[0, 0, -1],
                                  [0, 0, 1],
                                  [0, -1, 0],
                                  [0, 1, 0],
                                  [-1, 0, 0],
                                  [1, 0, 0]])),
        Case(conn=18,
             not_reflexive=True,
             offset_exp=np.array([[0, 0, -1],
                                  [0, 0, 1],
                                  [0, -1, 0],
                                  [0, 1, 0],
                                  [-1, 0, 0],
                                  [1, 0, 0],
                                  [0, -1, -1],
                                  [0, -1, 1],
                                  [0, 1, -1],
                                  [0, 1, 1],
                                  [-1, 0, -1],
                                  [-1, 0, 1],
                                  [1, 0, -1],
                                  [1, 0, 1],
                                  [-1, -1, 0],
                                  [-1, 1, 0],
                                  [1, -1, 0],
                                  [1, 1, 0]])),
        Case(conn=26,
             not_reflexive=True,
             offset_exp=np.array([[x, y, z]
                                  for x in (-1, 0, 1)
                                  for y in (-1, 0, 1)
                                  for z in (-1, 0, 1)
                                  if not (x == y == z == 0)]))
    ]

    for idx, case in enumerate(case_list):
        offset = get_neighbor_offsets(case.conn,
                                      not_reflexive=case.not_reflexive)
        offset_set = set(tuple(ijk) for ijk in offset)
        offset_exp_set = set(tuple(ijk) for ijk in case.offset_exp)
        assert offset_set == offset_exp_set, f'case: {idx}'


def test_iter_neighbor():
    # 8-connectivity in 2d (center excluded -> non-reflexive)
    conn2d = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # a 3x3 grid of distinct values:
    #   0 1 2
    #   3 4 5
    #   6 7 8
    a = np.arange(9).reshape(3, 3)

    # interior voxel (1,1)=4: all 8 surrounding voxels are in-bounds neighbours
    assert set(iter_neighbor(ijk=(1, 1), a=a, conn=conn2d)) == \
        {0, 1, 2, 3, 5, 6, 7, 8}

    # corner voxel (0,0)=0: only the 3 in-bounds neighbours remain after the
    # out-of-grid offsets are clipped -- exercises glow's bounds handling
    assert set(iter_neighbor(ijk=(0, 0), a=a, conn=conn2d)) == {1, 3, 4}

    # edge voxel (0,1)=1: 5 in-bounds neighbours
    assert set(iter_neighbor(ijk=(0, 1), a=a, conn=conn2d)) == {0, 2, 3, 4, 5}

    # mask_active drops any neighbour at a False position -- exercises glow's
    # mask handling. Mask out voxels 3 and 5; the interior voxel (1,1) should
    # then see only the remaining active neighbours.
    mask_active = np.ones((3, 3), dtype=bool)
    mask_active[1, 0] = False  # value 3
    mask_active[1, 2] = False  # value 5
    assert set(iter_neighbor(ijk=(1, 1), a=a, conn=conn2d,
                             mask_active=mask_active)) == {0, 1, 2, 6, 7, 8}


