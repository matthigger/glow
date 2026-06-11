from collections import namedtuple

import pandas as pd

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


def test_confusion_counts():
    # voxel-level confusion counts; stats_from_counts then
    # turns these into dice/sens/ppv/spec (tested separately below)
    Case = namedtuple('Case', ['y_true', 'y_pred', 'mask_active',
                               'tp', 'fp', 'tn', 'fn'])

    cases = [
        # normal mix of positives/negatives
        Case(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]), None,
             tp=1, fp=1, tn=1, fn=1),
        # no actual positives
        Case(np.array([0, 0, 0, 0]), np.array([0, 1, 0, 1]), None,
             tp=0, fp=2, tn=2, fn=0),
        # no actual negatives
        Case(np.array([1, 1, 1, 1]), np.array([1, 0, 1, 0]), None,
             tp=2, fp=0, tn=0, fn=2),
        # mask_active keeps only the first two voxels
        Case(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]),
             np.array([True, True, False, False]),
             tp=0, fp=1, tn=1, fn=0),
    ]

    for i, case in enumerate(cases, start=1):
        counts = confusion_counts(mask_pred=case.y_pred, mask_target=case.y_true,
                                  mask_active=case.mask_active)
        assert counts == {'tp': case.tp, 'fp': case.fp,
                          'tn': case.tn, 'fn': case.fn}, f'case {i}'


def test_stats_from_counts():
    # the four test_confusion_counts regions, plus an empty region (all zero) to
    # exercise the 0/0 fills: spec -> 1, the others -> 0
    tp = np.array([1, 0, 2, 0, 0])
    fp = np.array([1, 2, 0, 1, 0])
    tn = np.array([1, 2, 0, 1, 0])
    fn = np.array([1, 0, 2, 0, 0])

    stats = stats_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)

    assert np.allclose(stats['dice'], [0.5, 0.0, 4 / 6, 0.0, 0.0])
    assert np.allclose(stats['sens'], [0.5, 0.0, 0.5, 0.0, 0.0])
    assert np.allclose(stats['ppv'], [0.5, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(stats['spec'], [0.5, 0.5, 1.0, 0.5, 1.0])


def test_stats_from_counts_preserves_series():
    # pandas Series in -> Series out, with the index intact
    idx = pd.Index([10, 20], name='region_idx')
    counts = {k: pd.Series([v, 0], index=idx)
              for k, v in (('tp', 1), ('fp', 1), ('tn', 1), ('fn', 1))}

    stats = stats_from_counts(**counts)

    for key in ('dice', 'sens', 'ppv', 'spec'):
        assert isinstance(stats[key], pd.Series)
        assert stats[key].index.equals(idx)
    # the all-zero second region takes the fills
    assert stats['spec'].iloc[1] == 1.0
    assert stats['ppv'].iloc[1] == 0.0


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


