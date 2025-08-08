from collections import namedtuple

from hglm.mask import *


def test_get_mask_idx():
    mask = np.ones((4, 4))
    mask[0, :] = 0
    mask_idx = get_mask_idx(mask)

    mask_idx_expect = np.array([[-1, -1, -1, -1],
                                [0, 1, 2, 3],
                                [4, 5, 6, 7],
                                [8, 9, 10, 11]])

    assert np.array_equal(mask_idx, mask_idx_expect)


def test_get_score():
    Case = namedtuple('Case', ['y_true', 'y_pred', 'f1', 'sens', 'spec',
                               'mask_active'])

    cases = [
        # Normal mix of positives/negatives
        Case(
            y_true=np.array([0, 0, 1, 1], dtype=int),
            y_pred=np.array([0, 1, 1, 0], dtype=int),
            f1=0.5,
            sens=0.5,
            spec=0.5,
            mask_active=None
        ),
        # No actual positives → sens=0, spec computed normally
        Case(
            y_true=np.array([0, 0, 0, 0], dtype=int),
            y_pred=np.array([0, 1, 0, 1], dtype=int),
            f1=0.0,
            sens=0.0,
            spec=0.5,
            mask_active=None
        ),
        # No actual negatives → spec=1
        Case(
            y_true=np.array([1, 1, 1, 1], dtype=int),
            y_pred=np.array([1, 0, 1, 0], dtype=int),
            f1=4 / 6,
            sens=0.5,
            spec=1.0,
            mask_active=None
        ),
        # With mask_active filtering
        Case(
            y_true=np.array([0, 0, 1, 1], dtype=int),
            y_pred=np.array([0, 1, 1, 0], dtype=int),
            f1=0.0,
            sens=0.0,
            spec=0.5,
            mask_active=np.array([True, True, False, False])
        )
    ]

    for i, case in enumerate(cases, start=1):
        f1, sens, spec = get_score(
            mask_pred=case.y_pred,
            mask_target=case.y_true,
            mask_active=case.mask_active
        )
        assert np.isclose(f1, case.f1), f'Case {i} failed F1'
        assert np.isclose(sens, case.sens), f'Case {i} failed Sensitivity'
        assert np.isclose(spec, case.spec), f'Case {i} failed Specificity'
