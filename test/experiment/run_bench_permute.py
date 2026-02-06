"""benchmark: freed_lane construction speed

named run_bench_* so pytest won't auto-run it.
run manually: python -m test.experiment.run_bench_permute
"""

import time

import numpy as np

from glow.experiment import ExperimentImageOnly
from glow.experiment.permute import get_freed_lane
from test.experiment.test_permute import _get_freed_lane_dense


def bench_get_freed_lane():
    """index-based construction avoids dense permutation matrix multiply"""
    perm_idx = 1
    n_rep = 200

    print()
    for num_img in (200, 1000):
        exp = ExperimentImageOnly.from_gauss(seed=0, b=3, num_img=num_img,
                                             shape=(100,))
        exp = exp.sample_x(a=2, add_bias=True)

        # warm up
        get_freed_lane(exp.x, exp.contrast, perm_idx)
        _get_freed_lane_dense(exp.x, exp.contrast, perm_idx)

        t0 = time.perf_counter()
        for _ in range(n_rep):
            fl_new = get_freed_lane(exp.x, exp.contrast, perm_idx)
        t_new = (time.perf_counter() - t0) / n_rep

        t0 = time.perf_counter()
        for _ in range(n_rep):
            fl_old = _get_freed_lane_dense(exp.x, exp.contrast, perm_idx)
        t_old = (time.perf_counter() - t0) / n_rep

        speedup = t_old / t_new
        print(f'  num_img={num_img:4d}: old={t_old*1000:.2f}ms  '
              f'new={t_new*1000:.2f}ms  speedup={speedup:.1f}x')
        assert np.allclose(fl_new, fl_old)

    # index-based row selection should be faster than P @ M matmul
    assert speedup > 2.0, f'expected speedup at num_img=1000, got {speedup:.2f}x'
    print('\n✓ bench_get_freed_lane PASSED')


if __name__ == '__main__':
    bench_get_freed_lane()
