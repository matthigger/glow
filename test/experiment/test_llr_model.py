import numpy as np

from hrba.experiment.llr_model import *


def test_llr_model():
    # build test case
    n = 10000
    m, b, mp, bp = 1, 0, 2, .1

    rng = np.random.default_rng(seed=0)
    size = rng.uniform(low=0, high=1, size=n)
    error = rng.standard_normal(size=n) * np.exp(size * mp + bp) ** .5
    llr = size * m + b + error

    mod = LLRModel()
    mod.fit(llr=np.exp(llr), size=np.exp(size))

    assert np.allclose(np.array([m, b, mp, bp]),
                       np.array([mod.m, mod.b, mod.mp, mod.bp]),
                       rtol=0, atol=1e-1)
