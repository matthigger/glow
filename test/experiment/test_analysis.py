from hrba.experiment import *
from .test_exper import get_rand_exp


def test_run():
    # build experiment with strong effect to be found (whole region)
    shape = 10, 10
    mask = np.ones(shape, dtype=bool)
    exp = get_rand_exp(shape=shape, a=2, b=1, add_effect=True, seed=0)

    ana_hrba = AnalysisHRBA(exp=exp, alpha=.05, n_permute=25)
    ana_hrba.run(verbose=True)

    # one large effect found
    assert len(ana_hrba.effect_tup) == 1
    assert np.allclose(ana_hrba.effect_tup[0].mask, mask)
