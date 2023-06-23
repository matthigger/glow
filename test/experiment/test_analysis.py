from hrba.experiment import *
from .test_exper import get_rand_exp

# build experiment with strong effect to be found (whole region)
shape = 5, 5, 5
mask = np.ones(shape, dtype=bool)
exp = get_rand_exp(shape=shape, a=2, b=1, add_effect=True, seed=0)


def test_run():
    for cls in (AnalysisTFCE, AnalysisHRBA):
        analysis = cls(exp=exp, alpha=.05, n_permute=25)
        analysis.run(verbose=True)

        # one large effect found
        if not len(analysis.effect_tup) == 1:
            raise AssertionError(f'no single region found: {cls}')

        if not np.allclose(analysis.effect_tup[0].mask, mask):
            raise AssertionError(f'mask doesnt correspond to effect: {cls}')
