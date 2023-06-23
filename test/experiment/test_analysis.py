from hrba.experiment import *
from hrba.sample_effect import ExtenterSphere
from .test_exper import get_rand_exp

# build experiment with strong effect to be found (whole region)
shape = 5, 5
exp = get_rand_exp(shape=shape, a=2, b=1, seed=0)
exp, effect = exp.impose_effect(seed=0, extenter=ExtenterSphere(radius=2),
                                p_val=.0001)


def test_run():
    for cls in (AnalysisTFCE, AnalysisHRBA):
        analysis = cls(exp=exp, alpha=.05, n_permute=25)
        analysis.run(verbose=True)

        # one large effect found
        if not len(analysis.effect_tup) == 1:
            folder = str(analysis.epoch_list[0].to_nii())
            raise AssertionError(f'no single region found: {cls}')

        if not np.allclose(analysis.effect_tup[0].mask, effect.mask):
            raise AssertionError(f'mask doesnt correspond to effect: {cls}')
