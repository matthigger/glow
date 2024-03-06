from joblib import Parallel, delayed

from hglm.experiment.epoch import *
from test.experiment.test_exper import get_rand_exp

num_experiments = 1000

num_img = 10
shape = 3, 3
a = 2
b = 1


def run_exp(seed):
    exp = get_rand_exp(shape=shape, a=a, b=b, seed=seed, num_img=num_img)
    epoch = Analysishglm(exp, n_permute=25, n_permute_z=20)

    return bool(epoch.effect_list)


n_effect_found = sum(Parallel(n_jobs=-1)(
    delayed(run_exp)(i) for i in range(num_experiments)))

print(f'{n_effect_found} effects found in {num_experiments} experiments')
