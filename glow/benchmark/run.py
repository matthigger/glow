import gzip
import json
import pathlib
import time
import traceback
from uuid import uuid4

import cloudpickle as pickle
import numpy as np

import glow
from glow.benchmark.file import OUT, ERROR


def run(seed, hotel_tr, config):
    """ runs a single experiment (one hotel_tr & seed) """

    # trim experiment to reasonable size (for speedup)
    if config.radius is None:
        exp = config.exp
    else:
        extenter = glow.effect.ExtenterSphere(radius=config.radius)
        mask = extenter(mask_idx=config.exp.mask_idx, seed=seed,
                        contiguous=True)
        exp = config.exp.apply_mask(mask)

    # scale normalize before sampling minimum variance (each feature given
    # equal weight in sampling extent)
    exp = glow.experiment.ExperimentScaled.from_exp(exp)

    # sample effect space
    n = exp.y.shape[2] * config.effect_perc
    extenter = glow.effect.ExtenterMinVar(n=n)

    # impose effect
    _exp, effect = exp.impose_effect(extenter=extenter,
                                     seed=seed,
                                     hotel_tr=hotel_tr)

    for label, (Ana, kwargs) in config.ana_kwargs_dict.items():
        # cluster extent thresholding "peeks", its results should be
        # considered as an upper bound as this isn't feasible in practice
        if kwargs.get('cet_flag', False):
            kwargs['mask_eff'] = effect.mask

        # prep output file
        uuid = str(uuid4())[:8]
        file_out = config.folder / OUT / f'{uuid}_result.json'

        # run analysis
        start = time.time()
        if config.error_save:
            # catch errors and dump to json if any occur (allows us to
            # continue with experiment in event of errors)
            try:
                ana = Ana(exp=_exp, alpha_fwer=config.alpha_fwer, **kwargs)
            except Exception as e:
                d = {'error_msg': traceback.format_exc(),
                     'label': label,
                     'method': Ana.__name__,
                     'hotel_tr': hotel_tr,
                     'seed': int(seed)}
                print(f'error: {d}')

                file_out = str(file_out).replace(OUT, ERROR)
                file_out = pathlib.Path(file_out)
                file_out.parent.mkdir(exist_ok=True, parents=True)
                with open(file_out, 'w') as f:
                    json.dump(d, f, sort_keys=True, indent=4)
                continue

        else:
            # no error catching, will stop all experiments if any error
            ana = Ana(exp=_exp, alpha_fwer=config.alpha_fwer, **kwargs)
        total_time_sec = time.time() - start

        # build mask of predicted area (union of all effect masks)
        mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
        for _effect in ana.effect_list:
            mask_pred |= _effect.mask

        # compute scores
        f1, sens, spec = glow.mask.get_score(mask_pred=mask_pred,
                                             mask_target=effect.mask,
                                             mask_active=exp.mask_idx > -1)
        # dump summary
        d = {'hotel_tr': hotel_tr,
             'seed': int(seed),
             'stat': ana.get_stat.__name__.replace('get_', ''),
             'label': label,
             'Analysis': Ana.__name__,
             'f1': f1,
             'sens': sens,
             'spec': spec,
             'uuid': uuid,
             'vox_total': int(ana.exp.y.shape[2]),
             'vox_effect': int(effect.mask.sum()),
             'time_sec': total_time_sec}
        file_out.parent.mkdir(exist_ok=True, parents=True)
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)

        if config.detail_save:
            # dump detail
            file_out = config.folder / OUT / f'{uuid}_detail.p.gz'
            with gzip.open(file_out, 'wb') as f:
                pickle.dump((ana, effect), f)


if __name__ == '__main__':
    from glow.benchmark.config import Config

    # quick test
    ana_kwargs_dict = {'GLOW': (glow.experiment.AnalysisGLOW,
                                dict(n_perm=100)),
                       'VBA': (glow.experiment.AnalysisVBA, dict(n_perm=100))}
    config = Config(label='quick_test', source='wgn', n_seed=3,
                    hotel_tr_all=[0, 1], wgn_shape=(3, 3),
                    ana_kwargs_dict=ana_kwargs_dict)
    config.run_all()
