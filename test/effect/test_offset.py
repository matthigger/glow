from hglm.effect.impose import *
from hglm.f_stat import *
from ..helper import generate_dummy_data


def test_compute_offset():
    for seed in range(1):
        # generate dummy data
        x, y, contrast = generate_dummy_data(seed=seed)
        b, num_img, num_vox = y.shape

        for pval_exp in np.logspace(0, -3, 5):
            for rough_exp in (None,):
                # compute offset and space_cov_scale needed to achieve f stat
                offset = compute_offset(x=x, y=y,
                                        contrast=contrast,
                                        pval=pval_exp,
                                        rough=rough_exp)

                # impose effect that chi2 stat is achieved
                _y = y + offset[..., np.newaxis]
                # assert (sigma_gain is None) == (rough_exp is None), \
                #     'sigma_gain only when we pass desired roughness'
                # if sigma_gain is not None:
                #     _y = scale_sigma(y=_y, gain=sigma_gain)

                # compute sigma
                y_mean = _y.mean(axis=2)
                yr = _y.reshape((_y.shape[0], -1), order='F')
                sigma = yr @ yr.T / num_vox - y_mean @ y_mean.T

                # compute observed e and h
                q = hglm.experiment.decompose(x, contrast)
                yq1q1y = y_mean @ q[1].T @ q[1] @ y_mean.T
                yq2q2y = y_mean @ q[2].T @ q[2] @ y_mean.T
                h = yq1q1y / num_img
                e = sigma + yq2q2y / num_img

                # ensure proper pval achieved
                wilks = get_wilks(e, h)
                chi2, df = wilks_to_chi2(wilks, a=contrast.sum(), b=b,
                                        n=num_img)
                pval = 1 - scipy.stats.chi2.cdf(chi2, df=df)
                assert np.isclose(pval, pval_exp)

                # if rough_exp is not None:
                #     # ensure desired roughness is achieved
                #     assert np.isclose(rough_exp, rough)
                #
                # # check that roughness is computed properly
                # _rough_exp = get_rough(x, _y)
                # assert np.isclose(_rough_exp, rough)
