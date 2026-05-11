import warnings

from glow.experiment import ExperimentImageOnly
from glow.experiment.exper import NoBiasTermWarning
from glow.analysis.mancova import *
from glow.graph import iter_topo


def test_get_mancova():
    seed = 0
    a = 2
    b = 3
    num_vox = 5
    exp = ExperimentImageOnly.from_gauss(seed=seed, b=b, num_img=4, shape=(num_vox,))
    children = np.arange(2 * num_vox - 2).reshape((-1, 2), order='C')

    for add_bias in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NoBiasTermWarning)
            exp = exp.sample_x(a=a, seed=seed, add_bias=add_bias)

            for reg_idx in iter_topo(children=children, num_leaf=num_vox):
                vox = np.array(list(iter_topo(children=children,
                                              num_leaf=num_vox,
                                              node_start=reg_idx,
                                              only_leaf=True)))
                _y = exp.y[:, :, vox]

                e_obs, h_obs, sigma = get_mancova(x=exp.x, y=_y, contrast=exp.contrast)

                # slow and steady compute of e and h
                yr = np.concatenate([exp.y[:, :, _vox] for _vox in vox], axis=1)
                xr = np.concatenate([exp.x for _vox in vox], axis=1)

                if not exp.contrast.all():
                    # project yr into nullspace of covariates
                    _xr = xr[~exp.contrast, :]
                    p = np.eye(yr.shape[1]) - np.linalg.pinv(_xr) @ _xr
                    yr = yr @ p
                    xr = xr[exp.contrast, :] @ p

                hat = yr @ np.linalg.pinv(xr) @ xr
                err = yr - hat

                h_exp = hat @ hat.T
                e_exp = err @ err.T

                # test mancova stats.  Reference path uses float64
                # primitives (np.eye, np.linalg.pinv) while the new
                # default experiment dtype is float32, so the comparison
                # straddles dtypes — loosen tolerances to float32
                # working precision.
                assert np.allclose(h_obs, h_exp, rtol=1e-4, atol=1e-5)
                assert np.allclose(e_obs, e_exp, rtol=1e-4, atol=1e-5)


def test_get_hotel_tr():
    def get_hotel_tr_trusted(e, h):
        inv_e = np.linalg.inv(e)
        mat = inv_e @ h
        return np.trace(mat)

    rng = np.random.default_rng(0)
    b = 3
    for _ in range(10):
        e = rng.standard_normal((b, b))
        h = rng.standard_normal((b, b))

        exp = get_hotel_tr_trusted(e, h)
        obs = get_hotel_tr(e, h)
        assert np.isclose(exp, obs)


def test_get_mancova_with_q_tup():
    """test get_mancova with pre-computed q_tup"""
    seed = 0
    a = 2
    b = 3
    num_vox = 5
    exp = ExperimentImageOnly.from_gauss(seed=seed, b=b, num_img=4, shape=(num_vox,))
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NoBiasTermWarning)
        exp = exp.sample_x(a=a, seed=seed)
    
    y = exp.y[:, :, :3]  # subset of voxels
    
    # compute q_tup first
    q_tup = decompose(exp.x, exp.contrast)
    
    # call with q_tup
    e1, h1, sigma1 = get_mancova(y=y, q_tup=q_tup)
    
    # call with x
    e2, h2, sigma2 = get_mancova(x=exp.x, y=y, contrast=exp.contrast)
    
    # should be identical
    assert np.allclose(e1, e2)
    assert np.allclose(h1, h2)
    assert np.allclose(sigma1, sigma2)


def test_all_stat_functions():
    """test all MANCOVA statistics"""
    import pytest
    rng = np.random.default_rng(42)
    b = 3
    
    # create symmetric positive definite matrices
    e_raw = rng.standard_normal((b, b))
    e = e_raw @ e_raw.T + np.eye(b)  # ensure positive definite
    
    h_raw = rng.standard_normal((b, b))
    h = h_raw @ h_raw.T
    
    # test all stats run without error
    wilks = get_wilks(e=e, h=h)
    assert 0 < wilks <= 1

    pillai = get_pillai(e=e, h=h)
    assert pillai >= 0

    hotel = get_hotel_tr(e=e, h=h)
    assert hotel >= 0

    roy = get_roys_root(e=e, h=h)
    assert roy >= 0

    llr = get_llr(e=e, h=h, n=100)
    assert llr >= 0

    # test stat_dict
    assert len(stat_dict) == 5
    assert 'llr' in stat_dict
    assert 'wilks' in stat_dict
    assert 'pillai' in stat_dict
    assert 'hotel_tr' in stat_dict
    assert 'roys_root' in stat_dict

    for name, stat_func in stat_dict.items():
        result = stat_func(e=e, h=h, n=100)
        assert isinstance(result, (float, np.floating))

    # all stats oriented so larger = more evidence against H0
    for name, stat_func in stat_dict.items():
        result = stat_func(e=e, h=h, n=100)
        assert result >= 0, f'{name} should be non-negative'


def test_singular_matrix_errors():
    """stat functions raise LinAlgError with diagnostic message on singular input"""
    import pytest

    # singular E (rank 1) — affects hotel_tr and roys_root
    e_singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    h = np.eye(2)

    with pytest.raises(np.linalg.LinAlgError, match='num_img'):
        get_hotel_tr(e_singular, h)

    with pytest.raises(np.linalg.LinAlgError, match='num_img'):
        get_roys_root(e_singular, h)

    # singular H + E — affects pillai
    e_zero = np.zeros((2, 2))
    h_singular = np.array([[1.0, 0.0], [0.0, 0.0]])

    with pytest.raises(np.linalg.LinAlgError, match='num_img'):
        get_pillai(e_zero, h_singular)


def test_get_llr_fidelity():
    """New loglik_from_cov-based get_llr matches the original solve+slogdet."""
    def _get_llr_reference(e, h, n):
        """Original implementation (solve + slogdet on I + E^{-1}H)."""
        try:
            inv_e_h = np.linalg.solve(e, h)
        except np.linalg.LinAlgError:
            return np.nan
        s, logdet = np.linalg.slogdet(np.eye(e.shape[0]) + inv_e_h)
        if s <= 0:
            return np.nan
        return 0.5 * n * logdet

    rng = np.random.default_rng(42)
    for b in [1, 2, 4]:
        for _ in range(20):
            A = rng.standard_normal((b, b))
            e = A @ A.T + np.eye(b) * 0.1
            B = rng.standard_normal((b, b))
            h = B @ B.T
            n = rng.integers(1, 1000)

            ref = _get_llr_reference(e, h, n)
            new = get_llr(e, h, n)
            assert np.isclose(ref, new, rtol=1e-10), \
                f'b={b}, n={n}: ref={ref}, new={new}'

    # size_normalize=True should equal get_llr(e, h, n=1)
    A = rng.standard_normal((2, 2))
    e = A @ A.T + np.eye(2) * 0.1
    B = rng.standard_normal((2, 2))
    h = B @ B.T
    sn = get_llr(e, h, size_normalize=True)
    ref_sn = _get_llr_reference(e, h, n=1)
    assert np.isclose(sn, ref_sn, rtol=1e-10)
