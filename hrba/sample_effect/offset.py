import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize
from scipy.stats import f

from hrba.f_stat import get_f_degrees, get_f_const


def compute_offset(x, y, contrast, f_stat=None, p_val=None):
    """ get offset to y, constant across voxels, which imposes an f-stat

    notation:
    - epsilon = the covariance of the residuals
    - h = np.linalg.pinv(x) @ x, the estimate forming matrix
    - we use the index 0 and 1 to refer to the reduced and full models
    respectively

    observe:
    - an f-statistic is equivalent to a particular ratio of tr_eps1 / tr_eps0
    - adding offset to y in direction y_mean @ (I - h[1])  modifies tr_eps1 and
     tr_eps0, it is noise to both models
    - adding offset to y in direction y_mean @ (I - h[0]) @ h[1] modifies
    tr_eps0 but not tr_eps1, it is noise to the reduced model and entirely
    explained by the full model

    approach:
    - identify the values of  tr_eps0, tr_eps1 which are closest to the
    values implicit in the inputs which impose the desired f-statistic
    - scale the noise in the two directions listed below to achieve
    necessary tr_eps0, tr_eps1
         - y_mean @ (I - h[1]): all noise to both full & reduced
         - y_mean @ (I - h[0]) @ h[1]): noise to reduced, completely
            explained by full model

    Args:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        f_stat (float): target f statistic
        p_val (float): p-value may be passed in place of f stat
        
    Returns:
        offset (np.array): (b, num_img) offset to apply to all images to 
            produce desired f stat
    """
    assert (f_stat is None) != (p_val is None), 'f_stat xor p_val required'

    # prep constants
    b, num_img, reg_size = y.shape

    # prep matrices
    x = x[~contrast, :], x
    y_mean = y.mean(axis=2)
    h = [np.linalg.pinv(_x) @ _x for _x in x]

    if f_stat is None:
        # get target f_stat to impose given p_value
        dfn, dfd = get_f_degrees(y, contrast)
        f_stat = f.ppf(1 - p_val, dfn=dfn, dfd=dfd)

    # if tr_eps[0] (reduced) and tr_eps[1] (full) have the ratio eps1_over_eps0
    # then the f_stat will be achieved
    const = get_f_const(y, contrast)
    eps1_over_eps0 = const / (const + f_stat)
    assert eps1_over_eps0 <= 1, 'eps0 > eps1'

    # offset directions are the average (per voxel) noise in each model.
    # This direction has the advantage of allowing us to reduce the noise in
    # a model to zero (not true of an arbitrary direction)
    i = np.eye(num_img)
    offset = y_mean @ (i - h[0]) @ h[1], y_mean @ (i - h[1])
    offset = [off / np.linalg.norm(off.flatten()) for off in offset]

    def constraint(c, offset=offset):
        """ when this function output is zero, F-stat is achieved """
        offset = c[0] * offset[0] + c[1] * offset[1]

        # tr_eps_poly[i] is a polynomial representing the tr_eps of the model as
        # a function of the scale of the offset given above
        tr_eps_poly = \
            get_tr_eps_poly(x=x[0], y=y, offset=offset), \
                get_tr_eps_poly(x=x[1], y=y, offset=offset)

        return tr_eps_poly[1](1) - tr_eps_poly[0](1) * eps1_over_eps0

    def obj(c):
        """ since offsets are orthonormal, smallest c -> smallest offset"""
        return (c ** 2).sum()

    # find scale of each offset which achieves F while being as close as
    # possible to original problem
    res = minimize(fun=obj, x0=np.array([1, 1]),
                   constraints=[{'type': 'eq',
                                 'fun': constraint}],
                   options={'maxiter': 1000})
    assert res.success, 'optimization failed'

    offset = res.x[0] * offset[0] + res.x[1] * offset[1]

    return offset


def get_tr_eps_poly(x, y, offset):
    """ builds cubic polynomial of tr_eps as a function of offset scale

    Attributes:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        offset (np.array): (b, num_img) offset direction, if not passed then
            tr_eps is achieved with minimum squared difference to original y
    """
    # prep
    b, num_img, reg_size = y.shape
    h = np.linalg.pinv(x) @ x
    i_minus_h = np.eye(num_img) - h
    y_mean = y.mean(axis=2)
    yv_squared = (y ** 2).sum()

    # build polynomial of trace of error covariance as a function of
    # multiplier (minus target tr_eps so that roots are solutions)
    poly = Polynomial([
        yv_squared - reg_size * np.trace(y_mean @ h @ y_mean.T),
        reg_size * 2 * np.trace(offset @ i_minus_h @ y_mean.T),
        reg_size * np.trace(offset @ i_minus_h @ offset.T)])
    return poly / (num_img * reg_size)


def get_offset_to_tr_eps(x, y, tr_eps, offset=None):
    """ gets offset (scales current error) to achieve a particular trace(eps)

    Attributes:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        tr_eps (float): target error covariance trace
        offset (np.array): (b, num_img) offset direction, if not passed then
            tr_eps is achieved with minimum squared difference to original y

    Returns:
        offset (np.array): when added to every voxel in y, the resulting
            regression model has error tr_eps
    """
    # prep
    b, num_img, reg_size = y.shape
    h = np.linalg.pinv(x) @ x
    i_minus_h = np.eye(num_img) - h
    y_mean = y.mean(axis=2)
    yv_squared = (y ** 2).sum()

    if offset is None:
        # offset is in the direction of the current residuals in y
        # (if we add error in another direction then we'll never cancel all
        # error and introduce a lower bound on tr_eps which are viable,
        # if this default is used polynomial guaranteed to have real roots
        # below)
        offset = y_mean @ (np.eye(num_img) - h)

    # build polynomial of trace of error covariance as a function of
    # multiplier (minus target tr_eps so that roots are solutions)
    n = num_img * reg_size
    poly = Polynomial([
        yv_squared - reg_size * np.trace(y_mean @ h @ y_mean.T) - tr_eps * n,
        reg_size * 2 * np.trace(offset @ i_minus_h @ y_mean.T),
        reg_size * np.trace(offset @ i_minus_h @ offset.T)])

    # solve (and choose max for consistency)
    c = np.max(poly.roots())
    if not np.isreal(c):
        if np.isclose(c.imag, 0, atol=1e-7):
            # floating point error yields imaginary value, discard
            # (test case yields error in root of ~1e-8)
            c = c.real
        else:
            raise RuntimeError('imaginary root found, optimization failed')

    return offset * c
