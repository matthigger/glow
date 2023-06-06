import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import f

from hrba.f_stat import get_tr_eps


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
    a = (~contrast).sum(), contrast.size

    # prep matrices
    x = x[~contrast, :], x
    y_mean = y.mean(axis=2)
    h = [np.linalg.pinv(_x) @ _x for _x in x]

    if f_stat is None:
        # get target f_stat to impose given p_value
        f_stat = f.ppf(p_val,
                       dfn=num_img * reg_size - a[1],
                       dfd=a[1] - a[0])

    # compute tr_eps_init, vector of length two.  each entry is the trace of
    # residual covariance (reduced=0, full=1)
    tr_eps_init = np.array([get_tr_eps(_x, y) for _x in x])

    # if tr_eps[0] (reduced) and tr_eps[1] (full) have the ratio eps1_over_eps0
    # then the f_stat will be achieved
    const = (reg_size * num_img - b * a[1]) / b * (a[1] - a[0])
    eps1_over_eps0 = const / (const + f_stat)
    assert eps1_over_eps0 <= 1, 'eps0 > eps1'

    # find closest eps0, eps1  which achieves f stat (project tr_eps into
    # direction of u).  this may minimize the norm of the offset, needs
    # further study ...
    u = np.array([1, eps1_over_eps0])
    u /= np.linalg.norm(u)
    tr_eps_target = np.dot(u, tr_eps_init) * u

    # check that tr_eps_target obeys minimum tr_eps (sbj-pooled, spatial cov of
    # y is lower bound on error cov since we're adding constant offset within
    # each image and estimates don't vary across voxels within image)
    tr_eps_min = ((y - y_mean[..., np.newaxis]) ** 2).sum()
    tr_eps_min /= num_img * reg_size
    if (tr_eps_target < tr_eps_min).any():
        # choose closest point which obeys minimum
        # (eps0 >= eps1 since reduced model is contained in full)
        tr_eps_target = np.array([tr_eps_min / eps1_over_eps0, tr_eps_min])

    # get offset to induce trp_eps_after[1]
    offset1 = get_offset_to_tr_eps(x=x[1], y=y, tr_eps=tr_eps_target[1],
                                   offset=y_mean @ (np.eye(num_img) - h[1]))

    # get offset to induce trp_eps_after[0] (error in direction below impacts
    # only the reduced model)
    offset = (y_mean + offset1) @ (np.eye(num_img) - h[0]) @ h[1]
    offset0 = get_offset_to_tr_eps(x=x[0],
                                   y=y + offset1[..., np.newaxis],
                                   tr_eps=tr_eps_target[0],
                                   offset=offset)

    return offset0 + offset1, tr_eps_target


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
