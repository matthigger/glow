import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import f


def compute_offset(x, y, contrast, f_stat=None, p_val=None, seed=None):
    """ get an additive offset to y which, when applied, yields given f stat

    offset = c * delta      where       delta = beta @ x

    where beta is the MMSE mapping from x to y in the input data.  By giving an
    output MMSE beta in the same direction as the input data we can subtract
    out any incidental variance explained by X in Y to produce an F stat of 0.
    (sampling beta randomly often introduces a lower bound on F above 0).

    Attributes:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        f_stat (float): target f statistic
        p_val (float): p-value (to generate target f statistic)
        seed: random number generator (for beta)
        
    Returns:
        beta (np.array): (a, b) MMSE mapping from x to y
        offset (np.array): (b, num_img) offset to apply to all images to 
            produce desired f stat
    """
    assert (f_stat is None) != (p_val is None), 'f_stat xor p_val required'

    if f_stat is None:
        f(3)
        raise NotImplementedError('f stat from p-value here')

    # compute h and delta (from full model)
    b, num_img, reg_size = y.shape
    pinv_x = np.linalg.pinv(x)
    x_reduce = x[~contrast, :]
    h = np.linalg.pinv(x_reduce) @ x_reduce, pinv_x @ x
    y_mean = y.mean(axis=2)

    # get delta
    assert f_stat >= 0, 'non-negative f required'
    delta = y_mean @ (np.eye(num_img) - h[0]) @ h[1]
    # if f_stat == 0:
    #     # there is only one direction which yields an F-stat of 0
    #     delta = y_mean @ h[1]
    # else:
    #     # choose arbitrary additive direction to achieve other F-stats
    #     rng = np.random.default_rng(seed=seed)
    #     delta = rng.standard_normal((b, num_img))

    # prep
    yv_squared = (y ** 2).sum()
    a = (~contrast).sum(), contrast.size

    # compute polynomial of error covariance as a function of c
    poly = list()
    for _h in h:
        i_minus_h = np.eye(num_img) - _h
        _poly = Polynomial([
            yv_squared - reg_size * np.trace(y_mean @ _h @ y_mean.T),
            reg_size * 2 * np.trace(delta @ i_minus_h @ y_mean.T),
            reg_size * np.trace(delta @ i_minus_h @ delta.T)])
        _poly /= num_img * reg_size
        poly.append(_poly)

    # solve for constant c which yields given f stat
    # f = (poly[0] - poly[1]) / poly[1] * const_term
    # left_const = (reg_size * num_img - b * a[1]) / f_stat * b * (a[0] - a[1])
    left_const = f_stat * b * (a[0] - a[1]) / (reg_size * num_img - b * a[1])
    final_poly = left_const * poly[1] - (poly[0] - poly[1])

    # find c which achieves F stat desired
    roots = final_poly.roots()
    assert np.isreal(roots).any(), 'no real roots found'

    # cleanup
    offset = roots[0] * delta
    beta = (y_mean + offset) @ pinv_x

    return beta, offset
