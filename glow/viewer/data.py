"""Data preparation for the glow viewer.

Builds the per-region DataFrame and background images from an AnalysisGLOW.
"""

import numpy as np
import pandas as pd

import glow.graph
from glow.analysis.mancova import decompose, get_llr, get_roughness


def _get_adjusted_stat(ana_glow):
    """Return the adjusted stat array (llr_adjusted), always 1-D."""
    adj = getattr(ana_glow, 'llr_adjusted_0', ana_glow.llr_adjusted_0)
    return adj if adj.ndim == 1 else adj[0]


def _ensure_1d(arr):
    """Return a 1-D view: if 2-D (b, num_reg), take first row."""
    return arr if arr.ndim == 1 else arr[0]


def prep_df(ana_glow, mask_target=None, extra_df=None):
    """Build a DataFrame with one row per region (unpermuted only).

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        mask_target (np.array): optional boolean target mask (same shape
            as ana_glow.exp.mask_idx)
        extra_df (pd.DataFrame): optional DataFrame keyed on ``region_idx``
            to left-join onto the result.  Extra columns appear in the
            scatter dropdowns automatically.

    Returns:
        df (pd.DataFrame): one row per region with all available stats
    """
    children = ana_glow.children
    num_vox = ana_glow.exp.y.shape[2]
    num_reg = num_vox + children.shape[0]

    # compute per-region roughness via tree walk
    roughness_arr = np.full(num_reg, np.nan)
    q_tup = decompose(x=ana_glow.exp.x, contrast=ana_glow.exp.contrast)
    for reg_idx, size, ysum, yout in glow.graph.iter_size_ysum_yout(
            ana_glow.exp.y, children=children):
        ymean = ysum / size
        sigma = yout - ymean @ ymean.T * size
        a0 = ysum @ q_tup[0].T
        t = yout - a0 @ a0.T / size
        a1 = ysum @ q_tup[1].T
        h = a1 @ a1.T / size
        e = t - h
        roughness_arr[reg_idx] = get_roughness(e, sigma)

    d = {
        'region_idx': np.arange(num_reg),
        'n_voxel': ana_glow.size.astype(int),
        'llr': _ensure_1d(ana_glow.stat),
        'llr_adjusted': _get_adjusted_stat(ana_glow),
        'pval_fwer': ana_glow.pval,
        'roughness': roughness_arr,
    }

    # H0 null distribution parameters (stored by _finalize_analysis)
    if hasattr(ana_glow, 'stat_mu'):
        d['llr_mu_h0'] = ana_glow.stat_mu
    if hasattr(ana_glow, 'stat_std'):
        d['llr_std_h0'] = ana_glow.stat_std

    # significant flag (pval <= alpha_fwer)
    alpha_fwer = getattr(ana_glow, 'alpha_fwer', 0.05)
    d['significant'] = ~np.isnan(ana_glow.pval) & (ana_glow.pval <= alpha_fwer)

    # discovered flag (significant AND survived pruning)
    discovered = np.zeros(num_reg, dtype=bool)
    for effect in ana_glow.effect_list:
        discovered[effect.reg_idx] = True
    d['discovered'] = discovered

    # estimate_state: 'has_effect' if significant, else 'no_effect'
    estimate_state = np.where(d['discovered'], 'has_effect', 'no_effect')
    d['estimate_state'] = estimate_state

    # mask-target derived stats
    if mask_target is not None:
        dice, sens, spec = glow.graph.get_dice_sens_spec(
            children=children,
            mask_idx=ana_glow.exp.mask_idx,
            mask=mask_target)
        miss, hits = glow.graph.get_miss_hits(
            children=children,
            mask_idx=ana_glow.exp.mask_idx,
            mask=mask_target)
        d['dice'] = dice
        d['sens'] = sens
        d['spec'] = spec
        d['vox_in_target'] = hits.astype(int)
        d['vox_out_target'] = miss.astype(int)

        sig_mask = ~np.isnan(ana_glow.pval) & (ana_glow.pval <= alpha_fwer)
        max_dice_sig = float(dice[sig_mask].max()) if sig_mask.any() else 0.0
        if max_dice_sig > 0:
            d['pct_max_dice'] = dice / max_dice_sig
        else:
            d['pct_max_dice'] = np.full(num_reg, np.nan)

    df = pd.DataFrame(d)

    if extra_df is not None:
        extra_df = extra_df.copy()
        extra_df['region_idx'] = extra_df['region_idx'].astype(int)
        df = df.merge(extra_df, on='region_idx', how='left')

    return df


_GENERIC_FEATURES = {'n_voxel', 'roughness'}
_PRUNING_FEATURES = set()
_MASK_FEATURES = {'dice', 'sens', 'spec', 'pct_max_dice',
                   'vox_in_target', 'vox_out_target'}


def _compute_r2(stat, size, model, beta):
    """Compute R² for a size-adjustment model on the unpermuted data."""
    from glow.analysis import AnalysisGLOW
    valid = np.isfinite(stat) & (size > 0) & np.isfinite(size)
    if model in AnalysisGLOW._POSITIVE_STAT_MODELS:
        valid &= (stat > 0)
    s, y = size[valid].astype(float), stat[valid].astype(float)
    if len(y) < 3:
        return np.nan
    y_hat = AnalysisGLOW.predict_null_mean(s, model, beta)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def get_feature_columns(df):
    """Return feature columns grouped by category.

    Excludes boolean and index columns, and columns that are all NaN.

    Returns:
        generic (list[str]): generic features (e.g. n_voxel)
        significance (list[str]): significance-testing features
        pruning (list[str]): pruning-related features
        mask (list[str]): mask-target features (may be empty)
    """
    exclude = {'region_idx', 'discovered', 'significant', 'estimate_state'}
    generic, significance, pruning, mask = [], [], [], []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == bool:
            continue
        if df[c].isna().all():
            continue
        if c in _GENERIC_FEATURES:
            generic.append(c)
        elif c in _PRUNING_FEATURES or c.startswith('tree_wt_gain_sum'):
            pruning.append(c)
        elif c in _MASK_FEATURES:
            mask.append(c)
        else:
            significance.append(c)
    return sorted(generic), sorted(significance), sorted(pruning), sorted(mask)


def compute_target_stats(ana_glow, mask_target):
    """Compute stats for the full target mask treated as a single region.

    Computes LLR (and size-adjusted variant) for the union of all
    analysis voxels inside ``mask_target``, plus trivial mask-vs-self
    metrics.

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        mask_target (np.array): boolean target mask (same shape as mask_idx)

    Returns:
        dict or None: stat-name -> value.  Keys match the DataFrame columns
            produced by ``prep_df`` where computable; others are NaN.
            Returns None if the target has no analysis voxels.
    """
    exp = ana_glow.exp
    mask_idx = exp.mask_idx
    y = exp.y  # (b, num_img, num_vox)

    vox_indices = mask_idx[mask_target & (mask_idx >= 0)]
    n_voxel = len(vox_indices)
    if n_voxel == 0:
        return None

    y_sub = y[:, :, vox_indices]  # (b, num_img, n_vox)
    ysum = y_sub.sum(axis=2)  # (b, num_img)
    yout = np.einsum('bin,cin->bc', y_sub, y_sub)  # (b, b)

    q = decompose(x=exp.x, contrast=exp.contrast)
    a0 = ysum @ q[0].T
    t = yout - a0 @ a0.T / n_voxel
    a1 = ysum @ q[1].T
    h = a1 @ a1.T / n_voxel
    e = t - h

    llr = get_llr(e, h, n=n_voxel)

    stats = {
        'n_voxel': n_voxel,
        'llr': llr,
    }

    adj_model = getattr(ana_glow, 'adj_model', None)
    adj_beta = getattr(ana_glow, 'adj_beta', None)
    if adj_model is not None and adj_beta is not None and np.isfinite(llr):
        from glow.analysis import AnalysisGLOW
        predicted = AnalysisGLOW.predict_null_mean(n_voxel, adj_model, adj_beta)
        stats['llr_adjusted'] = llr - predicted
    else:
        stats['llr_adjusted'] = np.nan

    stats['dice'] = 1.0
    stats['sens'] = 1.0
    stats['spec'] = 1.0
    stats['vox_in_target'] = n_voxel
    stats['vox_out_target'] = 0

    stats['pval_fwer'] = np.nan
    stats['llr_mu_h0'] = np.nan
    stats['llr_std_h0'] = np.nan

    return stats


def get_original_y(exp):
    """Return the original (pre-scaling) imaging data.

    If *exp* is an ``ExperimentScaled``, inverts the zero-mean + whitening
    transform so the returned array has the same units as the user's input
    images.  Otherwise returns ``exp.y`` unchanged.

    Returns:
        y (np.array): ``(b, num_img, num_vox)``
    """
    if hasattr(exp, 'prep_inv'):
        return exp.prep_inv(exp.y)
    return exp.y


def compute_backgrounds(ana_glow, y_features=None, image_idx=None):
    """Compute per-feature background images from the experiment data.

    Uses original (pre-scaled) intensities so that backgrounds match
    the user's input images.

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        y_features (list[str] | None): human-readable names for each
            imaging feature.  Falls back to ``exp.meta['features']``,
            then ``"feature 0"``, ``"feature 1"``, ...
        image_idx (int | None): if provided, use a single image (0-indexed)
            instead of the mean across all images.

    Returns:
        bg_dict (dict): feature_name -> np.array with same shape as mask_idx.
            Voxels outside the analysis mask are NaN.
    """
    exp = ana_glow.exp
    mask_idx = exp.mask_idx
    y = get_original_y(exp)  # (b, num_img, num_vox)

    if image_idx is not None:
        y_mean = y[:, image_idx, :]  # (b, num_vox) single image
    else:
        y_mean = y.mean(axis=1)      # (b, num_vox) grand mean
    b = y_mean.shape[0]

    if y_features is None:
        y_features = getattr(exp, 'meta', {}).get('features')
    if y_features is None:
        y_features = [f'feature {i}' for i in range(b)]
    assert len(y_features) == b, \
        f'y_features length {len(y_features)} != b={b}'

    bg_dict = {}
    for feat_idx in range(b):
        name = y_features[feat_idx]
        img = np.full(mask_idx.shape, np.nan, dtype=float)
        img[mask_idx >= 0] = y_mean[feat_idx, mask_idx[mask_idx >= 0]]
        bg_dict[name] = img

    # RGB composite when the three canonical channels are present (2D only)
    _RGB_CHANNELS = ('red', 'green', 'blue')
    lower_names = [n.lower() for n in y_features]
    if (mask_idx.ndim == 2
            and set(lower_names) == set(_RGB_CHANNELS)):
        rgb = np.full((*mask_idx.shape, 3), np.nan, dtype=float)
        for ci, ch_name in enumerate(_RGB_CHANNELS):
            src_idx = lower_names.index(ch_name)
            rgb[:, :, ci] = bg_dict[y_features[src_idx]]
        bg_dict['RGB'] = rgb

    return bg_dict


def compute_bg_ranges(ana_glow, y_features=None):
    """Compute the global (vmin, vmax) for each background key across ALL images.

    This ensures the colour scale stays constant regardless of which image
    (mean or individual) is displayed.

    Returns:
        ranges (dict): bg_name -> (vmin, vmax) floats
    """
    exp = ana_glow.exp
    mask_idx = exp.mask_idx
    y = get_original_y(exp)  # (b, num_img, num_vox)
    b = y.shape[0]

    if y_features is None:
        y_features = getattr(exp, 'meta', {}).get('features')
    if y_features is None:
        y_features = [f'feature {i}' for i in range(b)]

    ranges = {}
    for feat_idx in range(b):
        vals = y[feat_idx][:, mask_idx[mask_idx >= 0]]  # (num_img, valid_vox)
        ranges[y_features[feat_idx]] = (
            float(np.nanmin(vals)), float(np.nanmax(vals)))

    _RGB_CHANNELS = ('red', 'green', 'blue')
    lower_names = [n.lower() for n in y_features]
    if mask_idx.ndim == 2 and set(lower_names) == set(_RGB_CHANNELS):
        all_vals = y[:, :, mask_idx[mask_idx >= 0].ravel()]
        ranges['RGB'] = (float(np.nanmin(all_vals)), float(np.nanmax(all_vals)))

    return ranges
