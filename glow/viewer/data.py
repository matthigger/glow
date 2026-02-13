"""Data preparation for the glow viewer.

Builds the per-region DataFrame and background images from an AnalysisGLOW.
"""

import numpy as np
import pandas as pd

import glow.graph


def _get_adjusted_stat(ana_glow):
    """Return the adjusted stat array (hotel_tr_adjusted)."""
    return ana_glow.hotel_tr_adjusted


def prep_df(ana_glow, mask_target=None):
    """Build a DataFrame with one row per region (unpermuted only).

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        mask_target (np.array): optional boolean target mask (same shape
            as ana_glow.exp.mask_idx)

    Returns:
        df (pd.DataFrame): one row per region with all available stats
    """
    num_vox = ana_glow.exp.y.shape[2]
    num_reg = num_vox * 2 - 1
    children = ana_glow.child_dict[0]

    d = {
        'region_idx': np.arange(num_reg),
        'n_voxel': ana_glow.size[0, :].astype(int),
        'hotel_tr': ana_glow.stat[0, :],
        'hotel_tr_adjusted': _get_adjusted_stat(ana_glow)[0, :],
        'pval_fwer': ana_glow.pval,
    }

    # H0 null distribution parameters (stored by _finalize_analysis)
    if hasattr(ana_glow, 'stat_mu'):
        d['hotel_tr_mu_h0'] = ana_glow.stat_mu
    if hasattr(ana_glow, 'stat_std'):
        d['hotel_tr_std_h0'] = ana_glow.stat_std

    # homogeneity pruning p-values (only for tested regions)
    homo_pval = np.full(num_reg, np.nan)
    if hasattr(ana_glow, 'homo_pval_dict'):
        for reg_idx, pval in ana_glow.homo_pval_dict.items():
            homo_pval[reg_idx] = pval
    d['pval_homo'] = homo_pval

    # significant flag (pval <= alpha_fwer)
    alpha_fwer = getattr(ana_glow, 'alpha_fwer', 0.05)
    d['significant'] = ~np.isnan(ana_glow.pval) & (ana_glow.pval <= alpha_fwer)

    # discovered flag (significant AND survived pruning)
    discovered = np.zeros(num_reg, dtype=bool)
    for effect in ana_glow.effect_list:
        discovered[effect.reg_idx] = True
    d['discovered'] = discovered

    # estimate_state: classify each region into one of four states
    #   no_effect      — not significant
    #   partial        — significant, descendant of a discovered effect (subsumed)
    #   full_effect    — significant and discovered
    #   multi_effect   — significant, not discovered, not descendant of any effect
    parent = glow.graph.get_parent(children, num_vox)
    sig = d['significant']
    estimate_state = np.full(num_reg, 'no_effect', dtype=object)
    estimate_state[discovered] = 'full_effect'

    disc_set = set(np.where(discovered)[0])
    for reg in np.where(sig & ~discovered)[0]:
        # walk ancestors to see if any is a discovered effect
        node = reg
        is_partial = False
        while True:
            node = parent[node]
            if node == -1:
                break
            if node in disc_set:
                is_partial = True
                break
        estimate_state[reg] = 'partial' if is_partial else 'multi_effect'

    d['estimate_state'] = estimate_state

    # mask-target derived stats
    if mask_target is not None:
        f1, sens, spec = glow.graph.get_f1_sens_spec(
            children=children,
            mask_idx=ana_glow.exp.mask_idx,
            mask=mask_target)
        miss, hits = glow.graph.get_miss_hits(
            children=children,
            mask_idx=ana_glow.exp.mask_idx,
            mask=mask_target)
        d['f1'] = f1
        d['sens'] = sens
        d['spec'] = spec
        d['vox_in_target'] = hits.astype(int)
        d['vox_out_target'] = miss.astype(int)

    df = pd.DataFrame(d)
    return df


_MASK_FEATURES = {'f1', 'sens', 'spec', 'vox_in_target', 'vox_out_target'}


def get_feature_columns(df):
    """Return two sorted lists of numeric columns for scatter axes/color.

    Excludes boolean and index columns, and columns that are all NaN.

    Returns:
        core (list[str]): alphabetised core analysis features
        mask (list[str]): alphabetised mask-target features (may be empty)
    """
    exclude = {'region_idx', 'discovered', 'significant', 'estimate_state'}
    core, mask = [], []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == bool:
            continue
        if df[c].isna().all():
            continue
        if c in _MASK_FEATURES:
            mask.append(c)
        else:
            core.append(c)
    return sorted(core), sorted(mask)


def compute_backgrounds(ana_glow, feature_names=None):
    """Compute per-feature background images from the experiment data.

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        feature_names (list[str] | None): optional human-readable names for
            each imaging feature.  Length must equal ``b`` (number of
            features).  Falls back to ``"feature 0"``, ``"feature 1"``, ...

    Returns:
        bg_dict (dict): feature_name -> np.array with same shape as mask_idx.
            Voxels outside the analysis mask are NaN.
    """
    exp = ana_glow.exp
    mask_idx = exp.mask_idx
    y = exp.y  # (b, num_img, num_vox)

    # grand mean across images: (b, num_vox)
    y_mean = y.mean(axis=1)
    b = y_mean.shape[0]

    if feature_names is None:
        feature_names = [f'feature {i}' for i in range(b)]
    assert len(feature_names) == b, \
        f'feature_names length {len(feature_names)} != b={b}'

    bg_dict = {}
    for feat_idx in range(b):
        name = feature_names[feat_idx]
        img = np.full(mask_idx.shape, np.nan, dtype=float)
        img[mask_idx >= 0] = y_mean[feat_idx, mask_idx[mask_idx >= 0]]
        bg_dict[name] = img

    return bg_dict
