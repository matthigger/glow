"""Export an experiment + effect to a folder for inspection.

Writes subject images (NIfTI or PNG), masks, a scatter plot of the
regression in the effect region, and a JSON metadata summary.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _is_3d(mask_idx):
    return mask_idx.ndim == 3


def _reconstruct_volume(voxel_data, mask_idx):
    """Place flat voxel data back into spatial coordinates."""
    vol = np.zeros(mask_idx.shape, dtype=voxel_data.dtype)
    vol[mask_idx >= 0] = voxel_data
    return vol


def _save_volume(vol, path, affine=None):
    """Save a spatial array as NIfTI (.nii.gz) or PNG."""
    path = Path(path)
    if _is_3d(vol):
        import nibabel as nib
        if affine is None:
            affine = np.eye(4)
        img = nib.Nifti1Image(vol.astype(np.float32), affine)
        nib.save(img, str(path.with_suffix('.nii.gz')))
    else:
        from PIL import Image
        arr = vol.astype(np.float64)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        Image.fromarray(arr.astype(np.uint8)).save(
            str(path.with_suffix('.png')))


def _save_mask(mask, path, affine=None):
    """Save a boolean mask as NIfTI or PNG."""
    _save_volume(mask.astype(np.uint8), path, affine=affine)


def _plot_scatter(exp, effect, path):
    """Scatter plot of treatment variable vs mean intensity in effect region.

    One panel per image feature (b).  Points = subjects, with error bars
    (within-subject std across effect voxels) and OLS regression line.
    """
    b, num_img, _ = exp.y.shape
    effect_idx = exp.mask_idx[effect.mask]
    y_eff = exp.y[:, :, effect_idx]

    features = exp.meta.get('features', [f'feat_{i}' for i in range(b)])
    subjects = exp.meta.get('subjects', [f'{i}' for i in range(num_img)])

    contrast_cols = np.where(exp.contrast)[0]
    if len(contrast_cols) == 0:
        return
    x_col = contrast_cols[0]
    x_vals = exp.x[x_col, :]

    fig, axes = plt.subplots(1, b, figsize=(5 * b, 4.5), squeeze=False)

    for feat_idx in range(b):
        ax = axes[0, feat_idx]
        y_mean = y_eff[feat_idx].mean(axis=1)
        y_std = y_eff[feat_idx].std(axis=1)

        ax.errorbar(x_vals, y_mean, yerr=y_std, fmt='o', ms=4, alpha=0.7,
                     capsize=2, label='subjects')

        # OLS fit
        coeffs = np.polyfit(x_vals, y_mean, 1)
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', lw=2,
                label=f'slope={coeffs[0]:.3g}')

        ax.set_xlabel(f'x[{x_col}]')
        ax.set_ylabel('mean intensity (effect region)')
        ax.set_title(features[feat_idx])
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def to_folder(exp, folder, effect=None):
    """Export experiment data to *folder* for inspection.

    Args:
        exp: Experiment (must have x, contrast, y, mask_idx, meta)
        folder: output directory (created if needed)
        effect: Effect with .mask and .effect_llr (optional)
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    b, num_img, num_vox = exp.y.shape
    affine = exp.meta.get('affine')
    features = exp.meta.get('features', [f'feat_{i}' for i in range(b)])
    subjects = exp.meta.get('subjects',
                            [f'subject_{i:03d}' for i in range(num_img)])

    # --- masks ---
    analysis_mask = exp.mask_idx >= 0
    _save_mask(analysis_mask, folder / 'mask_analysis', affine=affine)

    if effect is not None:
        _save_mask(effect.mask, folder / 'mask_effect', affine=affine)

    # --- subject images ---
    img_dir = folder / 'images'
    img_dir.mkdir(exist_ok=True)
    for sbj_idx, sbj_name in enumerate(subjects):
        for feat_idx, feat_name in enumerate(features):
            vox = exp.y[feat_idx, sbj_idx, :]
            vol = _reconstruct_volume(vox, exp.mask_idx)
            _save_volume(vol, img_dir / f'{sbj_name}_{feat_name}',
                         affine=affine)

    # --- scatter plot ---
    if effect is not None and hasattr(exp, 'x'):
        _plot_scatter(exp, effect, folder / 'scatter.png')

    # --- metadata ---
    info = {
        'shape': list(exp.mask_idx.shape),
        'b': b,
        'num_img': num_img,
        'num_vox': num_vox,
        'features': features,
        'subjects': subjects,
        'has_affine': affine is not None,
    }
    if effect is not None:
        info['effect_llr'] = float(getattr(effect, 'effect_llr', 0))
        info['seed'] = int(getattr(effect, 'seed', 0))
        info['effect_voxels'] = int(effect.mask.sum())

    if hasattr(exp, 'x'):
        info['x_shape'] = list(exp.x.shape)
        info['contrast'] = exp.contrast.tolist()

    with open(folder / 'info.json', 'w') as f:
        json.dump(info, f, indent=2, sort_keys=True)

    print(f'exported to {folder}')
    print(f'  {num_img} subjects x {b} features')
    print(f'  {num_vox} voxels, shape {tuple(exp.mask_idx.shape)}')
    if effect is not None:
        print(f'  effect: {info["effect_voxels"]} voxels, '
              f'llr={info["effect_llr"]:.4g}')
