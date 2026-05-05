"""CLI entry point for the glow viewer.

Usage::

    # interactive demo
    python -m glow.viewer --demo

    # load a pickled AnalysisGLOW
    python -m glow.viewer analysis.p.gz

    # with a target mask (nifti, numpy, or pickled Effect)
    python -m glow.viewer analysis.p.gz --mask target.nii.gz
    python -m glow.viewer analysis.p.gz --mask target.npy
    python -m glow.viewer analysis.p.gz --mask effect.pkl
"""

import argparse
import gzip
import pathlib
import pickle
import sys
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Data paths (relative to this package: src/glow/viewer/__main__.py)
# ---------------------------------------------------------------------------

_DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / 'test' / 'data'


def _data_path(filename):
    p = _DATA_DIR / filename
    if not p.exists():
        p = pathlib.Path('test/data') / filename
    return p


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_analysis(path):
    """Load a pickled AnalysisGLOW from a file."""
    path = pathlib.Path(path)
    suffixes = ''.join(path.suffixes)

    if suffixes.endswith('.p.gz') or suffixes.endswith('.pkl.gz'):
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
    else:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj


def _load_mask(path, mask_idx):
    """Load a target mask from a file."""
    path = pathlib.Path(path)
    suffixes = ''.join(path.suffixes)

    if suffixes.endswith('.nii') or suffixes.endswith('.nii.gz'):
        import nibabel as nib
        img = nib.load(str(path))
        mask = np.asarray(img.dataobj).astype(bool)
        warnings.warn(
            'Loaded nifti mask -- assuming it is in the same voxel space as '
            'the analysis images.  The affine is NOT checked.',
            stacklevel=2)
    elif suffixes.endswith('.npy'):
        mask = np.load(str(path)).astype(bool)
    else:
        obj = _load_analysis(path)
        if hasattr(obj, 'mask'):
            mask = np.asarray(obj.mask).astype(bool)
        elif isinstance(obj, np.ndarray):
            mask = obj.astype(bool)
        else:
            raise ValueError(
                f'Could not interpret {path} as a mask.  Expected a nifti '
                f'(.nii/.nii.gz), numpy array (.npy), or a pickled object '
                f'with a .mask attribute (e.g. Effect).')

    if mask.shape != mask_idx.shape:
        raise ValueError(
            f'Mask shape {mask.shape} does not match analysis mask_idx shape '
            f'{mask_idx.shape}.')
    return mask


# ---------------------------------------------------------------------------
# Interactive prompt helpers
# ---------------------------------------------------------------------------

def _choose(prompt, options, default=None):
    """Display a numbered menu and return the user's choice.

    *options* is a list of ``(key, label)`` tuples.  *default* (if given)
    is the *key* to select when the user presses Enter without typing.
    Echoes the chosen option so the user sees what was selected.
    """
    print(f'\n  {prompt}')
    label_by_key = {}
    for i, (key, label) in enumerate(options, 1):
        tag = ' [default]' if key == default else ''
        print(f'    {i}) {label}{tag}')
        label_by_key[key] = label
    hint = ' (enter for default)' if default is not None else ''
    while True:
        raw = input(f'  >{hint} ').strip()
        if raw == '' and default is not None:
            print(f'  -> {label_by_key[default]}')
            return default
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                chosen = options[idx - 1]
                print(f'  -> {chosen[1]}')
                return chosen[0]
        except ValueError:
            pass
        print(f'  please enter 1-{len(options)}')


def _choose_int(prompt, default, lo=1, hi=20):
    """Prompt for an integer with a default."""
    print(f'\n  {prompt} [{default}]')
    while True:
        raw = input('  > (enter for default) ').strip()
        if raw == '':
            print(f'  -> {default}')
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                print(f'  -> {v}')
                return v
        except ValueError:
            pass
        print(f'  please enter an integer between {lo} and {hi}')


# ---------------------------------------------------------------------------
# Shared builder helpers
# ---------------------------------------------------------------------------

_NUM_IMG = 12

# effect_llr is size-normalised: the observed region LLR is roughly
# ``effect_llr * |region|``.  So a "medium" 0.5 effect on a 614-voxel
# planted region produces an observed region LLR of ~307, not 0.5.
_EFFECT_MAP = {
    'none':   0.0,
    'mild':   0.25,
    'medium': 0.5,
    'strong': 1.0,
}


def _build_experiment(y, mask_idx, num_img=_NUM_IMG, meta=None):
    """Wrap imaging data into an Experiment with bias + linear regressor."""
    from glow.experiment.exper import Experiment
    x = np.arange(num_img, dtype=float).reshape(1, -1)
    contrast = np.array([True])
    return Experiment(x=x, contrast=contrast, y=y, mask_idx=mask_idx,
                      meta=meta, add_bias=True)


def _impose_and_run(exp, effect_llr, mask_target=None, seed=42,
                    roughness=None):
    """Optionally impose an effect, run AnalysisGLOW, and launch the viewer."""
    from glow.effect.extent import ExtenterMinVar
    from glow.analysis import AnalysisGLOW
    from glow.viewer import launch

    if effect_llr > 0:
        n_vox = (exp.mask_idx >= 0).sum()
        n_effect = max(int(0.15 * n_vox), 10)
        extenter = ExtenterMinVar(n_vox=n_effect)
        rough_str = f', roughness={roughness}' if roughness is not None else ''
        print(f'  imposing effect (llr={effect_llr}{rough_str}) in ~{n_effect}'
              f' voxels ({100 * n_effect / n_vox:.0f}% of mask) ...')
        try:
            exp_eff, effect = exp.impose_effect(
                effect_llr=effect_llr, extenter=extenter, seed=seed,
                roughness=roughness)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, AssertionError):
            print('  (extenter failed, falling back to sphere mask)')
            shape = exp.mask_idx.shape
            center = np.array([s // 2 for s in shape])
            coords = np.indices(shape).reshape(len(shape), -1).T
            dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
            target_n = n_effect
            radius = 1.0
            while (dist <= radius).sum() < target_n and radius < max(shape):
                radius += 0.5
            sphere = (dist <= radius).reshape(shape) & (exp.mask_idx >= 0)
            exp_eff, effect = exp.impose_effect(
                effect_llr=effect_llr, mask=sphere, seed=seed,
                roughness=roughness)
        mask_target = effect.mask
    else:
        exp_eff = exp
        print('  no effect imposed')

    print('  running AnalysisGLOW (n_perm_fwer=200, n_perm_inner=200) ...')
    ana = AnalysisGLOW(exp_eff, n_perm_fwer=200,
                       n_perm_inner=200,
                       n_jobs_perm=-1,
                       verbose=True)
    n_eff = len(ana.effect_list)
    print(f'  found {n_eff} effect{"s" if n_eff != 1 else ""}')
    return ana, mask_target


def _load_dti_mean(dim):
    """Load pre-computed mean FA/MD images.

    Args:
        dim: '2d' or '3d'

    Returns:
        fa, md (np.array), mask (bool array)
    """
    import nibabel as nib
    suffix = '_axial' if dim == '2d' else ''
    fa = nib.load(str(_data_path(f'hcp_mean_fa{suffix}.nii.gz'))).get_fdata()
    md = nib.load(str(_data_path(f'hcp_mean_md{suffix}.nii.gz'))).get_fdata()
    mask = fa > 0
    return fa.astype(np.float64), md.astype(np.float64), mask


def _sample_dti(mean_imgs, mask, num_img=_NUM_IMG, noise_frac=0.15,
                seed=42):
    """Generate synthetic DTI images: mean + WGN.

    Args:
        mean_imgs: dict of feat_name -> (spatial_shape) array
        mask: boolean mask
        num_img: number of synthetic images
        noise_frac: noise std as fraction of feature std within mask

    Returns:
        y (b, num_img, num_vox), mask_idx, y_features
    """
    from glow.mask import get_mask_idx
    mask_idx = get_mask_idx(mask)
    n_vox = int(mask.sum())
    feat_names = list(mean_imgs.keys())
    b = len(feat_names)

    rng = np.random.default_rng(seed)
    y = np.empty((b, num_img, n_vox), dtype=np.float64)
    for fi, name in enumerate(feat_names):
        vals = mean_imgs[name][mask]
        sigma = max(vals.std() * noise_frac, 1e-6)
        for i in range(num_img):
            y[fi, i, :] = vals + rng.normal(0, sigma, n_vox)

    return y, mask_idx, feat_names


# ---------------------------------------------------------------------------
# Demo builders (one per image-set choice)
# ---------------------------------------------------------------------------

def _demo_wgn_2d(b, effect_llr, seed=0, roughness=None, num_img=_NUM_IMG):
    """2D White Gaussian Noise demo."""
    from glow.experiment.exper import ExperimentImageOnly
    shape = (64, 64)
    print(f'  building 2D WGN: shape={shape}, b={b}, num_img={num_img}')
    exp_img = ExperimentImageOnly.from_gauss(
        b=b, num_img=num_img, shape=shape, seed=seed)
    exp = _build_experiment(exp_img.y, exp_img.mask_idx,
                            num_img=num_img, meta=exp_img.meta)
    ana, mask_target = _impose_and_run(exp, effect_llr, seed=seed,
                                       roughness=roughness)
    return ana, mask_target


def _demo_mandrill(channels, effect_llr, seed=0, roughness=None,
                   num_img=_NUM_IMG):
    """2D Mandrill RGB demo."""
    from PIL import Image
    from glow.mask import get_mask_idx

    img_path = _data_path('mandrill_small.png')
    print(f'  loading {img_path}')
    img_arr = np.array(Image.open(img_path)).astype(np.float64)  # (H, W, 3)
    h, w, _ = img_arr.shape

    channel_map = {'red': 0, 'green': 1, 'blue': 2}
    if channels == 'all':
        feat_indices = [0, 1, 2]
        feat_names = ['red', 'green', 'blue']
    else:
        feat_indices = [channel_map[channels]]
        feat_names = [channels]
    b = len(feat_indices)

    num_vox = h * w
    pixel_flat = img_arr.reshape(num_vox, 3).T  # (3, num_vox)
    rng = np.random.default_rng(seed)
    y = np.empty((b, num_img, num_vox))
    for fi, ci in enumerate(feat_indices):
        base = pixel_flat[ci]
        noise_scale = 15.0
        for i in range(num_img):
            y[fi, i, :] = base + rng.normal(0, noise_scale, num_vox)

    mask_idx = get_mask_idx(np.ones((h, w), dtype=bool))
    meta = {'features': feat_names}
    exp = _build_experiment(y, mask_idx, num_img=num_img, meta=meta)
    print(f'  mandrill: {h}x{w}, features={feat_names}, num_img={num_img}')
    ana, mask_target = _impose_and_run(exp, effect_llr, seed=seed,
                                       roughness=roughness)
    return ana, mask_target


def _demo_dti_2d(features, effect_llr, seed=0, roughness=None,
                 num_img=_NUM_IMG):
    """2D Axial Slice DTI demo."""
    fa, md, mask = _load_dti_mean('2d')
    mean_imgs = {}
    feat_names_map = {'fa': ('Fractional Anisotropy', fa),
                      'md': ('Mean Diffusivity', md)}
    if features == 'all':
        for key in ('fa', 'md'):
            mean_imgs[feat_names_map[key][0]] = feat_names_map[key][1]
    else:
        name, arr = feat_names_map[features]
        mean_imgs[name] = arr

    print(f'  2D axial DTI: shape={fa.shape}, mask={mask.sum()} vox, '
          f'features={list(mean_imgs.keys())}, num_img={num_img}')
    y, mask_idx, feat_names = _sample_dti(mean_imgs, mask, num_img=num_img,
                                          seed=seed)
    meta = {'features': feat_names}
    exp = _build_experiment(y, mask_idx, num_img=num_img, meta=meta)
    ana, mask_target = _impose_and_run(exp, effect_llr, seed=seed,
                                       roughness=roughness)
    return ana, mask_target


def _demo_wgn_3d(b, effect_llr, seed=0, roughness=None, num_img=_NUM_IMG):
    """3D White Gaussian Noise demo."""
    from glow.experiment.exper import ExperimentImageOnly
    shape = (15, 15, 15)
    print(f'  building 3D WGN: shape={shape}, b={b}, num_img={num_img}')
    exp_img = ExperimentImageOnly.from_gauss(
        b=b, num_img=num_img, shape=shape, seed=seed)
    exp = _build_experiment(exp_img.y, exp_img.mask_idx,
                            num_img=num_img, meta=exp_img.meta)
    ana, mask_target = _impose_and_run(exp, effect_llr, seed=seed,
                                       roughness=roughness)
    return ana, mask_target


def _demo_dti_3d(features, effect_llr, seed=0, roughness=None,
                 num_img=_NUM_IMG):
    """3D DTI demo."""
    fa, md, mask = _load_dti_mean('3d')
    mean_imgs = {}
    feat_names_map = {'fa': ('Fractional Anisotropy', fa),
                      'md': ('Mean Diffusivity', md)}
    if features == 'all':
        for key in ('fa', 'md'):
            mean_imgs[feat_names_map[key][0]] = feat_names_map[key][1]
    else:
        name, arr = feat_names_map[features]
        mean_imgs[name] = arr

    print(f'  3D DTI: shape={fa.shape}, mask={mask.sum()} vox, '
          f'features={list(mean_imgs.keys())}, num_img={num_img}')
    y, mask_idx, feat_names = _sample_dti(mean_imgs, mask, num_img=num_img,
                                          seed=seed)
    meta = {'features': feat_names}
    exp = _build_experiment(y, mask_idx, num_img=num_img, meta=meta)
    ana, mask_target = _impose_and_run(exp, effect_llr, seed=seed,
                                       roughness=roughness)
    return ana, mask_target


# ---------------------------------------------------------------------------
# Interactive demo entry point
# ---------------------------------------------------------------------------

def _run_demo():
    """Interactive demo: prompt for image set, features, effect severity."""
    from glow.viewer import launch

    print('\n=== GLOW Viewer Demo ===\n')

    # --- 1) image set ---
    image_set = _choose('Select an image set:', [
        ('wgn2d',      '2D White Gaussian Noise'),
        ('mandrill',   '2D Mandrill RGB'),
        ('dti2d',      '2D Axial Slice DTI'),
        ('wgn3d',      '3D White Gaussian Noise'),
        ('dti3d',      '3D DTI'),
    ], default='wgn2d')

    # --- 2) feature selection (depends on image set) ---
    feat_choice = None
    b_choice = 1
    if image_set in ('wgn2d', 'wgn3d'):
        b_choice = _choose_int(
            'Number of image features (b):', default=1, lo=1, hi=20)
    elif image_set == 'mandrill':
        feat_choice = _choose('Select colour channel(s):', [
            ('red',   'Red'),
            ('blue',  'Blue'),
            ('green', 'Green'),
            ('all',   'All (RGB)'),
        ], default='all')
    elif image_set in ('dti2d', 'dti3d'):
        feat_choice = _choose('Select DTI feature(s):', [
            ('fa',  'Fractional Anisotropy (FA)'),
            ('md',  'Mean Diffusivity (MD)'),
            ('all', 'All (FA + MD)'),
        ], default='all')

    # --- 3) effect severity ---
    # values are per-voxel size-normalised LLR; observed region LLR
    # ≈ value × region_size (e.g. 0.5 × 614 ≈ 307 for medium / mandrill).
    severity = _choose('Effect severity to impose (per-voxel LLR):', [
        ('none',   'None     (effect_llr/vox = 0)'),
        ('mild',   'Mild     (effect_llr/vox = 0.25)'),
        ('medium', 'Medium   (effect_llr/vox = 0.5)'),
        ('strong', 'Strong   (effect_llr/vox = 1.0)'),
    ], default='medium')
    effect_llr = _EFFECT_MAP[severity]

    # --- 3b) roughness (only when an effect is imposed) ---
    roughness = None
    if effect_llr > 0:
        rough_choice = _choose('Roughness (spatial covariance fraction):', [
            ('natural', 'Natural (no roughness control)'),
            ('smooth',  'Smooth (0.0 — all residual mean)'),
            ('low',     'Low   (0.25)'),
            ('mid',     'Mid   (0.50)'),
            ('high',    'High  (0.75)'),
            ('rough',   'Rough (1.0 — all spatial covariance)'),
        ], default='natural')
        _ROUGH_MAP = {
            'natural': None, 'smooth': 0.0, 'low': 0.25,
            'mid': 0.5, 'high': 0.75, 'rough': 1.0,
        }
        roughness = _ROUGH_MAP[rough_choice]

    # --- 4) number of images (subjects) ---
    # LLR scales roughly with sample size, so num_img controls how
    # peaked the H0 LLR distribution is and how much the mean drifts
    # with region size.  Default 12 is fast; production scale is 100+.
    num_img = _choose_int('Number of images (subjects):',
                           default=_NUM_IMG, lo=4, hi=1000)

    # --- 5) random seed ---
    seed = _choose_int('Random seed:', default=0, lo=0, hi=2**31 - 1)

    # --- build ---
    rough_str = f', roughness={roughness}' if roughness is not None else ''
    print(f'\n  Building demo (effect_llr={effect_llr:.2g}{rough_str},'
          f' num_img={num_img}, seed={seed}) ...')
    kw = dict(effect_llr=effect_llr, seed=seed, roughness=roughness,
              num_img=num_img)
    if image_set == 'wgn2d':
        ana, mask_target = _demo_wgn_2d(b_choice, **kw)
    elif image_set == 'mandrill':
        ana, mask_target = _demo_mandrill(feat_choice, **kw)
    elif image_set == 'dti2d':
        ana, mask_target = _demo_dti_2d(feat_choice, **kw)
    elif image_set == 'wgn3d':
        ana, mask_target = _demo_wgn_3d(b_choice, **kw)
    elif image_set == 'dti3d':
        ana, mask_target = _demo_dti_3d(feat_choice, **kw)

    launch(ana, mask_target=mask_target)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog='python -m glow.viewer',
        description='Launch the glow:viewer interactive dashboard.')

    parser.add_argument(
        'analysis', nargs='?', default=None,
        help='Path to a pickled AnalysisGLOW object (.pkl, .p, .p.gz)')
    parser.add_argument(
        '--mask', default=None,
        help='Path to a target mask (.nii, .nii.gz, .npy, or pickled Effect)')
    parser.add_argument(
        '--demo', action='store_true',
        help='Run interactive demo')
    parser.add_argument(
        '--port', type=int, default=8050,
        help='Server port (default: 8050)')
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable Dash debug mode')
    parser.add_argument(
        '--csv', default=None,
        help='Path to a CSV with extra per-region data (must contain a '
             'region_idx column).')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show Dash/Werkzeug request logs (suppressed by default)')

    args = parser.parse_args()

    if args.demo:
        _run_demo()
        return

    if args.analysis is None:
        parser.error('either --demo or an analysis file path is required')

    print(f'Loading analysis from {args.analysis} ...')
    ana = _load_analysis(args.analysis)

    mask_target = None
    if args.mask is not None:
        print(f'Loading mask from {args.mask} ...')
        mask_target = _load_mask(args.mask, ana.exp.mask_idx)
        print(f'  mask shape: {mask_target.shape}, '
              f'{mask_target.sum()} voxels active')

    extra_df = None
    if args.csv is not None:
        import pandas as pd
        print(f'Loading CSV from {args.csv} ...')
        extra_df = pd.read_csv(args.csv)
        if 'region_idx' not in extra_df.columns:
            parser.error('CSV must contain a region_idx column')
        print(f'  {len(extra_df)} rows, columns: {list(extra_df.columns)}')

    from glow.viewer import launch
    launch(ana, mask_target=mask_target, port=args.port, debug=args.debug,
           extra_df=extra_df, quiet=not args.verbose)


if __name__ == '__main__':
    main()
