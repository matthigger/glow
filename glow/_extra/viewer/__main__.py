"""CLI entry point for the glow viewer.

Usage:
    # interactive demo
    python -m glow._extra.viewer --demo

    # load a bundle pickle {'ana': AnalysisGLOW, 'exp': Experiment, ...}
    # (as written by glow._extra.viewer.web.bake_demos); a bare AnalysisGLOW
    # does not carry the experiment the viewer needs
    python -m glow._extra.viewer bundle.p.gz

    # with a target mask (nifti, numpy, or pickled EffectEstimate)
    python -m glow._extra.viewer bundle.p.gz --mask target.nii.gz
    python -m glow._extra.viewer bundle.p.gz --mask target.npy
    python -m glow._extra.viewer bundle.p.gz --mask effect.pkl
"""

import argparse
import gzip
import pathlib
import pickle
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Data paths (relative to this package: src/glow/_extra/viewer/__main__.py)
# ---------------------------------------------------------------------------

_DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / 'test' / 'data'


def _data_path(filename):
    """Resolve a demo data file, falling back to test/data under the cwd."""
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
                f'with a .mask attribute (e.g. EffectEstimate).')

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

    Args:
        prompt (str): heading shown above the options.
        options (list[tuple[str, str]]): (key, label) pairs; the key is
            returned, the label is shown.
        default (str | None): key selected when the user just presses Enter.

    Returns:
        the key of the chosen option.
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
# effect_llr * |region|. So a "medium" 0.5 effect on a 614-voxel planted
# region produces an observed region LLR of ~307, not 0.5.
_EFFECT_MAP = {
    'none':   0.0,
    'mild':   0.25,
    'medium': 0.5,
    'strong': 1.0,
}


def _impose_and_run(exp, effect_llr, mask_target=None, seed=42):
    """Optionally impose an effect and run AnalysisGLOW.

    Returns:
        ana (AnalysisGLOW): the fitted analysis.
        exp_eff (Experiment): the experiment it was fit on (the analysis no
            longer stores it; the viewer needs it).
        mask_target (np.array | None): planted target mask, same shape as
            exp.mask_idx; None when no effect imposed.
    """
    from glow.effect.extent import ExtenterMinVar
    from glow.analysis import AnalysisGLOW

    if effect_llr > 0:
        from glow.effect import EffectSynthetic
        n_vox = (exp.mask_idx >= 0).sum()
        n_effect = max(int(0.15 * n_vox), 10)
        extenter = ExtenterMinVar(n_vox=n_effect, seed=seed)
        print(f'  imposing effect (llr={effect_llr}) in ~{n_effect}'
              f' voxels ({100 * n_effect / n_vox:.0f}% of mask) ...')
        try:
            effect = EffectSynthetic(extenter=extenter, effect_llr=effect_llr)
            fit = effect.fit(exp)
        except (ValueError, RuntimeError, np.linalg.LinAlgError,
                AssertionError):
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
            effect = EffectSynthetic(mask=sphere, effect_llr=effect_llr)
            fit = effect.fit(exp)
        exp_eff, mask_target = fit
    else:
        exp_eff = exp
        print('  no effect imposed')

    print('  running AnalysisGLOW (n_perm_fwer=200) ...')
    ana = AnalysisGLOW(n_perm_fwer=200).fit(exp_eff, verbose=True)
    n_eff = len(ana.effect_list)
    print(f'  found {n_eff} effect{"s" if n_eff != 1 else ""}')
    return ana, exp_eff, mask_target


# ---------------------------------------------------------------------------
# Demo builders (one per image-set choice)
# ---------------------------------------------------------------------------

def _demo_wgn_2d(b, effect_llr, seed=0, num_img=_NUM_IMG):
    """2D White Gaussian Noise demo."""
    from glow.experiment.exper import Experiment
    shape = (64, 64)
    print(f'  building 2D WGN: shape={shape}, b={b}, num_img={num_img}')
    exp = Experiment.from_gauss(a=1, b=b, num_img=num_img, shape=shape,
                                 seed=seed, add_bias=True)
    ana, exp_eff, mask_target = _impose_and_run(exp, effect_llr, seed=seed)
    return ana, exp_eff, mask_target


def _demo_mandrill(channels, effect_llr, seed=0, num_img=_NUM_IMG):
    """Build a 2D Mandrill RGB demo via the public factories.

    Goes through from_paths + bootstrap_img + sample_x, the same path a
    real user would call.
    """
    from glow.experiment.exper import ExperimentImageOnly

    img_path = _data_path('mandrill_small.png')
    print(f'  loading {img_path}')
    # load_image_color splits an RGB PNG into 3 features automatically
    img_only = ExperimentImageOnly.from_paths(
        {'mandrill': {'rgb': str(img_path)}},
        channel_names={'rgb': ['red', 'green', 'blue']})

    # noise_scale is data-relative (multiplies sample-cov^0.5), so 0.3 is
    # ~15 absolute units for mandrill
    img_only = img_only.bootstrap_img(num_img, seed=seed, noise_scale=0.3)

    # contrast is one feature-of-interest plus a bias column
    exp = img_only.sample_x(a=1, seed=seed, add_bias=True)

    if channels != 'all':
        print(f'  (note: channel selection ({channels!r}) is currently'
              f' shown for all 3 RGB channels)')
    h, w = exp.mask_idx.shape
    print(f'  mandrill: {h}x{w}, features={exp.meta["features"]}, '
          f'num_img={num_img}')
    ana, exp_eff, mask_target = _impose_and_run(exp, effect_llr, seed=seed)
    return ana, exp_eff, mask_target


def _demo_dti_2d(features, effect_llr, seed=0, num_img=_NUM_IMG):
    """Build a 2D axial-slice DTI demo (see _build_dti_demo)."""
    return _build_dti_demo('2d', features, effect_llr, seed=seed,
                           num_img=num_img)


def _demo_wgn_3d(b, effect_llr, seed=0, num_img=_NUM_IMG):
    """3D White Gaussian Noise demo."""
    from glow.experiment.exper import Experiment
    shape = (15, 15, 15)
    print(f'  building 3D WGN: shape={shape}, b={b}, num_img={num_img}')
    exp = Experiment.from_gauss(a=1, b=b, num_img=num_img, shape=shape,
                                 seed=seed, add_bias=True)
    ana, exp_eff, mask_target = _impose_and_run(exp, effect_llr, seed=seed)
    return ana, exp_eff, mask_target


def _demo_dti_3d(features, effect_llr, seed=0, num_img=_NUM_IMG):
    """Build a 3D DTI demo (see _build_dti_demo)."""
    return _build_dti_demo('3d', features, effect_llr, seed=seed,
                           num_img=num_img)


def _build_dti_demo(dim, features, effect_llr, seed, num_img):
    """Build a 2D or 3D DTI demo from mean fa/md niftis.

    Loads via from_paths (one 'mean' subject with feature-keyed paths),
    bootstraps to num_img noisy copies, attaches a random design via
    sample_x, then imposes an effect.
    """
    from glow.experiment.exper import ExperimentImageOnly
    suffix = '_axial' if dim == '2d' else ''
    fa_path = _data_path(f'hcp_mean_fa{suffix}.nii.gz')
    md_path = _data_path(f'hcp_mean_md{suffix}.nii.gz')
    feat_path_map = {'Fractional Anisotropy': str(fa_path),
                     'Mean Diffusivity': str(md_path)}
    if features == 'all':
        feat_paths = feat_path_map
    elif features == 'fa':
        feat_paths = {'Fractional Anisotropy': str(fa_path)}
    elif features == 'md':
        feat_paths = {'Mean Diffusivity': str(md_path)}
    else:
        raise ValueError(f'unknown DTI feature selection: {features!r}')

    print(f'  {dim} DTI: loading {list(feat_paths.keys())}, num_img={num_img}')
    img_only = ExperimentImageOnly.from_paths({'mean': feat_paths})
    img_only = img_only.bootstrap_img(num_img, seed=seed, noise_scale=0.15)
    exp = img_only.sample_x(a=1, seed=seed, add_bias=True)
    ana, exp_eff, mask_target = _impose_and_run(exp, effect_llr, seed=seed)
    return ana, exp_eff, mask_target


# ---------------------------------------------------------------------------
# Interactive demo entry point
# ---------------------------------------------------------------------------

def _run_demo():
    """Interactive demo: prompt for image set, features, effect severity."""
    from glow._extra.viewer import launch

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

    # --- 4) number of images (subjects) ---
    # LLR scales roughly with sample size, so num_img controls how
    # peaked the H0 LLR distribution is and how much the mean drifts
    # with region size.  Default 12 is fast; production scale is 100+.
    num_img = _choose_int('Number of images (subjects):',
                           default=_NUM_IMG, lo=4, hi=1000)

    # --- 5) random seed ---
    seed = _choose_int('Random seed:', default=0, lo=0, hi=2**31 - 1)

    # --- build ---
    print(f'\n  Building demo (effect_llr={effect_llr:.2g},'
          f' num_img={num_img}, seed={seed}) ...')
    kw = dict(effect_llr=effect_llr, seed=seed, num_img=num_img)
    if image_set == 'wgn2d':
        ana, exp, mask_target = _demo_wgn_2d(b_choice, **kw)
    elif image_set == 'mandrill':
        ana, exp, mask_target = _demo_mandrill(feat_choice, **kw)
    elif image_set == 'dti2d':
        ana, exp, mask_target = _demo_dti_2d(feat_choice, **kw)
    elif image_set == 'wgn3d':
        ana, exp, mask_target = _demo_wgn_3d(b_choice, **kw)
    elif image_set == 'dti3d':
        ana, exp, mask_target = _demo_dti_3d(feat_choice, **kw)

    launch(ana, exp, mask_target=mask_target)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Parse CLI args and launch the viewer (demo or loaded bundle)."""
    parser = argparse.ArgumentParser(
        prog='python -m glow._extra.viewer',
        description='Launch the glow:viewer interactive dashboard.')

    parser.add_argument(
        'analysis', nargs='?', default=None,
        help='Path to a pickled AnalysisGLOW object (.pkl, .p, .p.gz)')
    parser.add_argument(
        '--mask', default=None,
        help='Path to a target mask (.nii, .nii.gz, .npy, or pickled '
             'EffectEstimate)')
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
    parser.add_argument(
        '--min-vox', type=int, default=None,
        help='Scatter only regions with at least this many voxels (0 shows '
             'all).  Default: prompt when the tree exceeds --max-regions.')
    parser.add_argument(
        '--max-regions', type=int, default=10_000,
        help='Region ceiling used to pick a default --min-vox cutoff '
             '(default: 10000).')

    args = parser.parse_args()

    if args.demo:
        _run_demo()
        return

    if args.analysis is None:
        parser.error('either --demo or an analysis file path is required')

    print(f'Loading analysis from {args.analysis} ...')
    obj = _load_analysis(args.analysis)
    # the analysis does not carry exp, so the viewer needs a bundle
    # pickle {'ana', 'exp', ...} (as written by bake_demos), not a bare
    # AnalysisGLOW.
    if not (isinstance(obj, dict) and 'ana' in obj and 'exp' in obj):
        parser.error(
            'the viewer needs the experiment the analysis was fit on; pass a '
            "bundle pickle containing {'ana', 'exp'} (e.g. one written by "
            'glow._extra.viewer.web.bake_demos), not a bare AnalysisGLOW.')
    ana, exp = obj['ana'], obj['exp']

    mask_target = obj.get('mask_target')
    if args.mask is not None:
        print(f'Loading mask from {args.mask} ...')
        mask_target = _load_mask(args.mask, exp.mask_idx)
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

    from glow._extra.viewer import launch
    launch(ana, exp, mask_target=mask_target, port=args.port, debug=args.debug,
           extra_df=extra_df, quiet=not args.verbose,
           min_vox=args.min_vox, max_regions=args.max_regions)


if __name__ == '__main__':
    main()
