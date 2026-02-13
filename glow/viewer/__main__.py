"""CLI entry point for the glow viewer.

Usage::

    # built-in 3D demo (15x15x15 WGN cube with sphere effect)
    python -m glow.viewer --demo

    # built-in 2D demo (mandrill image with ExtenterMinVar effect)
    python -m glow.viewer --demo2d

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


def _load_analysis(path):
    """Load a pickled AnalysisGLOW from a file.

    Supports plain pickle (.pkl, .p) and gzip-compressed pickle (.p.gz).
    """
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
    """Load a target mask from a file.

    Supports:
        - .nii / .nii.gz  (nibabel, boolean volume)
        - .npy            (numpy boolean array)
        - anything else   (try pickle, expect Effect with .mask attribute)

    Args:
        path (str): file path
        mask_idx (np.array): analysis mask_idx (for shape validation)

    Returns:
        mask (np.array): boolean mask, same shape as mask_idx
    """
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
        # try pickle (expect an Effect with .mask)
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
            f'{mask_idx.shape}.  Ensure the mask is in the same voxel space.')

    return mask


def _run_demo():
    """Build and launch a small demo: 15x15x15 WGN cube with sphere effect."""
    from glow.experiment.exper import Experiment
    from glow.experiment.analysis import AnalysisGLOW
    from glow.viewer import launch

    print('Building demo: 15x15x15 WGN cube with sphere effect ...')

    shape = (15, 15, 15)
    center = np.array([s // 2 for s in shape])

    # build a sphere mask (radius ~5 voxels)
    coords = np.indices(shape).reshape(3, -1).T  # (N, 3)
    dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
    sphere_flat = dist <= 5.0
    mask_sphere = sphere_flat.reshape(shape)
    n_sphere = mask_sphere.sum()
    print(f'  sphere: {n_sphere} voxels at center {tuple(center)}, radius=5')

    # create experiment (WGN, b=1 feature, 12 images)
    exp = Experiment.from_gauss(b=1, num_img=12, shape=shape, seed=0, a=2)
    print(f'  experiment: y.shape={exp.y.shape}')

    # impose effect inside sphere
    hotel_tr = 30.0
    exp_eff, effect = exp.impose_effect(hotel_tr=hotel_tr, mask=mask_sphere,
                                        seed=0)
    print(f'  imposed hotel_tr={hotel_tr} in sphere')

    # run analysis (serial, small)
    print('  running AnalysisGLOW (n_perm=20) ...')
    ana = AnalysisGLOW(exp_eff, n_perm=20, verbose=True)
    print(f'  found {len(ana.effect_list)} effects')

    # launch viewer with sphere as target mask
    launch(ana, mask_target=mask_sphere)


def _run_demo2d():
    """Build and launch a 2D demo: mandrill image with ExtenterMinVar effect."""
    from PIL import Image

    from glow.effect.extent import ExtenterMinVar
    from glow.experiment.exper import Experiment, ExperimentImageOnly
    from glow.experiment.analysis import AnalysisGLOW
    from glow.mask import get_mask_idx
    from glow.viewer import launch

    print('Building demo2d: mandrill_small.png with ExtenterMinVar effect ...')

    # locate the image relative to this package (src/glow/viewer/__main__.py)
    img_path = pathlib.Path(__file__).resolve().parents[2] / \
        'test' / 'data' / 'mandrill_small.png'
    if not img_path.exists():
        # fallback: try from workspace root
        img_path = pathlib.Path('test/data/mandrill_small.png')
    print(f'  loading {img_path}')

    img_arr = np.array(Image.open(img_path)).astype(np.float64)  # (H, W, 3)
    h, w, b = img_arr.shape
    num_vox = h * w
    print(f'  image: {h}x{w}, {b} channels, {num_vox} pixels')

    # treat each pixel as a voxel, each channel as a feature (b=3)
    # build synthetic subjects: replicate image + gaussian noise
    num_img = 12
    rng = np.random.default_rng(seed=42)
    # y shape: (b, num_img, num_vox)
    pixel_flat = img_arr.reshape(num_vox, b).T  # (b, num_vox)
    noise_scale = 15.0  # std-dev of per-pixel noise
    y = np.empty((b, num_img, num_vox))
    for i in range(num_img):
        y[:, i, :] = pixel_flat + rng.normal(0, noise_scale, pixel_flat.shape)

    # build experiment
    shape = (h, w)
    mask_idx = get_mask_idx(np.ones(shape, dtype=bool))
    exp_img = ExperimentImageOnly(y=y, mask_idx=mask_idx)
    exp = exp_img.sample_x(a=2, seed=42, add_bias=True)
    print(f'  experiment: y.shape={exp.y.shape}')

    # use ExtenterMinVar for ~15% of pixels
    n_effect = int(0.15 * num_vox)
    extenter = ExtenterMinVar(n=n_effect)
    print(f'  growing ExtenterMinVar extent ({n_effect} pixels, '
          f'{100 * n_effect / num_vox:.1f}% of image) ...')

    hotel_tr = 30.0
    exp_eff, effect = exp.impose_effect(hotel_tr=hotel_tr, extenter=extenter,
                                        seed=42)
    effect_mask = effect.mask
    print(f'  imposed hotel_tr={hotel_tr} in {effect_mask.sum()} pixels')

    # run analysis
    print('  running AnalysisGLOW (n_perm=20) ...')
    ana = AnalysisGLOW(exp_eff, n_perm=20, verbose=True)
    print(f'  found {len(ana.effect_list)} effects')

    # launch viewer with effect mask as target
    launch(ana, mask_target=effect_mask,
           feature_names=['red', 'green', 'blue'])


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
        help='Run built-in 3D demo (15x15x15 WGN cube with sphere effect)')
    parser.add_argument(
        '--demo2d', action='store_true',
        help='Run built-in 2D demo (mandrill image with ExtenterMinVar effect)')
    parser.add_argument(
        '--port', type=int, default=8050,
        help='Server port (default: 8050)')
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable Dash debug mode')

    args = parser.parse_args()

    if args.demo:
        _run_demo()
        return

    if args.demo2d:
        _run_demo2d()
        return

    if args.analysis is None:
        parser.error(
            'either --demo, --demo2d, or an analysis file path is required')

    # load analysis
    print(f'Loading analysis from {args.analysis} ...')
    ana = _load_analysis(args.analysis)

    # load mask (if provided)
    mask_target = None
    if args.mask is not None:
        print(f'Loading mask from {args.mask} ...')
        mask_target = _load_mask(args.mask, ana.exp.mask_idx)
        print(f'  mask shape: {mask_target.shape}, '
              f'{mask_target.sum()} voxels active')

    from glow.viewer import launch
    launch(ana, mask_target=mask_target, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
