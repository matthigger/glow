"""Build a benchmark Experiment from synthetic or HCP-YA imaging data.

data_factory_wgn / data_factory_hcp each build an image-only Experiment,
sample a design matrix x onto it, and crop it to an Extenter's support -- the
clean (effect-free) data a trial then plants an effect on. Each build is
memoised on disk (MEMORY) with the recorder nested inside the cache, so a
cache hit returns the stored Experiment and only a real (cache-miss) build is
recorded. The recorder keys each build by joblib's own args hash (see
Recorder), so a record lines up one-to-one with the cached artifact on disk.
data_factory dispatches to the builders on source.
"""
import joblib

from glow.effect import Extenter
from glow.experiment import ExperimentImageOnly

from . import hcp
from .file import get_path_cache, get_path_records
from .recorder import Recorder

# disk memoisation of the experiment builds, keyed on the build inputs, so a
# repeated (source, ...) cell is loaded rather than rebuilt across runs.
MEMORY = joblib.Memory(get_path_cache(), verbose=0)

# captures each build's inputs / output / timing for provenance (see Recorder).
# Keyed by joblib's args hash, mirrored to the records dir beside the cache.
RECORDER = Recorder(folder=get_path_records())


def _sample_x_and_crop(exp_img, *, a: int, contrast, has_bias: bool,
                       extenter: Extenter, seed: int):
    """Sample the design matrix x onto an image-only experiment, then crop.

    Args:
        exp_img: image-only Experiment (y of shape (b, num_img, num_vox)).
        a (int): design-matrix feature count; ignored when contrast is given.
        contrast (np.array): (a,) boolean, True for features of interest, or
            None to sample a interest-only features.
        has_bias (bool): prepend a bias (all-ones) column to x.
        extenter (Extenter): extent whose support the experiment is cropped
            to, or None for no crop.
        seed (int): RNG seed for the x sample.

    Returns:
        Experiment with x and contrast attached, cropped to the extenter
        support when set.
    """
    # sample_x requires exactly one of a / contrast
    exp = exp_img.sample_x(a=None if contrast is not None else a,
                           contrast=contrast, seed=seed, add_bias=has_bias)
    if extenter is not None:
        # data-driven extenters (ExtenterMinVar) need y; geometric ones ignore it
        exp = exp.apply_mask(extenter(mask_idx=exp.mask_idx, y=exp.y))
    return exp


@MEMORY.cache
@RECORDER(output_name='exp')
def data_factory_wgn(*, shape: tuple = (5, 5, 5), b: int = 2,
                     num_img: int = 100, a: int = 1, contrast=None,
                     has_bias: bool = True, extenter: Extenter = None,
                     seed: int = 0):
    """Build a white-Gaussian-noise Experiment (no planted effect).

    Args:
        shape (tuple[int]): spatial shape of each image; num_vox is its product.
        b (int): imaging features per voxel (y channels).
        num_img (int): number of images (subjects).
        a (int): design-matrix feature count (see _sample_x_and_crop).
        contrast (np.array): (a,) boolean of features of interest, or None.
        has_bias (bool): prepend a bias column to x.
        extenter (Extenter): extent to crop to, or None.
        seed (int): RNG seed shared by the image draw and the x sample.

    Returns:
        Experiment with y of shape (b, num_img, num_vox), x, and contrast,
        cropped to the extenter support when set.
    """
    exp_img = ExperimentImageOnly.from_gauss(
        shape=shape, b=b, num_img=num_img, seed=seed)
    return _sample_x_and_crop(exp_img, a=a, contrast=contrast,
                              has_bias=has_bias, extenter=extenter, seed=seed)


@MEMORY.cache
@RECORDER(output_name='exp')
def data_factory_hcp(*, hcp_feats: tuple = hcp.HCP_FEATS, a: int = 1,
                     contrast=None, has_bias: bool = True,
                     extenter: Extenter = None, seed: int = 0):
    """Build an HCP-YA diffusion-microstructure Experiment (no planted effect).

    Args:
        hcp_feats (tuple[str]): subset of hcp.HCP_FEATS to load, in order;
            its length is b.
        a (int): design-matrix feature count (see _sample_x_and_crop).
        contrast (np.array): (a,) boolean of features of interest, or None.
        has_bias (bool): prepend a bias column to x.
        extenter (Extenter): extent to crop to, or None.
        seed (int): RNG seed for the x sample.

    Returns:
        Experiment with x and contrast attached, cropped to the extenter
        support when set.
    """
    # the archive's brain mask is the analysis support (NODDI isovf is
    # legitimately zero in-brain, so the maps can't infer it); see hcp.py
    folder = hcp.ensure_hcp_data()
    exp_img = ExperimentImageOnly.from_search(
        folder=folder, sbj_regex=hcp.SBJ_REGEX,
        img_glob_dict={feat: hcp.IMG_GLOB_DICT[feat] for feat in hcp_feats},
        mask=next(folder.glob(hcp.MASK_GLOB)))
    return _sample_x_and_crop(exp_img, a=a, contrast=contrast,
                              has_bias=has_bias, extenter=extenter, seed=seed)


def data_factory(source: str, **kwargs):
    """Build a clean Experiment from 'wgn' or 'hcp' (forwards kwargs).

    Args:
        source (str): 'wgn' or 'hcp', selecting the builder kwargs go to.

    Returns:
        the selected builder's Experiment.

    Raises:
        ValueError: if source is neither 'wgn' nor 'hcp'.
    """
    if source == 'wgn':
        return data_factory_wgn(**kwargs)
    if source == 'hcp':
        return data_factory_hcp(**kwargs)
    raise ValueError(f"source must be 'wgn' or 'hcp', got {source!r}")
