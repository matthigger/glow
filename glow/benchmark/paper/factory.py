"""Shared builders for the paper benchmarks: scalar axes -> heavy objects.

The paper catalogue (config.py) keeps every trial axis as a plain
scalar/categorical in iter_kwargs (source, b, num_img, n_vox_eff,
effect_llr, seed, ...); the trial functions (run.py) turn those scalars
back into the heavy DataSource / Extenter objects here. Both config and
run import from this module, so it must not import either of them (no
import cycle).

The DataSource build is memoised on its scalar identity (_ds_factory is
lru_cached), so every method / effect_llr / effect-seed trial sharing one
(source, b, num_img, feats) reuses a single ds.exp build -- the same
amortisation the old object-valued catalogue got from sharing one ds
instance across the seed loop.
"""
import math
from functools import lru_cache

import numpy as np

from glow.benchmark import hcp
from glow.benchmark.data import DataSourceHCP, DataSourceWGN
from glow.effect import ExtenterSphere


# ---------- shared structural constants -------------------------------------
CROP_N_VOX = 25_000
_WGN_SIDE_3D = math.ceil(CROP_N_VOX ** (1 / 3))

# Data-source seed is held fixed (only the effect seed is swept), matching
# the pre-flatten catalogue: the base images / design matrix are constant
# across the effect-seed replicates within a structural cell.
DS_SEED = 0

# Every source is cropped to the same connected sphere so num_vox matches
# across WGN and HCP.
_CROP_EXTENTER = ExtenterSphere(n_vox=CROP_N_VOX, connected=True)


def sample_hcp_feats(b: int, seed: int) -> tuple:
    """Draw b distinct HCP features from hcp.HCP_FEATS, per seed.

    Each effect seed gets its own random feature subset so a b-feature
    result averages over which features were chosen rather than fixing an
    arbitrary one. Sorted for a stable, order-invariant identity. The
    b-sweep grid + random draws size themselves off the six-feature pool
    (config.B_GRID clamps to it).

    Args:
        b (int): number of features to draw
        seed (int): effect seed; also seeds the feature draw

    Returns:
        a sorted tuple of b feature names from hcp.HCP_FEATS

    Raises:
        ValueError: if b exceeds the pool size
    """
    if b > len(hcp.HCP_FEATS):
        raise ValueError(
            f'requested b={b} > HCP feature pool {len(hcp.HCP_FEATS)}')
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(hcp.HCP_FEATS), size=b, replace=False)
    return tuple(sorted(hcp.HCP_FEATS[i] for i in idx))


@lru_cache(maxsize=None)
def _ds_factory(source: str, b: int, num_img: int, feats: tuple):
    """Build (and memoise) one DataSource from its scalar identity.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count (WGN only; HCP uses len(feats))
        num_img (int): subject count (WGN only; HCP uses its full cohort)
        feats (tuple | None): HCP feature subset, or None for WGN

    Returns:
        the DataSource; its .exp is built once per identity and shared

    Raises:
        ValueError: if source is unknown
    """
    if source == 'wgn':
        return DataSourceWGN(shape=(_WGN_SIDE_3D,) * 3, b=b, num_img=num_img,
                             seed=DS_SEED, extenter=_CROP_EXTENTER)
    if source == 'hcp':
        return DataSourceHCP(hcp_feats=feats, seed=DS_SEED,
                             extenter=_CROP_EXTENTER)
    raise ValueError(f'unknown source: {source!r}')


def build_ds(source: str, *, b: int, num_img: int, seed: int):
    """Resolve a trial's scalar axes to a memoised DataSource.

    For HCP, draws the b-feature subset for this seed (sample_hcp_feats)
    and keys the build on it; for WGN, feats is None and b/num_img drive
    the build directly.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN only)
        seed (int): effect seed (selects the HCP feature subset)

    Returns:
        ds: the memoised DataSource for this cell
        feats (tuple | None): the realized HCP feature subset, or None
    """
    feats = sample_hcp_feats(b, seed) if source == 'hcp' else None
    return _ds_factory(source, b, num_img, feats), feats
