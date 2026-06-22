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
from collections import namedtuple
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

# Mutually independent per-trial sub-seeds (see derive_seeds).
TrialSeeds = namedtuple('TrialSeeds', 'ds feat effect')


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
def _ds_factory(source: str, b: int, num_img: int, feats: tuple,
                ds_seed: int):
    """Build (and memoise) one DataSource from its scalar identity.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count (WGN only; HCP uses len(feats))
        num_img (int): subject count (WGN only; HCP uses its full cohort)
        feats (tuple | None): HCP feature subset, or None for WGN
        ds_seed (int): seed for the X design + sphere crop. The shared
            DS_SEED gives one base experiment per cell (the sweeps); a
            per-trial seed gives an independent data realization (min_size).

    Returns:
        the DataSource; its .exp is built once per identity and shared

    Raises:
        ValueError: if source is unknown
    """
    # Every source is cropped to the same connected sphere (seed baked in,
    # resampled until contiguous) so num_vox matches across WGN and HCP.
    crop = ExtenterSphere(n_vox=CROP_N_VOX, connected=True,
                          contiguous=True, seed=ds_seed)
    if source == 'wgn':
        return DataSourceWGN(shape=(_WGN_SIDE_3D,) * 3, b=b, num_img=num_img,
                             seed=ds_seed, extenter=crop)
    if source == 'hcp':
        return DataSourceHCP(hcp_feats=feats, seed=ds_seed, extenter=crop)
    raise ValueError(f'unknown source: {source!r}')


def build_ds(source: str, *, b: int, num_img: int, seed: int,
             ds_seed: int = DS_SEED):
    """Resolve a trial's scalar axes to a memoised DataSource.

    For HCP, draws the b-feature subset for this seed (sample_hcp_feats) and
    keys the build on it; for WGN, feats is None and b/num_img drive the build
    directly. The X design + crop use ds_seed, defaulting to the shared
    DS_SEED (one base experiment per cell, the sweeps' policy). Pass a
    per-trial ds_seed (from derive_seeds) for an independent data realization,
    as min_size does.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN only)
        seed (int): feature-subset seed (HCP); unused for WGN
        ds_seed (int): seed for the X design + crop; default DS_SEED (shared)

    Returns:
        ds: the DataSource for this cell (memoised on its full identity)
        feats (tuple | None): the realized HCP feature subset, or None
    """
    feats = sample_hcp_feats(b, seed) if source == 'hcp' else None
    return _ds_factory(source, b, num_img, feats, ds_seed), feats


def derive_seeds(seed: int) -> TrialSeeds:
    """Split a trial seed into independent (ds, feat, effect) sub-seeds.

    A single trial seed otherwise drives the DataSource, the HCP feature draw,
    and the planted effect off one RNG stream, coupling them (e.g. the effect
    placement correlated with the data realization). SeedSequence.spawn yields
    three mutually independent child seeds, so an experiment that varies the
    data source per trial (min_size) keeps the effect independent of the data.
    Sweeps that share one base experiment leave ds at DS_SEED (build_ds's
    default) and use the raw trial seed, so they do not call this.

    Args:
        seed (int): the trial seed (an iter_kwargs axis).

    Returns:
        TrialSeeds(ds, feat, effect): three mutually independent sub-seeds.
    """
    ds, feat, effect = (int(s.generate_state(1)[0])
                        for s in np.random.SeedSequence(seed).spawn(3))
    return TrialSeeds(ds=ds, feat=feat, effect=effect)
