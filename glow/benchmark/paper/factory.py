"""Shared builders for the paper benchmarks: scalar axes -> heavy objects.

The paper catalogue (config.py) keeps every trial axis as a plain
scalar/categorical in iter_kwargs (source, b, num_img, n_vox_eff,
effect_llr, seed, ...); the trial functions (run.py) turn those scalars
back into the heavy DataSource / Extenter objects here. Both config and
run import from this module, so it must not import either of them (no
import cycle).

The DataSource build is memoised on its scalar identity (_ds_factory is
lru_cached on source / b / num_img / feats / ds_seed), so every method /
effect_llr trial of one trial seed -- which derive_seeds maps to a single
(feats, ds_seed) -- reuses that seed's single ds.exp build. Distinct trial
seeds get distinct ds_seeds (an independent data realization each), so the
build is amortised within a seed rather than shared across the seed loop.
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

# Fallback DataSource seed for build_ds's ds_seed. The trial setups (run.py)
# now derive an independent per-seed ds_seed via derive_seeds, so each trial
# seed gets its own data realization; this fixed default only covers a direct
# build_ds call that wants one shared base experiment (e.g. an ad-hoc probe).
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
        ds_seed (int): seed for the X design + sphere crop. A per-trial
            ds_seed (from derive_seeds, what every trial setup passes) gives
            each trial seed an independent data realization; the fixed DS_SEED
            fallback gives one shared base experiment.

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
    directly. The X design + crop use ds_seed: the trial setups pass a per-trial
    ds_seed (from derive_seeds) so each trial seed is an independent data
    realization; it falls back to the shared DS_SEED (one base experiment) only
    when no ds_seed is given.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN only)
        seed (int): feature-subset seed (HCP); unused for WGN
        ds_seed (int): seed for the X design + crop; the setups pass a per-seed
            value (derive_seeds). Defaults to DS_SEED (shared base) as a fallback

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
    three mutually independent child seeds, so each trial seed is an independent
    data realization with the planted effect kept independent of it. Every trial
    setup (run._setup_trial / _setup_two_effect, and the min_size curve setup)
    derives its sub-seeds here; the DS_SEED-pinned shared base is no longer used.

    Args:
        seed (int): the trial seed (an iter_kwargs axis).

    Returns:
        TrialSeeds(ds, feat, effect): three mutually independent sub-seeds.
    """
    ds, feat, effect = (int(s.generate_state(1)[0])
                        for s in np.random.SeedSequence(seed).spawn(3))
    return TrialSeeds(ds=ds, feat=feat, effect=effect)


def build_ds_for_seed(source: str, *, b: int, num_img: int, seed: int):
    """Build a trial's DataSource on its own data realization; hand back the
    effect sub-seed.

    The one entry point every run.py setup builds through, so the "independent
    per-seed realization" convention lives in one place: the trial seed is split
    (derive_seeds) into mutually independent ds / feature / effect sub-seeds, the
    ds is built on the ds + feature sub-seeds (each trial seed its own draw), and
    the effect sub-seed is returned for planting -- kept independent of that
    draw. build_ds raises on an infeasible cell (e.g. HCP b > pool); the caller's
    recorder records and swallows it.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        seed (int): the trial seed (an iter_kwargs axis), split here

    Returns:
        ds: the DataSource for this trial (memoised on its identity; .exp is
            built once per identity and shared across the seed's trials)
        effect_seed (int): the effect sub-seed, independent of the ds draw
    """
    s = derive_seeds(seed)
    ds, _ = build_ds(source, b=b, num_img=num_img, seed=s.feat, ds_seed=s.ds)
    return ds, s.effect
