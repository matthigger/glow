"""What benchmark state to mirror to S3, as (local_dir, key_prefix) pairs.

s3.py is the generic file mover; this is the policy: which of the benchmark's
content-addressed trees to ship, and where each lands under the run's S3
prefix. The driver and worker both build their sync pairs here so the layout
agrees.

The records are keyed by a stable, machine-independent S3 prefix:

  - records -> {prefix}/records   (the per-hash <hash>.json files)

They are the only tree mirrored both ways: a worker uploads the records its
cell produces, and the driver pulls them to build the CSVs. Nothing else is
synced -- a worker runs one whole cell and shares no cache with another, and
the heavy exp caches (data_factory_wgn / data_factory_hcp / effect_factory)
rebuild on the worker (a WGN seed draw, or an HCP nifti load from the staged
data below), cheaper than shipping tens of MB.

HCP reference data (the npy bundle, pulled one way): a brain mask + one
float32 (num_img, num_vox) array per feature + small meta (see hcp.py). It is
staged to S3 once with infra.stage_hcp (hcp_bundle_pair, the whole dir), and an
HCP worker pulls only the files its cell needs -- the shared mask / affine /
meta plus its hcp_feats arrays (hcp_bundle_keys) -- into hcp.bundle_dir() so
data_factory_hcp builds from it with no niftis and no DUA prompt. Per-feature,
so a b=1 cell pulls one ~100 MB array, not the whole ~600 MB panel. It is never
produced by a worker, so it is pulled but never pushed.
"""

from pathlib import Path
from typing import List, Tuple

from glow._extra.benchmark import hcp
from glow._extra.benchmark.file import get_path_records

from .config import s3_key

Pair = Tuple[Path, str]

# S3 key prefix (under the run prefix) the HCP npy bundle is mirrored to.
HCP_BUNDLE_PREFIX = 'hcp_bundle'


def records_pair(prefix: str) -> Pair:
    """The (local records dir, S3 key prefix) pair for the per-hash records.

    The one tree mirrored both ways: a worker pushes the records its cell
    produces, and the driver pulls them to build the CSVs.

    Args:
        prefix (str): the run's s3_prefix.

    Returns:
        (local_dir, key_prefix): the local records dir and its S3 key prefix.
    """
    return get_path_records(), s3_key(prefix, 'records')


def hcp_bundle_pair(prefix: str) -> Pair:
    """The (local bundle dir, S3 key prefix) for staging the whole npy bundle.

    Used by infra.stage_hcp to upload the entire bundle once (upload_dir).
    Workers pull selectively instead (hcp_bundle_keys); the bundle is never
    produced by a worker, so this is never pushed up by one.

    Args:
        prefix (str): the run's s3_prefix.

    Returns:
        (local_dir, key_prefix): hcp.bundle_dir() and its S3 key prefix.
    """
    return hcp.bundle_dir(), s3_key(prefix, HCP_BUNDLE_PREFIX)


def hcp_bundle_keys(prefix: str, hcp_feats) -> List[Tuple[str, Path]]:
    """The (s3_key, local_path) pairs an HCP cell needs from the staged bundle.

    The shared mask / affine / meta plus one per-feature array per requested
    feat -- so a worker pulls only its cell's features (s3.download_each), not
    the whole panel. The keys mirror hcp_bundle_pair's upload layout, so a
    staged bundle and these keys line up.

    Args:
        prefix (str): the run's s3_prefix.
        hcp_feats (iterable[str]): the cell's features (its kwargs_data).

    Returns:
        list[(s3_key, local_path)]: the bundle files to pull, in pull order.
    """
    base = s3_key(prefix, HCP_BUNDLE_PREFIX)
    pairs = [
        (s3_key(base, 'mask.npy'), hcp.bundle_mask_path()),
        (s3_key(base, 'affine.npy'), hcp.bundle_affine_path()),
        (s3_key(base, 'meta.json'), hcp.bundle_meta_path()),
    ]
    for feat in hcp_feats:
        pairs.append(
            (s3_key(base, 'feat', f'{feat}.npy'), hcp.bundle_feat_path(feat)))
    return pairs
