"""Download and load glow's reference HCP-YA diffusion-microstructure data.

Six per-subject diffusion-microstructure maps (DKI fa/md/mk, NODDI
icvf/isovf/od) for the 100 unrelated WU-Minn HCP-Young-Adult subjects, plus
a single brain mask, in MNI152NLin2009cAsym space at 2 mm, published as a
single Zenodo archive (record 20736221).

ensure_hcp_data returns a local directory of the maps, downloading and
extracting the archive on first use.  Because the maps derive from HCP
open-access data, the first download is gated on the user accepting the
WU-Minn HCP Open Access Data Use Terms; once the data is on disk it loads
with no prompt.

The archive also ships a brain mask (MASK_GLOB); DataSourceHCP passes it
to from_paths as the analysis support, so the maps are not used to infer
it -- the every-image-nonzero heuristic wrongly drops in-brain voxels
where NODDI isovf is legitimately zero.
"""

import hashlib
import json
import os
import pathlib
import shutil
import sys
import urllib.request
import zipfile

import numpy as np

from glow._extra.benchmark import file


ZENODO_RECORD_ID = '20736221'
DUA_URL = ('https://www.humanconnectome.org/study/hcp-young-adult/document/'
           'wu-minn-hcp-consortium-open-access-data-use-terms')
DUA_ACK_PHRASE = 'I agree to the WU-Minn HCP Open Access Data Use Terms'

# feature -> recursive glob over the extracted archive. The QSIRecon
# dwimap filenames carry the DKI / NODDI model and the parameter, so each
# glob selects exactly one map per subject however deep the archive nests.
IMG_GLOB_DICT = {
    'fa':    '**/*model-dki_param-fa_dwimap.nii.gz',
    'md':    '**/*model-dki_param-md_dwimap.nii.gz',
    'mk':    '**/*model-dki_param-mk_dwimap.nii.gz',
    'icvf':  '**/*model-noddi_param-icvf_dwimap.nii.gz',
    'isovf': '**/*model-noddi_param-isovf_dwimap.nii.gz',
    'od':    '**/*model-noddi_param-od_dwimap.nii.gz',
}

# The full six-feature panel, in canonical order (DKI then NODDI).
HCP_FEATS = tuple(IMG_GLOB_DICT)

# Recursive glob for the single brain-mask map shipped with the archive.
# It shares the maps' MNI152NLin2009cAsym 2 mm grid and marks the in-brain
# voxels used as the analysis support (see load_brain_mask).
MASK_GLOB = '**/brain_mask_space-*.nii.gz'

# Subject id repeats in the sub-<id>/ directory and the filename
# (from_search dedupes); the sub- prefix avoids matching the digits in the
# MNI152NLin2009cAsym space tag.
SBJ_REGEX = r'sub-(\d+)'

_TIMEOUT = 60
_CHUNK = 1 << 20


def data_dir() -> pathlib.Path:
    """Return the local HCP dataset directory."""
    return file.get_path_data() / 'hcp100_dki_noddi_mni'


def is_present(folder: pathlib.Path = None) -> bool:
    """Return whether the six feature maps and brain mask are under folder.

    A partial / aborted extraction (any feature or the mask missing) reads
    as absent, so it is re-downloaded rather than silently half-loaded.
    Requiring the mask also forces a re-download of a pre-mask archive.
    """
    folder = folder or data_dir()
    globs = list(IMG_GLOB_DICT.values()) + [MASK_GLOB]
    return folder.is_dir() and all(
        next(folder.glob(glob), None) is not None for glob in globs)


def ensure_hcp_data() -> pathlib.Path:
    """Return the local HCP dataset dir, downloading it on first use.

    If the maps are already present the directory is returned with no
    prompt.  Otherwise the user must accept the HCP Data Use Terms before
    the Zenodo archive is downloaded, MD5-verified, and extracted.

    Raises:
        RuntimeError: if the terms are declined, or the download did not
            yield the expected maps.
    """
    folder = data_dir()
    if is_present(folder):
        return folder
    if not _accept_dua():
        raise RuntimeError(
            'HCP data is absent and its Data Use Terms were not accepted; '
            're-run interactively and type the acknowledgment. ' + DUA_URL)
    _download_and_extract(folder)
    if not is_present(folder):
        raise RuntimeError(f'HCP download yielded no maps under {folder}')
    return folder


# ---------- npy bundle (the per-feature, AWS-shippable representation) -------
# A compact stand-in for the niftis: a brain mask plus one float32
# (num_img, num_vox) array per feature plus a small meta (subject ids +
# affine). It holds exactly the arrays from_search builds -- y[feat] is
# arr.get_fdata(float32)[mask] in the mask's C-order, the same for every
# feature -- so build_exp_img_from_bundle reconstructs a byte-identical
# ExperimentImageOnly (np.save round-trips dtype + values; data_factory's
# _with_canonical_y normalises layout), and the experiment hashes identically
# whether built from the niftis or the bundle.
#
# It is the single HCP build path: data_factory_hcp builds from the bundle
# both locally (converting from the niftis on first use) and on an AWS worker
# (where the bundle is pre-staged from S3, so no niftis and no DUA prompt are
# needed -- see glow._extra.aws). Per-feature files let a worker pull only the
# features its cell uses.

BUNDLE_DIRNAME = 'hcp100_dki_noddi_npy'


def bundle_dir() -> pathlib.Path:
    """Return the local npy-bundle directory (sibling of the niftis)."""
    return file.get_path_data() / BUNDLE_DIRNAME


def bundle_feat_path(feat: str) -> pathlib.Path:
    """Path to one feature's (num_img, num_vox) float32 array in the bundle."""
    return bundle_dir() / 'feat' / f'{feat}.npy'


def bundle_mask_path() -> pathlib.Path:
    """Path to the bundle's boolean brain-mask array (full-volume shape)."""
    return bundle_dir() / 'mask.npy'


def bundle_affine_path() -> pathlib.Path:
    """Path to the bundle's affine array (the maps' world transform)."""
    return bundle_dir() / 'affine.npy'


def bundle_meta_path() -> pathlib.Path:
    """Path to the bundle's small JSON meta (the subject id list)."""
    return bundle_dir() / 'meta.json'


def is_bundle_present(feats=HCP_FEATS) -> bool:
    """Return whether the mask / affine / meta and each feat array are present.

    Checks only the requested feats (plus the shared mask / affine / meta), so
    a worker that staged a subset reads as present for that subset.
    """
    base = (bundle_mask_path().exists() and bundle_affine_path().exists()
            and bundle_meta_path().exists())
    return base and all(bundle_feat_path(f).exists() for f in feats)


def ensure_hcp_bundle(feats=HCP_FEATS) -> pathlib.Path:
    """Return the bundle dir, converting it from the niftis on first use.

    Idempotent: if the requested feats (and the shared mask / affine / meta)
    are present the dir is returned untouched -- so on an AWS worker, where the
    bundle is pre-staged from S3, this never reaches the niftis or the DUA
    gate. Otherwise the niftis are loaded once (ensure_hcp_data -> from_search
    over the full panel) and dumped as the bundle.

    Raises:
        RuntimeError: the conversion did not yield the requested feats.
    """
    if is_bundle_present(feats):
        return bundle_dir()
    from glow.experiment import ExperimentImageOnly

    folder = ensure_hcp_data()
    exp_img = ExperimentImageOnly.from_search(
        folder=folder, sbj_regex=SBJ_REGEX, img_glob_dict=IMG_GLOB_DICT,
        mask=next(folder.glob(MASK_GLOB)))
    _dump_bundle(exp_img)
    if not is_bundle_present(feats):
        raise RuntimeError(
            f'HCP bundle build did not yield all of {list(feats)}')
    return bundle_dir()


def _dump_bundle(exp_img) -> None:
    """Write an ExperimentImageOnly out as the per-feature npy bundle.

    Dumps the boolean mask (mask_idx > -1), the affine, the subject ids, and
    one float32 (num_img, num_vox) array per feature -- exactly the arrays the
    glue reloads (see build_exp_img_from_bundle).
    """
    (bundle_dir() / 'feat').mkdir(parents=True, exist_ok=True)
    np.save(bundle_mask_path(), exp_img.mask_idx > -1)
    np.save(bundle_affine_path(), np.asarray(exp_img.meta['affine']))
    bundle_meta_path().write_text(
        json.dumps({'subjects': exp_img.meta['subjects']}))
    for feat_idx, feat in enumerate(exp_img.meta['features']):
        np.save(bundle_feat_path(feat), exp_img.y[feat_idx])


def build_exp_img_from_bundle(feats):
    """Glue the bundle into an ExperimentImageOnly for a feature subset.

    The single HCP build path (see the module section above): ensure the
    bundle is present (locally convert from the niftis; on a worker it is
    pre-staged), then load the shared mask / affine / subjects and stack the
    requested feats into y in their given order. The result is byte-identical
    to ExperimentImageOnly.from_search for the same feats, so it hashes the
    same (verified in the tests).

    Args:
        feats (tuple[str]): the features to load, in order; sets b and the
            meta 'features' order.

    Returns:
        ExperimentImageOnly with y of shape (b, num_img, num_vox) float32.
    """
    import glow.mask
    from glow.experiment import ExperimentImageOnly

    feats = tuple(feats)
    ensure_hcp_bundle(feats)

    mask_idx = glow.mask.get_mask_idx(np.load(bundle_mask_path()))
    affine = np.load(bundle_affine_path())
    subjects = json.loads(bundle_meta_path().read_text())['subjects']

    y0 = np.load(bundle_feat_path(feats[0]))
    y = np.empty((len(feats), *y0.shape), dtype=y0.dtype)
    y[0] = y0
    for feat_idx, feat in enumerate(feats[1:], start=1):
        y[feat_idx] = np.load(bundle_feat_path(feat))

    # mirror from_paths' meta exactly (key order included) so the hash matches
    meta = {'subjects': subjects, 'features': list(feats), 'affine': affine}
    return ExperimentImageOnly(y=y, mask_idx=mask_idx, meta=meta)


def _accept_dua() -> bool:
    """Prompt for and verify acceptance of the HCP Data Use Terms.

    Returns False without prompting when stdin is not a TTY, so a headless
    caller gets a clear error rather than hanging on input.
    """
    if not (sys.stdin and sys.stdin.isatty()):
        return False
    print(f"\nglow's HCP dataset (~1.1 GB) derives from Human Connectome "
          f"Project open-access data, shared under the WU-Minn HCP Open "
          f"Access Data Use Terms (which permit redistribution under the "
          f"same terms):\n    {DUA_URL}\n"
          f"To download it, type the following line exactly:\n"
          f"    {DUA_ACK_PHRASE}")
    try:
        reply = input('> ')
    except EOFError:
        return False
    return reply.strip().rstrip('.').strip().casefold() \
        == DUA_ACK_PHRASE.casefold()


def _archive_url():
    """Resolve the Zenodo record's archive to (url, md5).

    Returns:
        url (str): direct content URL of the archive file.
        md5 (str): its expected MD5 hex digest ('' if the record omits it).

    Raises:
        RuntimeError: if the record lists no downloadable file.
    """
    req = urllib.request.Request(
        f'https://zenodo.org/api/records/{ZENODO_RECORD_ID}',
        headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        files = json.loads(resp.read().decode('utf-8')).get('files') or []
    if not files:
        raise RuntimeError(
            f'Zenodo record {ZENODO_RECORD_ID} lists no downloadable file')
    entry = files[0]
    md5 = (entry.get('checksum') or '').removeprefix('md5:')
    return entry['links']['self'], md5


def _download_and_extract(folder: pathlib.Path) -> None:
    """Download the Zenodo archive, MD5-verify, and extract it into folder.

    Streams to a sibling .zip with a progress line, extracts into a
    .partial staging directory, and renames it into place so an interrupted
    run never leaves a half-populated folder that is_present would accept.
    """
    url, md5 = _archive_url()
    folder.parent.mkdir(parents=True, exist_ok=True)
    zip_path = folder.with_suffix('.zip')

    digest = hashlib.md5()
    done = 0
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp, \
            open(zip_path, 'wb') as f:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            done += len(chunk)
            digest.update(chunk)
            f.write(chunk)
            print(f'\r  downloading HCP data {done / 1e6:,.0f} MB',
                  end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if md5 and digest.hexdigest() != md5:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError('HCP archive MD5 mismatch')

    staging = folder.with_name(folder.name + '.partial')
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging)
    shutil.rmtree(folder, ignore_errors=True)
    os.replace(staging, folder)
    zip_path.unlink(missing_ok=True)
