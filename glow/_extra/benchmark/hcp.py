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
