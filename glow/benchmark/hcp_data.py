"""HCP data download and path management."""

from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from platformdirs import user_data_dir

# default storage location (platform-appropriate)
_base = Path(user_data_dir('glow', 'glow_author'))
DEFAULT_HCP_PATH = _base / 'hcp100_aug25_registered'

# Zenodo direct-download URL for the zip archive (~441 MB)
_ZENODO_URL = ('https://zenodo.org/records/17306498/files/'
               'hcp100_aug25_registered.zip?download=1')

# where users must agree to the data use terms
_DATA_USE_TERMS_URL = 'https://balsa.wustl.edu/project?project=HCP_YA'


def _prompt_data_use_terms():
    """prompt user to confirm agreement to HCP data use terms."""
    print()
    print('=' * 70)
    print('HCP DATA USE TERMS')
    print('=' * 70)
    print()
    print('This dataset is derived from the Human Connectome Project (HCP)')
    print('and is subject to the WU-Minn HCP Open Access Data Use Terms.')
    print()
    print('See also:')
    print('  https://zenodo.org/records/17306498')
    print('  https://github.com/matthigger/glow_paper_hcp_data_prep')
    print()
    print('You must agree to the terms here before downloading:')
    print(f'  {_DATA_USE_TERMS_URL}')
    print()
    response = input('Type "I have agreed to data use terms" to proceed: ')
    if response.strip() != 'I have agreed to data use terms':
        raise SystemExit('Download cancelled: data use terms not confirmed.')


def _download_progress(block_num, block_size, total_size):
    """progress callback for urllib.request.urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb_done = downloaded / 1024 ** 2
        mb_total = total_size / 1024 ** 2
        print(f'\r  downloading: {mb_done:.0f}/{mb_total:.0f} MB '
              f'({pct:.0f}%)', end='', flush=True)
    else:
        mb_done = downloaded / 1024 ** 2
        print(f'\r  downloading: {mb_done:.0f} MB', end='', flush=True)


def download_hcp_data(dest=None):
    """download and extract the HCP dataset from Zenodo.

    Args:
        dest: target directory (defaults to DEFAULT_HCP_PATH)

    Returns:
        Path to the extracted dataset directory
    """
    if dest is None:
        dest = DEFAULT_HCP_PATH
    dest = Path(dest)

    _prompt_data_use_terms()

    print(f'\n  destination: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)

    # download to a temporary file, then extract
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / 'hcp100_aug25_registered.zip'
        print(f'  source: {_ZENODO_URL}')
        urllib.request.urlretrieve(_ZENODO_URL, zip_path,
                                   reporthook=_download_progress)
        print()  # newline after progress

        print('  extracting ...')
        with zipfile.ZipFile(zip_path) as zf:
            # extract to temp dir first, then move into place
            extract_dir = Path(tmp) / 'extract'
            zf.extractall(extract_dir)

            # the zip may contain a top-level directory; detect it
            top_items = list(extract_dir.iterdir())
            if (len(top_items) == 1 and top_items[0].is_dir()):
                src = top_items[0]
            else:
                src = extract_dir

            # move to final destination
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))

    print(f'  done: {dest}')
    return dest


def get_hcp_path(path=None):
    """return the HCP data path, downloading if necessary.

    Args:
        path: explicit path override. if None, uses DEFAULT_HCP_PATH
            and downloads when missing.

    Returns:
        Path to the HCP dataset directory

    Raises:
        FileNotFoundError: if an explicit path does not exist
    """
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f'HCP data not found at explicit path: {path}')
        return path

    # default path - download if missing
    if DEFAULT_HCP_PATH.exists():
        return DEFAULT_HCP_PATH

    print(f'HCP data not found at: {DEFAULT_HCP_PATH}')
    return download_hcp_data(DEFAULT_HCP_PATH)
