"""Fetch individual pickled experiments from one published Zenodo record.

The web viewer lets a user browse the files of a single, pre-configured
Zenodo record and load one pickle into an interactive Dash view. The
record id is an allowlist: only files belonging to that record are ever
downloaded and unpickled, so the viewer never deserialises bytes from an
untrusted source. This is the whole point of fetching by record id rather
than accepting an upload -- pickle.load on attacker-supplied bytes is
arbitrary code execution (see glow/_extra/viewer/web/README.md).

Zenodo's public read API needs no authentication. A GET of
{api_base}/records/{record_id} returns a JSON record whose files live
either in a top-level files list (current zenodo.org) or, on some
InvenioRDM deployments, under files.entries. Each entry carries key
(filename), size (bytes), checksum ("md5:..."), and a links.self content
URL. Requesting a concept (version-independent) id redirects to the latest
version; pin a version-specific id for a stable demo set.

Set GLOW_ZENODO_API_BASE to https://sandbox.zenodo.org/api to test
against a Zenodo Sandbox deposit before a real DOI exists.
"""

import hashlib
import json
import os
import pathlib
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import platformdirs


DEFAULT_API_BASE = 'https://zenodo.org/api'
# Cap one download so a giant (e.g. full-brain runtime-cache) object cannot
# exhaust a small Space's RAM/disk. 64 MB clears a 25k-voxel/num_img=100
# analysis (~13 MB) with headroom.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_HTTP_TIMEOUT = 30
_CHUNK = 1 << 16


def api_base() -> str:
    """Return the Zenodo API base, honouring GLOW_ZENODO_API_BASE."""
    return os.environ.get('GLOW_ZENODO_API_BASE', DEFAULT_API_BASE).rstrip('/')


def cache_dir() -> pathlib.Path:
    """Return the download cache dir (GLOW_ZENODO_CACHE_DIR or default)."""
    env = os.environ.get('GLOW_ZENODO_CACHE_DIR')
    base = pathlib.Path(env) if env else (
        pathlib.Path(platformdirs.user_cache_dir('glow')) / 'zenodo')
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass(frozen=True)
class ZenodoFile:
    """One downloadable file in a Zenodo record.

    Attributes:
        key (str): filename as published in the record.
        size (int): size in bytes, as reported by the record metadata.
        md5 (str): expected MD5 hex digest (the "md5:" prefix stripped).
        url (str): direct content URL (the entry's links.self).
    """

    key: str
    size: int
    md5: str
    url: str


def _get_json(url: str, *, timeout: int = _HTTP_TIMEOUT) -> dict:
    """GET a URL and parse the response body as JSON."""
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _normalize_files(record: dict) -> List[ZenodoFile]:
    """Extract ZenodoFile entries from a record's files, both API shapes.

    Handles the top-level files list (current zenodo.org) and the
    InvenioRDM files.entries dict/list, ignoring entries that lack a usable
    content link.

    Args:
        record (dict): parsed /api/records/<id> JSON.

    Returns:
        files (list): ZenodoFile, in the order the record lists them.
    """
    raw = record.get('files')
    if isinstance(raw, dict):
        entries = raw.get('entries', [])
        raw = list(entries.values()) if isinstance(entries, dict) else entries
    if not isinstance(raw, list):
        return []

    out: List[ZenodoFile] = []
    for entry in raw:
        links = entry.get('links') or {}
        url = (links.get('self') or links.get('content')
               or links.get('download'))
        key = entry.get('key')
        if not url or not key:
            continue
        checksum = (entry.get('checksum') or '')
        md5 = checksum.split(':', 1)[1] if checksum.startswith('md5:') else ''
        out.append(ZenodoFile(
            key=key, size=int(entry.get('size') or 0), md5=md5, url=url))
    return out


def list_record_files(record_id: str, *,
                       base: Optional[str] = None) -> List[ZenodoFile]:
    """List the downloadable files of one public Zenodo record.

    Args:
        record_id (str): Zenodo record id (pin a version-specific id; a
            concept id redirects to the latest version).
        base (str | None): API base override; defaults to api_base().

    Returns:
        files (list): ZenodoFile for every file with a content link.
    """
    base = (base or api_base()).rstrip('/')
    record = _get_json(f'{base}/records/{record_id}')
    return _normalize_files(record)


def fetch_file(zfile: ZenodoFile, *, max_bytes: int = DEFAULT_MAX_BYTES,
               dest_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Download one record file to the cache, size-capped and MD5-verified.

    Reuses a cached copy whose MD5 already matches, so repeat loads of the
    same file skip the network. The download aborts before exhausting
    memory or disk if either the advertised size or the streamed byte
    count exceeds max_bytes.

    Args:
        zfile (ZenodoFile): file to fetch (carries its url, size, md5).
        max_bytes (int): hard ceiling on the downloaded size, in bytes.
        dest_dir (pathlib.Path | None): cache directory; defaults to
            cache_dir().

    Returns:
        path (pathlib.Path): local path to the downloaded file.

    Raises:
        ValueError: if the file exceeds max_bytes or the MD5 mismatches.
    """
    if zfile.size and zfile.size > max_bytes:
        raise ValueError(
            f'{zfile.key} is {zfile.size} bytes, over the '
            f'{max_bytes}-byte cap')

    dest_dir = dest_dir or cache_dir()
    safe = zfile.key.replace('/', '_')
    dest = dest_dir / safe

    if dest.exists() and zfile.md5 and _md5(dest) == zfile.md5:
        return dest

    digest = hashlib.md5()
    total = 0
    tmp = dest.with_suffix(dest.suffix + '.part')
    req = urllib.request.Request(zfile.url)
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp, \
            open(tmp, 'wb') as f:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                tmp.unlink(missing_ok=True)
                raise ValueError(
                    f'{zfile.key} exceeded the {max_bytes}-byte cap '
                    f'mid-download')
            digest.update(chunk)
            f.write(chunk)

    if zfile.md5 and digest.hexdigest() != zfile.md5:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f'{zfile.key} MD5 mismatch: expected {zfile.md5}, '
            f'got {digest.hexdigest()}')

    tmp.replace(dest)
    return dest


def _md5(path: pathlib.Path) -> str:
    """Return the MD5 hex digest of a file, read in chunks."""
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()
