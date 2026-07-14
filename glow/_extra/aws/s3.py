"""Sync the benchmark's content-addressed state to and from S3.

The whole AWS path rests on one property of the rebuilt benchmark layer: its
on-disk state -- the joblib.Memory cache and the per-hash <hash>.json records
-- is content-addressed. Each artifact's path is a pure function of the call
that produced it (joblib's args hash; see glow._extra.benchmark.recorder), so
the same call writes the same file on any machine. Two workers computing
different cells therefore write disjoint files, and "share the cache" reduces
to copying files: no locking, no coordination, no merge.

So this module is a thin file mover, not a cache backend. It mirrors local
directories under glow's per-user data dir to an S3 prefix and back:

  - upload_dir / download_prefix copy a directory tree, skipping objects that
    already exist (a HEAD probe up, a local-stat down) -- so a re-sync only
    moves the new artifacts, and the content-addressing makes "already there"
    safe to skip.
  - BackgroundUploader pushes a set of directories on an interval while the
    worker computes, so finished records land at the driver as the worker goes
    (and a Spot-interrupted worker has already shipped the records it
    finished), each a small <hash>.json.

What to sync is the caller's policy (see the driver / worker): the records dir
(tiny, and it carries the provenance DAG the CSVs are built from). The compute
caches are not shipped -- a worker runs one whole cell and shares no cache with
another, and the exp caches (WGN data_factory) rebuild on the worker rather
than ship.

Upload safety: joblib writes a cache entry to a temp dir and renames it into
place, and the recorder writes each <hash>.json via temp-file + os.replace, so
a file visible under these trees is already complete -- the uploader never
races a half-written artifact.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

UPLOAD_THREADS = 16
# default seconds between background upload sweeps while a worker computes.
SYNC_INTERVAL_SEC = 60


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Split an s3://bucket/key URI into (bucket, key).

    Args:
        uri (str): an s3:// URI.

    Returns:
        (bucket, key): the bucket name and the (possibly empty) key.

    Raises:
        ValueError: uri does not start with s3://.
    """
    if not uri.startswith('s3://'):
        raise ValueError(f'not an s3:// URI: {uri!r}')
    rest = uri[len('s3://'):]
    bucket, _, key = rest.partition('/')
    return bucket, key


def _object_exists(s3, bucket: str, key: str) -> bool:
    """Whether key exists in bucket (a head_object probe, 404 -> False)."""
    from botocore.exceptions import ClientError
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] in ('404', 'NoSuchKey', 'NotFound'):
            return False
        raise


def _iter_files(local_dir: Path):
    """Yield every file under local_dir (recursively), or nothing if absent."""
    if not local_dir.is_dir():
        return
    for p in local_dir.rglob('*'):
        if p.is_file():
            yield p


def _log_upload(p: Path, key: str) -> None:
    """Print one line per uploaded object for the worker's CloudWatch stream.

    Tags a <hash>.json record with its 'function' so the log names what shipped
    (run_stat / effect_factory_single / ...) and when; a cache file logs by key
    alone. Diagnostic only: it lets a Spot-killed attempt reveal what it never
    got to upload -- pair it with the recorder's [record] write log (see
    glow._extra.benchmark.recorder) to tell lost from never-written.
    """
    label = key
    if p.suffix == '.json':
        try:
            label = f"{json.loads(p.read_text()).get('function', '?')} {key}"
        except Exception:
            pass
    print(f'[upload] {label}', flush=True)


def upload_dir(s3, bucket: str, local_dir, key_prefix: str, *,
               skip_existing: bool = True, threads: int = UPLOAD_THREADS,
               seen: set = None) -> int:
    """Upload every file under local_dir to bucket under key_prefix.

    A file at local_dir/<rel> becomes the object key_prefix/<rel>. Existing
    objects are skipped (a head_object probe) when skip_existing, so a re-sync
    only ships new artifacts -- safe because the tree is content-addressed (an
    object that exists holds the same bytes). A missing local_dir uploads
    nothing.

    Args:
        s3: boto3 S3 client.
        bucket (str): destination bucket.
        local_dir (str | Path): the directory tree to upload.
        key_prefix (str): S3 key prefix the tree is mirrored under.
        skip_existing (bool): head_object-probe and skip objects already there.
        threads (int): parallel upload workers.
        seen (set | None): if given, a set of already-uploaded local paths
            (as str) to skip without a probe and add to -- lets a repeated
            call (the background uploader) avoid re-probing files it shipped.

    Returns:
        n_uploaded (int): the number of files actually uploaded this call.
    """
    local_dir = Path(local_dir)
    files = [p for p in _iter_files(local_dir)
             if seen is None or str(p) not in seen]
    if not files:
        return 0

    def _put_one(p: Path) -> bool:
        key = f'{key_prefix}/{p.relative_to(local_dir).as_posix()}'
        if skip_existing and _object_exists(s3, bucket, key):
            return False
        s3.upload_file(str(p), bucket, key)
        _log_upload(p, key)
        return True

    n = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for p, uploaded in zip(files, pool.map(_put_one, files)):
            if seen is not None:
                seen.add(str(p))
            n += int(uploaded)
    return n


def download_prefix(s3, bucket: str, key_prefix: str, local_dir, *,
                    skip_existing: bool = True,
                    threads: int = UPLOAD_THREADS) -> int:
    """Download every object under key_prefix into local_dir.

    The object key_prefix/<rel> becomes the file local_dir/<rel> (parent dirs
    created). Existing local files are skipped (a stat) when skip_existing, so
    pulling onto a warm tree only fetches the missing artifacts.

    Args:
        s3: boto3 S3 client.
        bucket (str): source bucket.
        key_prefix (str): the S3 prefix to pull (no trailing slash needed).
        local_dir (str | Path): the directory the tree is written under.
        skip_existing (bool): skip objects whose local file already exists.
        threads (int): parallel download workers.

    Returns:
        n_downloaded (int): the number of objects actually downloaded.
    """
    local_dir = Path(local_dir)
    prefix = key_prefix.rstrip('/') + '/'

    keys: List[str] = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            keys.append(obj['Key'])

    def _get_one(key: str) -> bool:
        rel = key[len(prefix):]
        if not rel:
            return False
        dest = local_dir / rel
        if skip_existing and dest.exists():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        return True

    if not keys:
        return 0
    n = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for downloaded in pool.map(_get_one, keys):
            n += int(downloaded)
    return n


def download_each(s3, bucket: str, pairs, *, skip_existing: bool = True,
                  threads: int = UPLOAD_THREADS) -> int:
    """Download an explicit list of (s3_key, local_path) objects.

    The selective counterpart to download_prefix: the caller names exactly
    which keys to fetch and where each lands, so a worker pulls just the
    bundle files its cell needs rather than a whole prefix (see
    glow._extra.aws.sync.hcp_bundle_keys). Existing local files are skipped
    when skip_existing; parent dirs are created.

    Args:
        s3: boto3 S3 client.
        bucket (str): source bucket.
        pairs (iterable[(str, str | Path)]): (s3_key, local_path) to fetch.
        skip_existing (bool): skip a pair whose local file already exists.
        threads (int): parallel download workers.

    Returns:
        n_downloaded (int): the number of objects actually downloaded.
    """
    pairs = list(pairs)

    def _get_one(pair) -> bool:
        key, dest = pair
        dest = Path(dest)
        if skip_existing and dest.exists():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        return True

    if not pairs:
        return 0
    n = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for downloaded in pool.map(_get_one, pairs):
            n += int(downloaded)
    return n


class BackgroundUploader:
    """Periodically upload a set of (local_dir, key_prefix) pairs in a thread.

    Ships finished records off the worker while it keeps computing, so the
    driver lands results as workers finish them (and a Spot-interrupted attempt
    has already persisted the records it finished). The compute thread is never
    blocked: S3 PUT is I/O-bound and releases the GIL, and the sweep only walks
    for new files. A per-pair seen set means each file is probed/uploaded once
    across sweeps. flush() forces an immediate sweep (called on normal exit and
    on the SIGTERM Batch sends ~2 min before a Spot reclaim).

    Attributes:
        pairs (list[tuple]): the (local_dir, key_prefix) trees to mirror up.
        interval (float): seconds between sweeps.
    """

    def __init__(self, s3, bucket: str, pairs, *,
                 interval: float = SYNC_INTERVAL_SEC):
        self._s3 = s3
        self._bucket = bucket
        self.pairs = list(pairs)
        self.interval = interval
        self._seen = {i: set() for i in range(len(self.pairs))}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sweep(self) -> int:
        """Upload any new files across all pairs once; return the count."""
        # serialise sweeps (the loop and an explicit flush) so a file is not
        # uploaded twice and the seen sets stay consistent.
        with self._lock:
            n = 0
            for i, (local_dir, key_prefix) in enumerate(self.pairs):
                n += upload_dir(self._s3, self._bucket, local_dir, key_prefix,
                                seen=self._seen[i])
            return n

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sweep()

    def start(self) -> 'BackgroundUploader':
        """Start the background sweep thread; return self."""
        self._thread.start()
        return self

    def flush(self) -> int:
        """Run one sweep now (e.g. on exit); return files uploaded."""
        return self._sweep()

    def stop(self) -> int:
        """Stop the thread and run a final flush; return files uploaded."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        return self.flush()

    def __enter__(self) -> 'BackgroundUploader':
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
