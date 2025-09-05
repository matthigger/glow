import hashlib
import zipfile
from pathlib import Path

from tqdm import tqdm


def md5sum(path: Path, blocksize: int = 65536) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()


def check_and_unzip(folder: Path, delete: bool = False,
                    skip_existing: bool = False):
    """
    Check md5 sums and unzip all zips in a folder.

    Args:
        folder (Path): Path to the folder containing .zip and .md5 files.
        delete (bool): If True, delete the zip after successful extraction.
        skip_existing (bool): If True, skip extraction if target folder already exists.
    """
    folder = Path(folder)

    file_list = list(folder.glob("*.zip"))
    for zip_path in tqdm(file_list):
        base = zip_path.stem
        md5_path = zip_path.with_suffix('.zip.md5')

        target_dir = folder / base
        if skip_existing and target_dir.exists():
            print(f"[SKIP] {zip_path.name}: target folder exists")
            continue

        # Check md5 sum if file exists
        if md5_path.exists():
            expected = md5_path.read_text().strip().split()[0]
            actual = md5sum(zip_path)
            if actual != expected:
                print(f"[FAIL] {zip_path.name}: md5 mismatch")
                continue
            else:
                print(f"[OK]   {zip_path.name}: md5 verified")
        else:
            print(f"[WARN] No md5 file for {zip_path.name}")

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        print(f"[DONE] Extracted {zip_path.name} to {target_dir}")

        if delete and target_dir.exists():
            zip_path.unlink()
            print(f"[DEL]  Removed {zip_path.name}")


if __name__ == '__main__':
    check_and_unzip(folder='/home/matt/data/hcp100_aug25',
                    delete=False,
                    skip_existing=True)
