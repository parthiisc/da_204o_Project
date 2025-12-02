"""Small helper to download model files listed in `models_manifest.json`.

Design:
- The downloader looks for `models_manifest.json` at the repo root and maps filenames -> URLs.
- Provides `download_file(name, dest_path)` and `try_download_all(dest_dir)` helpers.

This module is intentionally minimal and fails gracefully when no manifest is present.
"""
from pathlib import Path
import json
import logging
import shutil

logger = logging.getLogger(__name__)

try:
    import requests
except Exception:
    requests = None


_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _ROOT / "models_manifest.json"


def load_manifest():
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        with _MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load manifest %s: %s", _MANIFEST_PATH, e)
        return {}


def download_file(name: str, dest_path: str) -> bool:
    """Download a file specified by `name` in the manifest to `dest_path`.

    Returns True on success, False on failure or when no manifest entry exists.
    """
    manifest = load_manifest()
    url = manifest.get(name)
    if not url:
        logger.info("No URL for %s found in manifest", name)
        return False

    if requests is None:
        logger.warning("`requests` not available; cannot download %s", name)
        return False

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            # Write to a temp file then move to final path
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            shutil.move(str(tmp), str(dest))
        logger.info("Downloaded %s -> %s", url, dest)
        return True
    except Exception as e:
        logger.warning("Failed to download %s from %s: %s", name, url, e)
        return False


def try_download_all(dest_dir: str):
    """Attempt to download all files listed in the manifest into `dest_dir`.

    Returns list of filenames successfully downloaded.
    """
    manifest = load_manifest()
    if not manifest:
        logger.info("No manifest found; nothing to download")
        return []

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name in manifest.keys():
        dest = dest_dir / name
        if dest.exists():
            logger.info("File already exists, skipping download: %s", dest)
            continue
        ok = download_file(name, str(dest))
        if ok:
            downloaded.append(name)
    return downloaded
