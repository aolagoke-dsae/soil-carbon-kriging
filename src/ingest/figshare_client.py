"""figshare ingestion: the Potash et al. (2023) field data, version 2, pinned.

Eight commercial U.S. Midwest fields: soil organic carbon concentration and
bulk density by depth segment at georeferenced locations, CC BY 4.0
(config/sources.csv). The article is read from the figshare REST API at its
version, never at "latest", so a new upload cannot change what this repository
analyses without a change to this file:

    https://api.figshare.com/v2/articles/23669304/versions/2

Same discipline and the same manifest as grid-carbon-intensity:
  * raw/ is append-only (src/ingest/raw_store.py). The article record and each
    file it lists are logical items, and every retrieval is a new file.
  * Every response is hashed and recorded with its URL, status, size and time.
  * A file is accepted only if its size and MD5 equal those the article states
    (figshare's computed_md5). Otherwise it goes to _failed/ and the pull fails.

The manifest records each file's download_url as the article lists it, not the
address it redirects to: that is a signed storage URL whose query string
carries a temporary credential.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

try:                        # run as a module from the repository root
    from src.ingest import http_get, raw_store
except ImportError:         # imported with src/ingest on sys.path (pytest)
    import http_get
    import raw_store

ARTICLE_ID = 23669304
VERSION = 2
API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{VERSION}"
RAW = Path("raw")
MANIFEST = RAW / "manifest.csv"


def _now() -> dt.datetime:
    """The retrieval time, in UTC (replaced in the tests)."""
    return dt.datetime.now(dt.timezone.utc)


def root() -> Path:
    """raw/FIGSHARE/23669304/v2/"""
    return RAW / "FIGSHARE" / str(ARTICLE_ID) / f"v{VERSION}"


def article_item() -> Path:
    """The article record's logical item, raw/FIGSHARE/23669304/v2/article.json"""
    return root() / "article.json"


def file_item(name: str) -> Path:
    """A file's logical item, e.g. raw/FIGSHARE/23669304/v2/files/locations.csv"""
    return root() / "files" / name


def usable_article(status: int, body: bytes) -> bool:
    """HTTP 200 and the article record at the pinned version, under a stated
    licence, listing files that each have a name, a download URL, an MD5 and
    a size. An error page or another version is not the record."""
    if status != 200:
        return False
    try:
        doc = json.loads(body)
    except ValueError:
        return False
    if not isinstance(doc, dict):
        return False
    if doc.get("id") != ARTICLE_ID or doc.get("version") != VERSION:
        return False
    if not (doc.get("license") or {}).get("name"):
        return False
    files = doc.get("files")
    keys = ("name", "download_url", "computed_md5", "size")
    return bool(files) and all(
        isinstance(f, dict) and all(f.get(k) for k in keys) for f in files)


def file_matches(meta: dict, body: bytes) -> bool:
    """The bytes are the file the article lists: same size, same MD5. MD5 is
    figshare's integrity check here, not a security measure."""
    return (len(body) == meta["size"]
            and hashlib.md5(body, usedforsecurity=False).hexdigest() == meta["computed_md5"])


def _record(item: Path, url: str, status: int, body: bytes, accepted: bool,
            attempt: int) -> dict:
    """Store the bytes once and append their manifest row."""
    now = _now()
    path = raw_store.store(item, body, accepted, now)
    row = {
        "retrieved_at_utc": now.isoformat(timespec="seconds"),
        "path": path.as_posix(),
        "url": url,
        "http_status": status,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "rejected": int(not accepted),
        "attempt": attempt,
        "revision": f"v{VERSION}",
    }
    raw_store.append_manifest(MANIFEST, row)
    return row


def fetch_article(http) -> tuple[Optional[dict], dict]:
    """The article record at the pinned version: (record or None, manifest row)."""
    r, attempt = http_get.get(http, API)
    body = r.content
    ok = usable_article(r.status_code, body)
    row = _record(article_item(), API, r.status_code, body, ok, attempt)
    return (json.loads(body) if ok else None), row


def fetch_file(http, meta: dict) -> dict:
    """One file the article lists, accepted only if its size and MD5 match."""
    url = meta["download_url"]
    r, attempt = http_get.get(http, url)
    body = r.content
    ok = r.status_code == 200 and file_matches(meta, body)
    return _record(file_item(meta["name"]), url, r.status_code, body, ok, attempt)


def stored_article() -> Optional[dict]:
    """The newest accepted article record in raw/, or None."""
    current = raw_store.current_version(article_item())
    return json.loads(current.read_bytes()) if current else None


def pull(refresh: bool = False) -> list[dict]:
    """Fetch what raw/ does not hold yet: the article record, then each file
    it lists. With refresh, fetch all of it again, as new files beside the
    old. Returns the manifest rows of the requests made, and raises if a
    response was rejected (its row is in the manifest all the same)."""
    raw_store.check_manifest(MANIFEST)
    doc = None if refresh else stored_article()
    if doc is not None and all(raw_store.current_version(file_item(f["name"]))
                               for f in doc["files"]):
        return []
    http = http_get.session()
    rows = []
    if doc is None:
        doc, row = fetch_article(http)
        rows.append(row)
        if doc is None:
            raise RuntimeError(f"article record rejected: HTTP {row['http_status']}"
                               f" -> {row['path']}")
    for meta in doc["files"]:
        if not refresh and raw_store.current_version(file_item(meta["name"])):
            continue
        row = fetch_file(http, meta)
        rows.append(row)
        if row["rejected"]:
            raise RuntimeError(f"{meta['name']} rejected: HTTP {row['http_status']}, or its"
                               f" size or MD5 differs from the article -> {row['path']}")
    return rows


def main(argv: list[str]) -> int:
    """python -m src.ingest.figshare_client [--refresh]

    The Potash et al. (2023) field data, figshare article 23669304 version 2:
    the article record and each file not yet in raw/. --refresh fetches all of
    it again, as new files beside the old.
    """
    if argv not in ([], ["--refresh"]):
        print(main.__doc__)
        return 2
    try:
        rows = pull(refresh=bool(argv))
    except RuntimeError as e:
        print(f"FAILED: {e}")
        return 1
    if not rows:
        print("nothing to fetch: the article record and every file are in raw/")
    for r in rows:
        print(f"HTTP {r['http_status']}  {r['bytes']:>8} bytes  {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
