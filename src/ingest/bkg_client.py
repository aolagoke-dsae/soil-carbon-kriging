"""BKG: the digital terrain model of Germany with a 200 m grid, DGM200, as a
GeoTIFF in UTM zone 32 (ETRS89) (RUNBOOK, Stage 6, D2).

Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-de/by-2-0); the
attribution BKG asks for is "© GeoBasis-DE / BKG (<year of retrieval>)
dl-de/by-2-0" (gdz.bkg.bund.de, read 15.11.2026). The zip is accepted only
if its MD5 equals the checksum BKG publishes beside it; MD5 is BKG's
integrity check here, not a security measure.

    python -m src.ingest.bkg_client [--refresh]
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

try:                        # run as a module from the repository root
    from src.ingest import http_get, raw_store
except ImportError:         # imported with src/ingest on sys.path (pytest)
    import http_get
    import raw_store

URL = "https://daten.gdz.bkg.bund.de/produkte/dgm/dgm200/aktuell/dgm200.utm32s.tiff.zip"
RAW = Path("raw")
MANIFEST = RAW / "manifest.csv"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def zip_item() -> Path:
    return RAW / "BKG" / "dgm200" / "dgm200.utm32s.tiff.zip"


def md5_item() -> Path:
    return RAW / "BKG" / "dgm200" / "dgm200.utm32s.tiff.zip.md5"


def checksum(body: bytes) -> str | None:
    """The 32-hex-digit MD5 in BKG's .md5 file, or None."""
    m = re.search(rb"\b([0-9a-fA-F]{32})\b", body)
    return m.group(1).decode().lower() if m else None


def usable_zip(status: int, body: bytes, md5: str) -> bool:
    """HTTP 200, the published MD5, and a zip holding a GeoTIFF."""
    if status != 200 or hashlib.md5(body, usedforsecurity=False).hexdigest() != md5:
        return False
    try:
        names = zipfile.ZipFile(io.BytesIO(body)).namelist()
    except zipfile.BadZipFile:
        return False
    return any(n.lower().endswith((".tif", ".tiff")) for n in names)


def _record(item: Path, url: str, status: int, body: bytes, accepted: bool, attempt: int, revision: str = "") -> dict:
    now = _now()
    path = raw_store.store(item, body, accepted, now)
    row = {"retrieved_at_utc": now.isoformat(timespec="seconds"), "path": path.as_posix(), "url": url,
           "http_status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
           "rejected": int(not accepted), "attempt": attempt, "revision": revision}
    raw_store.append_manifest(MANIFEST, row)
    return row


def pull(refresh: bool = False) -> list[dict]:
    raw_store.check_manifest(MANIFEST)
    if not refresh and raw_store.current_version(zip_item()):
        return []
    http = http_get.session()
    rows = []
    r, attempt = http_get.get(http, URL + ".md5")
    md5 = checksum(r.content) if r.status_code == 200 else None
    rows.append(_record(md5_item(), URL + ".md5", r.status_code, r.content, md5 is not None, attempt))
    if md5 is None:
        raise RuntimeError(f"the checksum file was rejected (HTTP {r.status_code})")
    r, attempt = http_get.get(http, URL)
    ok = usable_zip(r.status_code, r.content, md5)
    rows.append(_record(zip_item(), URL, r.status_code, r.content, ok, attempt, f"md5 {md5}"))
    if not ok:
        raise RuntimeError(f"DGM200 rejected (HTTP {r.status_code}, MD5 differs, or no GeoTIFF inside)")
    return rows


def main(argv: list[str]) -> int:
    if argv not in ([], ["--refresh"]):
        print(__doc__)
        return 2
    try:
        rows = pull(refresh=bool(argv))
    except RuntimeError as e:
        print(f"FAILED: {e}")
        return 1
    if not rows:
        print("nothing to fetch: raw/BKG/ holds DGM200")
    for r in rows:
        print(f"HTTP {r['http_status']}  {r['bytes']:>9} bytes  {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
