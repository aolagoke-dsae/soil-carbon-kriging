"""Thünen Atlas ingestion: the BZE-LW site layer, through OGC WFS 2.0.

Germany's first agricultural soil inventory (BZE-LW): 3,104 sites on an
8 x 8 km grid, each with its organic carbon stock in 0-30 cm (kv_0_30, Mg/ha),
its land use and its soil description, CC BY 4.0 (config/sources.csv). The
layer, geonode:bze_lw_standorte_verschleiert, is served by the Thünen Atlas'
GeoServer without a token:

    https://atlas.thuenen.de/geoserver/ows

Its coordinates are NOT the sampling sites. For data protection each was
generated at random within 4 km of the originally planned point (the
xcoord/ycoord attribute description, https://atlas.thuenen.de/api/v2/datasets/11).
Every analysis of these points carries that offset.

GetFeature is paged, PAGE features at a time, sorted by pointid so that the
pages partition the layer: without a sort order a server may return the rows
of consecutive pages in different orders, and a page boundary can then skip a
site or repeat one. Each page is a logical item in raw/ (src/ingest/raw_store.py)
named by its startIndex. The server states the total, numberMatched, in every
page; paging stops when the pages reach it, and Checkpoint 1 checks that they
add up to it.
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

ENDPOINT = "https://atlas.thuenen.de/geoserver/ows"
LAYER = "geonode:bze_lw_standorte_verschleiert"
PAGE = 1000
RAW = Path("raw")
MANIFEST = RAW / "manifest.csv"


def _now() -> dt.datetime:
    """The retrieval time, in UTC (replaced in the tests)."""
    return dt.datetime.now(dt.timezone.utc)


def root() -> Path:
    """raw/THUENEN/bze_lw_standorte_verschleiert/"""
    return RAW / "THUENEN" / LAYER.split(":", 1)[1]


def page_item(start: int) -> Path:
    """A page's logical item, e.g. raw/THUENEN/bze_lw_standorte_verschleiert/start_01000.json"""
    return root() / f"start_{start:05d}.json"


def params(start: int) -> dict:
    """GetFeature for one page of the layer, as GeoJSON, in pointid order."""
    return {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": LAYER, "outputFormat": "application/json",
            "sortBy": "pointid", "count": PAGE, "startIndex": start}


def usable(status: int, body: bytes) -> bool:
    """HTTP 200 and a GeoJSON FeatureCollection stating numberMatched. A
    GeoServer exception report is XML, sometimes sent with HTTP 200: not data."""
    if status != 200:
        return False
    try:
        doc = json.loads(body)
    except ValueError:
        return False
    return (isinstance(doc, dict) and doc.get("type") == "FeatureCollection"
            and isinstance(doc.get("features"), list)
            and isinstance(doc.get("numberMatched"), int))


def fetch_page(http, start: int) -> tuple[Optional[dict], dict]:
    """One page: (FeatureCollection or None, manifest row)."""
    r, attempt = http_get.get(http, ENDPOINT, params(start))
    body = r.content
    ok = usable(r.status_code, body)
    now = _now()
    path = raw_store.store(page_item(start), body, ok, now)
    row = {
        "retrieved_at_utc": now.isoformat(timespec="seconds"),
        "path": path.as_posix(),
        "url": r.url,
        "http_status": r.status_code,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "rejected": int(not ok),
        "attempt": attempt,
        "revision": "",
    }
    raw_store.append_manifest(MANIFEST, row)
    return (json.loads(body) if ok else None), row


def pull(refresh: bool = False) -> list[dict]:
    """Every page raw/ does not hold yet, from startIndex 0 until the pages
    reach numberMatched. With refresh, every page again. Returns the manifest
    rows of the requests made; raises if a page was rejected, or came back
    empty before the total was reached."""
    raw_store.check_manifest(MANIFEST)
    http = None
    rows: list[dict] = []
    start = 0
    while True:
        stored = None if refresh else raw_store.current_version(page_item(start))
        if stored is not None:
            doc = json.loads(stored.read_bytes())
        else:
            http = http or http_get.session()
            doc, row = fetch_page(http, start)
            rows.append(row)
            if doc is None:
                raise RuntimeError(f"page at startIndex {start} rejected: "
                                   f"HTTP {row['http_status']} -> {row['path']}")
        n = len(doc["features"])
        if n == 0:
            raise RuntimeError(f"page at startIndex {start} is empty, but the server "
                               f"states numberMatched = {doc['numberMatched']}")
        start += n
        if start >= doc["numberMatched"]:
            return rows


def main(argv: list[str]) -> int:
    """python -m src.ingest.thuenen_wfs_client [--refresh]

    The BZE-LW site layer from the Thünen Atlas, page by page: every page not
    yet in raw/. --refresh fetches every page again, as new files beside the old.
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
        print("nothing to fetch: every page is in raw/")
    for r in rows:
        print(f"HTTP {r['http_status']}  {r['bytes']:>8} bytes  {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
