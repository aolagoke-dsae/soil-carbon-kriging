"""DWD Climate Data Center: the multi-annual 1 km grids of mean annual air
temperature and annual precipitation over Germany (RUNBOOK, Stage 6, D2).

CC BY 4.0 (https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf,
read 15.11.2026). The grids are gzipped ESRI ASCII rasters in Gauss-Krüger
zone 3 (EPSG:31467), one file per reference period and month, season or year;
the file ending in _17 is the whole year (readme_00_17.txt).

The directory listings are read first and stored, and the annual files are
chosen from them: the most recent reference period that both variables
publish. Their names are therefore recorded in raw/ and the manifest, not
assumed here. Same discipline as the other clients: write-once raw/,
a manifest row per response, the shared retry rules.

    python -m src.ingest.dwd_client [--refresh]
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

try:                        # run as a module from the repository root
    from src.ingest import http_get, raw_store
except ImportError:         # imported with src/ingest on sys.path (pytest)
    import http_get
    import raw_store

BASE = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/multi_annual/"
VARIABLES = ("air_temperature_mean", "precipitation")
RAW = Path("raw")
MANIFEST = RAW / "manifest.csv"
ANNUAL = re.compile(r'href="([^"]*?((?:19|20)\d\d)[-_]((?:19|20)\d\d)[^"/]*?_17\.asc\.gz)"')


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def listing_item(variable: str) -> Path:
    return RAW / "DWD" / variable / "listing.html"


def grid_item(variable: str, name: str) -> Path:
    return RAW / "DWD" / variable / name


ANNUAL_SHORT = re.compile(r'href="([^"]*?_(\d\d)(\d\d)_17\.asc\.gz)"')     # e.g. _6190_17: 1961-1990


def _year(yy: str) -> int:
    return (1900 if int(yy) >= 50 else 2000) + int(yy)


def annual_files(html: str) -> dict[tuple[int, int], str]:
    """{(first year, last year): file name} of the annual (_17) grids in a
    listing, whether the period is written 1991-2020, 1991_2020 or 9120."""
    out = {(int(a), int(b)): name.rsplit("/", 1)[-1] for name, a, b in ANNUAL.findall(html)}
    for name, a, b in ANNUAL_SHORT.findall(html):
        period = (_year(a), _year(b))
        if period[1] - period[0] == 29:
            out.setdefault(period, name.rsplit("/", 1)[-1])
    return out


def latest_common(listings: dict[str, str]) -> tuple[int, int]:
    """The most recent reference period whose annual grid every variable publishes."""
    periods = [set(annual_files(h)) for h in listings.values()]
    common = set.intersection(*periods)
    if not common:
        raise RuntimeError("no reference period is published for every variable")
    return max(common, key=lambda p: (p[1], p[0]))


def usable_listing(status: int, body: bytes) -> bool:
    return status == 200 and bool(annual_files(body.decode("utf-8", "replace")))


def usable_grid(status: int, body: bytes) -> bool:
    """HTTP 200 and a gzipped ESRI ASCII grid with its six header lines."""
    if status != 200:
        return False
    try:
        head = gzip.decompress(body)[:400].decode("ascii", "replace").lower().split()
    except OSError:
        return False
    return all(k in head for k in ("ncols", "nrows", "cellsize", "nodata_value"))


def _record(item: Path, url: str, status: int, body: bytes, accepted: bool, attempt: int,
            revision: str = "") -> dict:
    now = _now()
    path = raw_store.store(item, body, accepted, now)
    row = {"retrieved_at_utc": now.isoformat(timespec="seconds"), "path": path.as_posix(), "url": url,
           "http_status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
           "rejected": int(not accepted), "attempt": attempt, "revision": revision}
    raw_store.append_manifest(MANIFEST, row)
    return row


def stored_listing(variable: str) -> Optional[str]:
    p = raw_store.current_version(listing_item(variable))
    return p.read_bytes().decode("utf-8", "replace") if p else None


def pull(refresh: bool = False) -> list[dict]:
    raw_store.check_manifest(MANIFEST)
    http, rows, listings = None, [], {}
    for v in VARIABLES:
        html = None if refresh else stored_listing(v)
        if html is None:
            http = http or http_get.session()
            url = BASE + v + "/"
            r, attempt = http_get.get(http, url)
            ok = usable_listing(r.status_code, r.content)
            rows.append(_record(listing_item(v), url, r.status_code, r.content, ok, attempt))
            if not ok:
                raise RuntimeError(f"{v}: listing rejected (HTTP {r.status_code})")
            html = r.content.decode("utf-8", "replace")
        listings[v] = html
    period = latest_common(listings)
    for v in VARIABLES:
        name = annual_files(listings[v])[period]
        if not refresh and raw_store.current_version(grid_item(v, name)):
            continue
        http = http or http_get.session()
        url = urljoin(BASE + v + "/", name)
        r, attempt = http_get.get(http, url)
        ok = usable_grid(r.status_code, r.content)
        rows.append(_record(grid_item(v, name), url, r.status_code, r.content, ok, attempt,
                            f"{period[0]}-{period[1]}"))
        if not ok:
            raise RuntimeError(f"{name}: rejected (HTTP {r.status_code}, or not a gzipped ASCII grid)")
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
        print("nothing to fetch: raw/DWD/ holds the listings and both grids")
    for r in rows:
        print(f"HTTP {r['http_status']}  {r['bytes']:>9} bytes  {r['path']}  {r['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
