"""The National Map ingestion for Stage 3: the newest USGS 3DEP 1/3
arc-second tiles under the eight Potash fields, read in part.

For each field the TNM Access API is asked which tiles of the 1/3 arc-second
DEM cover the field's patch (RUNBOOK, Stage 3, D2). For each 1 x 1 degree
block the patch touches, the version titled "USGS 1/3 Arc Second" with the
latest publication date is taken (D1), unless the manifest already records a
version of that block: that one is then read again, so that a later pull
does not change what Stage 3 analyses.

Only the parts of a tile that Stage 3 needs are read, by HTTP range
requests: the header, from the first byte to the first byte of image data,
and the internal blocks that cover a patch with two native cells to spare.
The tiles are Cloud Optimized GeoTIFFs, so a block can be read on its own.
The client refuses to start on the blocks if they would come to more than
--max-mb megabytes (100 by default): whole tiles are 0.4-0.5 GB each.

Same discipline and manifest as Stage 1:
  * raw/ is append-only (src/ingest/raw_store.py): raw/TNM/listing/<field>/,
    raw/TNM/metadata/<block>/<version>/ and, for every range,
    raw/TNM/tiles/<block>/<version>/bytes_<first>_<last>/.
  * Every response is hashed and recorded with its URL, status, size and
    time. A range response's row also records, in `revision`, the byte
    range, the server's ETag and the file's total size.
  * A response to a range request that is not that range (another status,
    another range or another file size) is refused without its body being
    read, so a server that ignores the range cannot start a 0.4 GB
    download. Its status and headers go to _failed/ instead, and the pull
    stops.
No account and no token: the tiles are public.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
import tifffile

try:                        # run as a module from the repository root
    from src.ingest import http_get, raw_store
    from src.transform import cog, patches as P
except ImportError:         # imported with src/ingest and src/transform on sys.path (pytest)
    import cog
    import http_get
    import patches as P
    import raw_store

API = "https://tnmaccess.nationalmap.gov/api/v1/products"
DATASET = "National Elevation Dataset (NED) 1/3 arc-second"
TITLE = re.compile(r"^USGS 1/3 Arc Second (n\d{2}w\d{3}) (\d{8})$")
TILE_URL = cog.TILE_URL
MAX_ITEMS = 100
HEADER_CHUNK = 1 << 16
MAX_MB = 100.0
RAW = Path("raw")
MANIFEST = RAW / "manifest.csv"
STOCKS = Path("curated") / "potash_stocks_0_30.parquet"


def _now() -> dt.datetime:
    """The retrieval time, in UTC (replaced in the tests)."""
    return dt.datetime.now(dt.timezone.utc)


# --- logical items -------------------------------------------------------------------

def listing_item(site: str) -> Path:
    return RAW / "TNM" / "listing" / f"{site}.json"


def metadata_url(item: dict) -> str:
    """The metadata record of a listing entry: the FGDC XML (vendorMetaUrl)
    when it is one, which names the tile's sources; else the ScienceBase
    record (metaUrl) as JSON. Some entries' vendorMetaUrl is a page of the
    S3 bucket's browser, not a record."""
    vendor = str(item.get("vendorMetaUrl") or "")
    if urlparse(vendor).path.endswith(".xml"):
        return vendor
    if item.get("metaUrl"):
        return f"{item['metaUrl']}?format=json"
    if vendor:
        return vendor
    raise RuntimeError(f"{item.get('title')}: the listing links no metadata record")


def metadata_item(block: str, version: str, url: str) -> Path:
    suffix = Path(urlparse(url).path).suffix
    if "format=json" in urlparse(url).query:
        suffix = ".json"
    return RAW / "TNM" / "metadata" / block / f"{version}{suffix or '.bin'}"


def range_item(block: str, version: str, first: int, last: int) -> Path:
    return RAW / "TNM" / "tiles" / block / version / f"bytes_{first}_{last}.bin"


tile_version = cog.tile_version


# --- the listing ---------------------------------------------------------------------

def listing_url(bbox: tuple[float, float, float, float]) -> str:
    """The TNM Access API query for the 1/3 arc-second products in a box."""
    params = {"datasets": DATASET, "bbox": ",".join(f"{v:.6f}" for v in bbox),
              "outputFormat": "JSON", "max": MAX_ITEMS}
    return requests.Request("GET", API, params=params).prepare().url


def usable_listing(status: int, body: bytes) -> bool:
    """HTTP 200 and a complete listing: JSON with its items, all of them
    returned (total no larger than what came back)."""
    if status != 200:
        return False
    try:
        doc = json.loads(body)
    except ValueError:
        return False
    items = doc.get("items") if isinstance(doc, dict) else None
    return (isinstance(items, list) and isinstance(doc.get("total"), int)
            and doc["total"] == len(items) <= MAX_ITEMS)


def versions(items: list[dict]) -> dict[str, list[dict]]:
    """The 1/3 arc-second GeoTIFF versions in a listing, by block. Items
    with another title, such as the older "USGS 13 arc-second ... 1 x 1
    degree" products, are not versions of the tile."""
    out: dict[str, list[dict]] = {}
    for it in items:
        m = TITLE.match(str(it.get("title", "")))
        if not m or it.get("format") != "GeoTIFF":
            continue
        block, version = tile_version(str(it.get("downloadURL", "")))
        if block != m.group(1) or version != m.group(2):
            raise ValueError(f"title and URL disagree: {it.get('title')} / {it.get('downloadURL')}")
        out.setdefault(block, []).append(it)
    return out


def newest(items: list[dict]) -> dict:
    """The version with the latest publication date (D1); among equal
    dates, the one whose own date is latest."""
    return max(items, key=lambda it: (str(it["publicationDate"]), tile_version(it["downloadURL"])[1]))


def pinned(manifest: Path = None) -> dict[str, str]:
    """block -> the tile URL the manifest already records for it."""
    manifest = MANIFEST if manifest is None else manifest
    out: dict[str, str] = {}
    if not manifest.exists():
        return out
    for row in pd.read_csv(manifest, dtype=str, keep_default_na=False).itertuples():
        if row.rejected != "0" or not TILE_URL.search(row.url):
            continue
        block, _ = tile_version(row.url)
        if out.setdefault(block, row.url) != row.url:
            raise RuntimeError(f"the manifest records two versions of {block}")
    return out


# --- responses -----------------------------------------------------------------------

def _record(item: Path, url: str, status: int, body: bytes, accepted: bool,
            attempt: int, revision: str = "") -> dict:
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
        "revision": revision,
    }
    raw_store.append_manifest(MANIFEST, row)
    return row


def fetch_listing(http, site: str, bbox) -> tuple[Optional[dict], dict]:
    url = listing_url(bbox)
    r, attempt = http_get.get(http, url)
    ok = usable_listing(r.status_code, r.content)
    row = _record(listing_item(site), url, r.status_code, r.content, ok, attempt)
    return (json.loads(r.content) if ok else None), row


def fetch_metadata(http, item: dict) -> dict:
    """The metadata record a version's listing entry links (metadata_url)."""
    block, version = tile_version(item["downloadURL"])
    url = metadata_url(item)
    r, attempt = http_get.get(http, url)
    ok = r.status_code == 200 and len(r.content) > 0
    return _record(metadata_item(block, version, url), url, r.status_code, r.content, ok, attempt)


CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def get_range(http, url: str, first: int, last: int, *,
              max_retries: int = http_get.MAX_RETRIES,
              sleep: Optional[Callable[[float], None]] = None):
    """GET bytes first..last of url, streamed so that the body is read only
    once the caller has seen the status and headers. Retries as
    http_get.get does: network errors, 429 and 5xx."""
    sleep = sleep or time.sleep
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        wait = float(min(2 ** attempt, 30))
        try:
            r = http.get(url, headers={"Range": f"bytes={first}-{last}"},
                         stream=True, timeout=http_get.TIMEOUT)
        except requests.RequestException as e:
            last_error = e
        else:
            if r.status_code != 429 and r.status_code < 500:
                return r, attempt
            last_error = requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code == 429:
                wait = http_get.retry_after(r)
            r.close()
        if attempt < max_retries:
            sleep(wait)
    raise RuntimeError(f"{url} bytes {first}-{last}: no usable response after "
                       f"{max_retries} attempts ({last_error})")


class Tile:
    """One tile version: its stored ranges, and fetching what is missing."""

    def __init__(self, http, item: dict, rows: list[dict]):
        self.http, self.item, self.rows = http, item, rows
        self.url = item["downloadURL"]
        self.block, self.version = tile_version(self.url)
        self.size = int(item["sizeInBytes"])
        self.etag: Optional[str] = None
        self.ranges: dict[int, bytes] = {}
        self.fetched = 0
        for first, body, etag, size in stored_ranges(self.url):
            if size != self.size:
                raise RuntimeError(f"{self.url}: a stored range says the file has {size} bytes, "
                                   f"the listing {self.size}")
            self._keep(first, body, etag)

    def _keep(self, first: int, body: bytes, etag: str) -> None:
        if self.etag is None:
            self.etag = etag
        elif etag != self.etag:
            raise RuntimeError(f"{self.url} changed between requests: ETag {self.etag} then {etag}")
        self.ranges[first] = body

    def covered(self, first: int, last: int) -> bool:
        try:
            cog.RangeFile(self.ranges, self.size)._get(first, last - first + 1)
            return True
        except cog.OutsideStoredRanges:
            return False

    def ensure(self, first: int, last: int) -> None:
        """Fetch bytes first..last unless they are stored already."""
        last = min(last, self.size - 1)
        if first > last or self.covered(first, last):
            return
        r, attempt = get_range(self.http, self.url, first, last)
        m = CONTENT_RANGE.match(r.headers.get("Content-Range", ""))
        etag = r.headers.get("ETag", "")
        ok = (r.status_code == 206 and m is not None
              and (int(m.group(1)), int(m.group(2)), int(m.group(3))) == (first, last, self.size)
              and bool(etag))
        if not ok:
            note = json.dumps({"status": r.status_code, "headers": dict(r.headers),
                               "requested": f"bytes={first}-{last}",
                               "body": "not read: not the requested range"}, indent=1).encode()
            r.close()
            row = _record(range_item(self.block, self.version, first, last), self.url,
                          r.status_code, note, False, attempt, f"bytes={first}-{last}")
            self.rows.append(row)
            raise RuntimeError(f"{self.url} bytes {first}-{last}: HTTP {r.status_code}, "
                               f"Content-Range {r.headers.get('Content-Range')!r} -> {row['path']}")
        body = r.content
        if len(body) != last - first + 1:
            raise RuntimeError(f"{self.url} bytes {first}-{last}: {len(body)} bytes came")
        row = _record(range_item(self.block, self.version, first, last), self.url, 206, body,
                      True, attempt, f"bytes={first}-{last};etag={etag};size={self.size}")
        self.rows.append(row)
        self.fetched += len(body)
        self._keep(first, body, etag)

    def file(self) -> cog.RangeFile:
        return cog.RangeFile(self.ranges, self.size)

    def header(self):
        """Read the header, fetching it in chunks as tifffile needs it, then
        the rest of it up to the first byte of image data. Returns the
        grid and, per block, its offset and byte count."""
        self.ensure(0, HEADER_CHUNK - 1)
        for _ in range(256):
            try:
                with tifffile.TiffFile(self.file()) as tif:
                    page = tif.pages[0]
                    geo = cog.geometry(page)
                    first_data = cog.first_data_offset(tif)
                    offsets = [int(v) for v in page.dataoffsets]
                    counts = [int(v) for v in page.databytecounts]
                break
            except cog.OutsideStoredRanges as e:
                start = (e.pos // HEADER_CHUNK) * HEADER_CHUNK
                self.ensure(start, max(e.pos + e.n - 1, start + HEADER_CHUNK - 1))
        else:
            raise RuntimeError(f"{self.url}: the header did not parse")
        self.ensure(0, first_data - 1)
        return geo, offsets, counts


parse_revision = cog.parse_revision


def stored_ranges(url: str, manifest: Path = None) -> list[tuple[int, bytes, str, int]]:
    """(first byte, body, ETag, file size) of every accepted range of url."""
    manifest = MANIFEST if manifest is None else manifest
    if not manifest.exists():
        return []
    out = []
    for row in pd.read_csv(manifest, dtype=str, keep_default_na=False).itertuples():
        if row.url != url or row.rejected != "0" or not row.revision.startswith("bytes="):
            continue
        rev = parse_revision(row.revision)
        body = Path(row.path).read_bytes()
        if hashlib.sha256(body).hexdigest() != row.sha256:
            raise RuntimeError(f"{row.path} does not match its manifest row")
        out.append((rev["first"], body, rev["etag"], rev["size"]))
    return out


# --- the pull ------------------------------------------------------------------------

def plan(loc: pd.DataFrame) -> dict[str, tuple[P.Grid, tuple, list[str]]]:
    """site -> (patch, longitude-latitude box, blocks)."""
    out = {}
    for site, grid in P.patch_grids(loc).items():
        bbox = P.lonlat_bbox(grid)
        out[site] = (grid, bbox, P.block_names(bbox))
    return out


def pull(refresh: bool = False, max_mb: float = MAX_MB) -> list[dict]:
    """Fetch what raw/ does not hold yet: each field's listing, each tile
    version's header and metadata record, and the blocks that cover the
    patches. With refresh, ask for the listings again (the versions stay
    pinned). Returns the manifest rows of the requests made."""
    raw_store.check_manifest(MANIFEST)
    loc = pd.read_parquet(STOCKS, columns=P.LOCATION_COLUMNS)
    fields = plan(loc)
    pins = pinned()
    http = http_get.session()
    rows: list[dict] = []
    chosen: dict[str, dict] = {}
    for site, (grid, bbox, blocks) in fields.items():
        stored = None if refresh else raw_store.current_version(listing_item(site))
        if stored is not None:
            doc = json.loads(stored.read_bytes())
        else:
            doc, row = fetch_listing(http, site, bbox)
            rows.append(row)
            if doc is None:
                raise RuntimeError(f"{site}: listing rejected: HTTP {row['http_status']} -> {row['path']}")
        by = versions(doc["items"])
        for block in blocks:
            if block not in by:
                raise RuntimeError(f"{site}: the listing has no 1/3 arc-second version of {block}")
            if block in pins:
                match = [it for it in by[block] if it["downloadURL"] == pins[block]]
                if not match:
                    raise RuntimeError(f"{block}: the pinned version is not in {site}'s listing")
                item = match[0]
            else:
                item = newest(by[block])
            if chosen.setdefault(block, item)["downloadURL"] != item["downloadURL"]:
                raise RuntimeError(f"{block}: two fields' listings give different newest versions")
    tiles = {block: Tile(http, item, rows) for block, item in sorted(chosen.items())}
    wanted: dict[str, list[tuple[int, int]]] = {}
    for block, tile in tiles.items():
        geo, offsets, counts = tile.header()
        spans = []
        for site, (grid, _, blocks) in fields.items():
            if block not in blocks:
                continue
            window = cog.window_for(grid, geo)
            if window is None:
                raise RuntimeError(f"{site}: the patch misses tile {block}")
            idx = cog.blocks_for(geo, window)
            spans += [(offsets[i], offsets[i] + counts[i] - 1) for i in idx]
        wanted[block] = sorted(set(spans))
    planned = sum(b - a + 1 for block, spans in wanted.items() for a, b in spans
                  if not tiles[block].covered(a, b))
    if planned > max_mb * 1e6:
        raise RuntimeError(f"the blocks would come to {planned / 1e6:.1f} MB, more than {max_mb:.0f} MB")
    for block, spans in wanted.items():
        for a, b in merge(spans):
            tiles[block].ensure(a, b)
        meta = metadata_item(block, tiles[block].version, metadata_url(tiles[block].item))
        if refresh or raw_store.current_version(meta) is None:
            row = fetch_metadata(http, tiles[block].item)
            rows.append(row)
            if row["rejected"]:
                raise RuntimeError(f"{block}: metadata record rejected: HTTP {row['http_status']}")
    return rows


def merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Adjoining or overlapping byte spans joined into one request each."""
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def main(argv: list[str]) -> int:
    """python -m src.ingest.tnm_client [--refresh] [--max-mb N]

    The newest USGS 3DEP 1/3 arc-second tiles under the eight Potash fields,
    read in part from The National Map: what raw/ does not hold yet.
    --refresh asks for the listings again; the versions stay pinned.
    """
    args = list(argv)
    refresh = "--refresh" in args
    if refresh:
        args.remove("--refresh")
    max_mb = MAX_MB
    if args[:1] == ["--max-mb"] and len(args) == 2:
        max_mb, args = float(args[1]), []
    if args:
        print(main.__doc__)
        return 2
    try:
        rows = pull(refresh=refresh, max_mb=max_mb)
    except RuntimeError as e:
        print(f"FAILED: {e}")
        return 1
    if not rows:
        print("nothing to fetch: every listing, header, block and metadata record is in raw/")
    for r in rows:
        print(f"HTTP {r['http_status']}  {r['bytes']:>9} bytes  {r['path']}")
    total = sum(int(r["bytes"]) for r in rows)
    print(f"{len(rows)} responses, {total / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
