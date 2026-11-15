"""Soil Data Access (SDA): SSURGO under the eight Potash fields (RUNBOOK,
Stage 1, Addendum on the soil survey, U1; rules fixed 15.11.2026 before this
was written).

SSURGO is the USDA-NRCS soil survey, public domain. SDA answers SQL POSTed as
JSON to one endpoint, so a query is the logical item's content and the
manifest's `revision` holds the first 12 hex digits of its SHA-256:

    raw/SDA/polygons/<site>.json    map-unit polygons meeting the field's box
    raw/SDA/horizons.json           components and horizons of those map units
    raw/SDA/catalog.json            each survey area's last save date

Same discipline as the other clients: write-once storage (raw_store), a
manifest row per response, the shared retry rules (http_get.post). A
response is accepted only if it is HTTP 200 and SDA's table with the columns
asked for and at least one row; anything else goes to _failed/.

The field boxes come from Stage 2's stock table, as tnm_client's do.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:                        # run as a module from the repository root
    from src.ingest import http_get, raw_store
except ImportError:         # imported with src/ingest on sys.path (pytest)
    import http_get
    import raw_store

URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
RAW = Path("raw")
MANIFEST = RAW / "manifest.csv"
STOCKS = Path("curated") / "potash_stocks_0_30.parquet"
MARGIN_DEG = 0.001                  # about 100 m (U1)
POLYGON_COLUMNS = ["mupolygonkey", "mukey", "areasymbol", "wkt"]
HORIZON_COLUMNS = ["mukey", "cokey", "compname", "comppct_r", "majcompflag",
                   "hzdept_r", "hzdepb_r", "om_l", "om_r", "om_h", "dbthirdbar_r"]
CATALOG_COLUMNS = ["areasymbol", "saverest"]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def polygons_item(site: str) -> Path:
    return RAW / "SDA" / "polygons" / f"{site}.json"


def horizons_item() -> Path:
    return RAW / "SDA" / "horizons.json"


def catalog_item() -> Path:
    return RAW / "SDA" / "catalog.json"


def box(lon, lat, margin: float = MARGIN_DEG) -> tuple[float, float, float, float]:
    return (float(min(lon)) - margin, float(min(lat)) - margin,
            float(max(lon)) + margin, float(max(lat)) + margin)


def box_wkt(b: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = b
    return f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"


def _keys(values) -> str:
    """A SQL IN list of quoted keys; SDA's keys are digits and letters only,
    and anything else is refused rather than quoted."""
    out = []
    for v in sorted(set(map(str, values))):
        if not v.isalnum():
            raise ValueError(f"unexpected key {v!r}")
        out.append(f"'{v}'")
    return ", ".join(out)


def polygons_query(b) -> str:
    return ("SELECT P.mupolygonkey, P.mukey, P.areasymbol, P.mupolygongeo.STAsText() AS wkt "
            "FROM mupolygon AS P WHERE P.mupolygonkey IN (SELECT * FROM "
            f"SDA_Get_Mupolygonkey_from_intersection_with_WktWgs84('{box_wkt(b)}'))")


def horizons_query(mukeys) -> str:
    return ("SELECT C.mukey, C.cokey, C.compname, C.comppct_r, C.majcompflag, H.hzdept_r, H.hzdepb_r, "
            "H.om_l, H.om_r, H.om_h, H.dbthirdbar_r FROM component AS C "
            f"LEFT JOIN chorizon AS H ON H.cokey = C.cokey WHERE C.mukey IN ({_keys(mukeys)}) "
            "ORDER BY C.mukey, C.cokey, H.hzdept_r")


def catalog_query(areas) -> str:
    return f"SELECT areasymbol, saverest FROM sacatalog WHERE areasymbol IN ({_keys(areas)})"


def table(body: bytes, columns: list[str]) -> Optional[pd.DataFrame]:
    """SDA's JSON+COLUMNNAME answer as a frame, or None when it is not one:
    {"Table": [[names...], [row...], ...]} with exactly these columns and at
    least one row (SDA answers {} when no row matches)."""
    try:
        doc = json.loads(body)
    except ValueError:
        return None
    rows = doc.get("Table") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or len(rows) < 2 or rows[0] != columns:
        return None
    return pd.DataFrame(rows[1:], columns=columns)


def _record(item: Path, query: str, status: int, body: bytes, accepted: bool, attempt: int) -> dict:
    now = _now()
    path = raw_store.store(item, body, accepted, now)
    row = {"retrieved_at_utc": now.isoformat(timespec="seconds"), "path": path.as_posix(), "url": URL,
           "http_status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
           "rejected": int(not accepted), "attempt": attempt,
           "revision": "q" + hashlib.sha256(query.encode()).hexdigest()[:12]}
    raw_store.append_manifest(MANIFEST, row)
    return row


def fetch(http, item: Path, query: str, columns: list[str]) -> dict:
    r, attempt = http_get.post(http, URL, {"query": query, "format": "JSON+COLUMNNAME"})
    ok = r.status_code == 200 and table(r.content, columns) is not None
    return _record(item, query, r.status_code, r.content, ok, attempt)


def stored(item: Path, columns: list[str]) -> Optional[pd.DataFrame]:
    current = raw_store.current_version(item)
    return table(current.read_bytes(), columns) if current else None


def pull(refresh: bool = False, stocks: Path = STOCKS) -> list[dict]:
    """Fetch what raw/SDA/ does not hold yet: each field's polygons, then the
    horizons of every map unit in them, then their survey areas' save dates.
    Raises if a response was rejected (its row is in the manifest)."""
    raw_store.check_manifest(MANIFEST)
    t = pd.read_parquet(stocks)
    http, rows = http_get.session(), []

    def need(item):
        return refresh or raw_store.current_version(item) is None

    def get(item, query, columns):
        row = fetch(http, item, query, columns)
        rows.append(row)
        if row["rejected"]:
            raise RuntimeError(f"{item.as_posix()} rejected: HTTP {row['http_status']} -> {row['path']}")

    for site, g in t.groupby("site", sort=True):
        if need(polygons_item(site)):
            get(polygons_item(site), polygons_query(box(g["lon"], g["lat"])), POLYGON_COLUMNS)
    polys = pd.concat([stored(polygons_item(s), POLYGON_COLUMNS) for s in sorted(t["site"].unique())])
    if refresh or need(horizons_item()):
        get(horizons_item(), horizons_query(polys["mukey"]), HORIZON_COLUMNS)
    if refresh or need(catalog_item()):
        get(catalog_item(), catalog_query(polys["areasymbol"]), CATALOG_COLUMNS)
    return rows


def main(argv: list[str]) -> int:
    """python -m src.ingest.sda_client [--refresh]

    SSURGO from Soil Data Access under the eight Potash fields: needs
    curated/potash_stocks_0_30.parquet (python -m src.transform.stocks).
    """
    if argv not in ([], ["--refresh"]):
        print(main.__doc__)
        return 2
    if not STOCKS.exists():
        print(f"NOT THERE: {STOCKS.as_posix()} (python -m src.transform.stocks)")
        return 2
    try:
        rows = pull(refresh=bool(argv))
    except RuntimeError as e:
        print(f"FAILED: {e}")
        return 1
    if not rows:
        print("nothing to fetch: raw/SDA/ holds every item")
    for r in rows:
        print(f"HTTP {r['http_status']}  {r['bytes']:>9} bytes  {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
