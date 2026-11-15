"""Thünen Atlas: the updated map of organic soils in Germany (Wittnebel,
Frank & Tiemeyer 2023, CC BY 4.0), for RUNBOOK, Stage 6, D2.

The layer's name on the Atlas' GeoServer is not published in the dataset's
record, so this client first reads the server's capabilities and lists the
feature types that can be it:

    python -m src.ingest.organic_client --discover
        stores the map's record from the Atlas API and the WFS capabilities
        at each endpoint the Atlas has used, and prints the layer names and
        the service and download links they give

    python -m src.ingest.organic_client [--refresh]
        the dataset itself, from its DOI on OpenAgrar: the landing page,
        README.txt and Thuenen_Kulisse_OrgBoeden_v1.zip (a GeoPackage), the
        zip accepted only if it holds a .gpkg

On 15.11.2026 the WFS endpoint Stage 1 used, atlas.thuenen.de/geoserver/ows,
answered GetCapabilities with HTTP 404 (RUNBOOK, Stage 6, If it fails).

The pull of the chosen layer is added once its name, attributes and size
are known (RUNBOOK, Stage 6, If it fails, records the choice).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:                        # run as a module from the repository root
    from src.ingest import http_get, raw_store
except ImportError:         # imported with src/ingest on sys.path (pytest)
    import http_get
    import raw_store

ENDPOINT = "https://atlas.thuenen.de/geoserver/ows"
RAW = Path("raw")
MANIFEST = RAW / "manifest.csv"
CANDIDATE = re.compile(r"organ|moor|kulisse|torf", re.IGNORECASE)
WFS = "{http://www.opengis.net/wfs/2.0}"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def item(name: str) -> Path:
    return RAW / "THUENEN" / "organic" / name


def feature_types(xml: bytes) -> list[tuple[str, str]]:
    """(Name, Title) of every FeatureType in a WFS 2.0 capabilities document."""
    root = ET.fromstring(xml)
    out = []
    for ft in root.iter(f"{WFS}FeatureType"):
        name, title = ft.find(f"{WFS}Name"), ft.find(f"{WFS}Title")
        out.append(((name.text or "").strip() if name is not None else "",
                    (title.text or "").strip() if title is not None else ""))
    return out


def candidates(types: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(n, t) for n, t in types if CANDIDATE.search(n) or CANDIDATE.search(t)]


def number_matched(xml: bytes) -> int | None:
    m = re.search(rb'numberMatched="(\d+)"', xml)
    return int(m.group(1)) if m else None


def _record(it: Path, url: str, status: int, body: bytes, accepted: bool, attempt: int, revision: str = "") -> dict:
    now = _now()
    path = raw_store.store(it, body, accepted, now)
    row = {"retrieved_at_utc": now.isoformat(timespec="seconds"), "path": path.as_posix(), "url": url,
           "http_status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
           "rejected": int(not accepted), "attempt": attempt, "revision": revision}
    raw_store.append_manifest(MANIFEST, row)
    return row


API = "https://atlas.thuenen.de/api/v2/maps/243"       # the map's own page: atlas.thuenen.de/maps/243
ENDPOINTS = ("https://atlas.thuenen.de/geoserver/ows", "https://atlas.thuenen.de/geoserver/wfs",
             "https://atlas.thuenen.de/gs/ows")


def walk_links(doc) -> list[str]:
    """Every string in a JSON document that looks like an OGC or download URL."""
    out = []
    if isinstance(doc, dict):
        for v in doc.values():
            out += walk_links(v)
    elif isinstance(doc, list):
        for v in doc:
            out += walk_links(v)
    elif isinstance(doc, str) and doc.startswith("http") and re.search(r"ows|wfs|wms|zip|shp|json|gpkg|download", doc, re.I):
        out.append(doc)
    return out


def layer_names(doc) -> list[str]:
    """Every value of a key 'alternate' or 'name' that looks like workspace:layer."""
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k in ("alternate", "name") and isinstance(v, str) and ":" in v:
                out.append(v)
            else:
                out += layer_names(v)
    elif isinstance(doc, list):
        for v in doc:
            out += layer_names(v)
    return out


def discover() -> list[dict]:
    """The map's record from the Atlas API, then the WFS capabilities at each
    endpoint the Atlas has used, every response stored."""
    import json
    raw_store.check_manifest(MANIFEST)
    http = http_get.session()
    found = []
    r, attempt = http_get.get(http, API)
    try:
        doc = json.loads(r.content) if r.status_code == 200 else None
    except ValueError:
        doc = None
    _record(item("map_243.json"), r.url, r.status_code, r.content, doc is not None, attempt)
    found.append({"what": API, "status": r.status_code,
                  "layers": sorted(set(layer_names(doc))) if doc else [],
                  "links": sorted(set(walk_links(doc)))[:40] if doc else []})
    for i, ep in enumerate(ENDPOINTS):
        r, attempt = http_get.get(http, ep, {"service": "WFS", "version": "2.0.0", "request": "GetCapabilities"})
        try:
            types = feature_types(r.content) if r.status_code == 200 else []
        except ET.ParseError:
            types = []
        _record(item(f"capabilities_{i}.xml"), r.url, r.status_code, r.content, bool(types), attempt)
        found.append({"what": ep, "status": r.status_code, "layers": [f"{n} | {t}" for n, t in candidates(types)],
                      "links": [], "types": len(types)})
    return found


# --- the dataset on OpenAgrar ------------------------------------------------------------------

DOI = "https://doi.org/10.3220/DATA20230510130443-0"
ZIP_NAME = "Thuenen_Kulisse_OrgBoeden_v1.zip"       # 183.49 MB, dated 2023-08-04 on the record
README_NAME = "README.txt"


def file_links(html: str, base: str) -> dict[str, str]:
    """{file name: absolute URL} of the files a repository landing page links to."""
    from urllib.parse import urljoin, unquote
    out = {}
    for href in re.findall(r'href="([^"]+)"', html):
        name = unquote(href.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1])
        if name in (ZIP_NAME, README_NAME):
            out.setdefault(name, urljoin(base, href.replace("&amp;", "&")))
    return out


def usable_zip(status: int, body: bytes) -> bool:
    """HTTP 200 and a zip that holds a GeoPackage."""
    import io
    import zipfile
    if status != 200:
        return False
    try:
        names = zipfile.ZipFile(io.BytesIO(body)).namelist()
    except zipfile.BadZipFile:
        return False
    return any(n.lower().endswith(".gpkg") for n in names)


def pull(refresh: bool = False) -> list[dict]:
    """The DOI's landing page, then the README and the zip it links to."""
    raw_store.check_manifest(MANIFEST)
    if not refresh and raw_store.current_version(item(ZIP_NAME)):
        return []
    http = http_get.session()
    rows = []
    r, attempt = http_get.get(http, DOI)
    links = file_links(r.content.decode("utf-8", "replace"), r.url) if r.status_code == 200 else {}
    ok = ZIP_NAME in links
    rows.append(_record(item("landing.html"), r.url, r.status_code, r.content, ok, attempt))
    if not ok:
        raise RuntimeError(f"the DOI's landing page (HTTP {r.status_code}) links no {ZIP_NAME}")
    if README_NAME in links:
        r, attempt = http_get.get(http, links[README_NAME])
        rows.append(_record(item(README_NAME), links[README_NAME], r.status_code, r.content, r.status_code == 200,
                            attempt))
    r, attempt = http_get.get(http, links[ZIP_NAME])
    ok = usable_zip(r.status_code, r.content)
    rows.append(_record(item(ZIP_NAME), links[ZIP_NAME], r.status_code, r.content, ok, attempt, DOI))
    if not ok:
        raise RuntimeError(f"{ZIP_NAME} rejected (HTTP {r.status_code}, or no GeoPackage inside)")
    return rows


def main(argv: list[str]) -> int:
    if argv in ([], ["--refresh"]):
        try:
            rows = pull(refresh=bool(argv))
        except RuntimeError as e:
            print(f"FAILED: {e}")
            return 1
        if not rows:
            print(f"nothing to fetch: raw/ holds {ZIP_NAME}")
        for r in rows:
            print(f"HTTP {r['http_status']}  {r['bytes']:>10} bytes  {r['path']}")
        return 0
    if argv != ["--discover"]:
        print(__doc__)
        return 2
    for f in discover():
        extra = f" ({f['types']} feature types)" if "types" in f else ""
        print(f"{f['what']}: HTTP {f['status']}{extra}")
        for x in f["layers"]:
            print(f"    layer  {x}")
        for x in f["links"]:
            print(f"    link   {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
