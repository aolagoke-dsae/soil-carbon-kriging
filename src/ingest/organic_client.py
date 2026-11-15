"""Thünen Atlas: the updated map of organic soils in Germany (Wittnebel,
Frank & Tiemeyer 2023, CC BY 4.0), for RUNBOOK, Stage 6, D2.

The layer's name on the Atlas' GeoServer is not published in the dataset's
record, so this client first reads the server's capabilities and lists the
feature types that can be it:

    python -m src.ingest.organic_client --discover
        stores the map's record from the Atlas API and the WFS capabilities
        at each endpoint the Atlas has used, and prints the layer names and
        the service and download links they give

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


def _record(it: Path, url: str, status: int, body: bytes, accepted: bool, attempt: int) -> dict:
    now = _now()
    path = raw_store.store(it, body, accepted, now)
    row = {"retrieved_at_utc": now.isoformat(timespec="seconds"), "path": path.as_posix(), "url": url,
           "http_status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
           "rejected": int(not accepted), "attempt": attempt, "revision": ""}
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


def main(argv: list[str]) -> int:
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
