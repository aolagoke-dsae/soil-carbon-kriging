"""Thünen Atlas: the updated map of organic soils in Germany (Wittnebel,
Frank & Tiemeyer 2023, CC BY 4.0), for RUNBOOK, Stage 6, D2.

The layer's name on the Atlas' GeoServer is not published in the dataset's
record, so this client first reads the server's capabilities and lists the
feature types that can be it:

    python -m src.ingest.organic_client --discover
        stores the WFS capabilities, and for each candidate layer its
        DescribeFeatureType and its number of features, and prints them

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


def discover() -> list[dict]:
    raw_store.check_manifest(MANIFEST)
    http = http_get.session()
    r, attempt = http_get.get(http, ENDPOINT, {"service": "WFS", "version": "2.0.0", "request": "GetCapabilities"})
    try:
        types = feature_types(r.content) if r.status_code == 200 else []
    except ET.ParseError:
        types = []
    _record(item("capabilities.xml"), r.url, r.status_code, r.content, bool(types), attempt)
    if not types:
        raise RuntimeError(f"capabilities rejected (HTTP {r.status_code})")
    found = []
    for name, title in candidates(types):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        r, attempt = http_get.get(http, ENDPOINT, {"service": "WFS", "version": "2.0.0",
                                                   "request": "DescribeFeatureType", "typeNames": name})
        _record(item(f"describe_{safe}.xml"), r.url, r.status_code, r.content, r.status_code == 200, attempt)
        attrs = re.findall(rb'<xsd:element[^>]*name="([^"]+)"[^>]*type="([^"]+)"', r.content)
        r2, attempt = http_get.get(http, ENDPOINT, {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                                                    "typeNames": name, "resultType": "hits"})
        _record(item(f"hits_{safe}.xml"), r2.url, r2.status_code, r2.content, r2.status_code == 200, attempt)
        found.append({"name": name, "title": title, "features": number_matched(r2.content),
                      "attributes": [(a.decode(), t.decode()) for a, t in attrs]})
    return found


def main(argv: list[str]) -> int:
    if argv != ["--discover"]:
        print(__doc__)
        return 2
    try:
        found = discover()
    except RuntimeError as e:
        print(f"FAILED: {e}")
        return 1
    if not found:
        print("no feature type's name or title mentions organic soils, peat or a 'Kulisse'")
    for f in found:
        print(f"{f['name']}  |  {f['title']}  |  {f['features']} features")
        print("    " + ", ".join(f"{a}:{t.split(':')[-1]}" for a, t in f["attributes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
