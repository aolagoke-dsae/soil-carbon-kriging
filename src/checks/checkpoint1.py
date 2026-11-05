"""Checkpoint 1: both datasets are complete, intact and what their sources say.

The rules were fixed in RUNBOOK.md on 05.11.2026, before any data was pulled,
and each is one function here. A rule passes or fails. Diagnostics are
reported and recorded, never judged: they are what Stage 2 needs to know
before any variogram (how many locations each field has, how far apart the
published BZE-LW points lie), not tests the data could pass by being convenient.

    python -m src.checks.checkpoint1     exit 0: every rule passed
                                         exit 1: a rule failed
                                         exit 2: a dataset is not in raw/ yet

Reads raw/ and config/ only; makes no request.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:                        # run as a module from the repository root
    from src.ingest import figshare_client as F
    from src.ingest import raw_store
    from src.ingest import thuenen_wfs_client as W
except ImportError:         # imported with src/ingest on sys.path (pytest)
    import figshare_client as F
    import raw_store
    import thuenen_wfs_client as W

CONFIG = Path(__file__).resolve().parents[2] / "config"

LICENCE = "CC BY 4.0"
POTASH_SITES = 8            # "eight commercial U.S. Midwest fields" (figshare description)
POTASH_COLUMNS = {
    "locations.csv": {"site", "location_id", "X", "Y"},
    "measurements.csv": {"site", "location_id", "sample_depth_min",
                         "sample_depth_max", "SOCc", "BD"},
}
KEY = ["site", "location_id"]   # location_id is unique only within a site
BD_MAX = 2.65               # g/cm3: quartz particle density, which no bulk density exceeds
STOCK_DEPTH_CM = 30         # the depth Stage 2 harmonises to; coverage is reported

BZE_SITES = 3104            # "3104 Standorte" (Atlas record abstract)
BZE_CRS = "urn:ogc:def:crs:EPSG::25832"
BZE_LAND_USE = {"A", "G", "SO"}  # cropland, grassland, special crops (attribute description)
BZE_ATTRIBUTES = {"pointid", "xcoord", "ycoord", "kv_0_30", "kv_30_100",
                  "landnutzung", "bze_moor", "probenahme_jahr"}
KV_MAX = 1000.0             # Mg/ha in 0-30 cm: a gross-error bound, not a plausibility claim
TOL_M = 0.001               # geometry against xcoord/ycoord, metres
QUANTILES = [5, 25, 50, 75, 95]


@dataclass
class Rule:
    rule: str
    passed: bool
    detail: str


class NotPulled(Exception):
    """A dataset Checkpoint 1 needs is not in raw/ yet."""


# --- config -------------------------------------------------------------------

def _config(name: str, basis: str) -> list[dict]:
    """Rows of config/<name>; a row without its citation or basis is refused
    (provenance is a column, not a comment)."""
    with (CONFIG / name).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if not (r.get(basis) or "").strip():
            raise ValueError(f"config/{name}: a row has no {basis}: {r}")
    return rows


def extent(source_id: str) -> dict:
    """The envelope a source's points must fall in (config/extents.csv)."""
    for r in _config("extents.csv", "basis"):
        if r["source_id"] == source_id:
            return {"crs": r["crs"], **{k: float(r[k]) for k in ("xmin", "xmax", "ymin", "ymax")}}
    raise KeyError(source_id)


def reading(n: int) -> str:
    """What Webster and Oliver (1992) say of a variogram from n points
    (config/variogram_sample_sizes.csv)."""
    for r in _config("variogram_sample_sizes.csv", "citation"):
        if n >= int(r["n_min"]) and (not r["n_max"] or n <= int(r["n_max"])):
            return r["reading"]
    raise ValueError(n)


# --- Potash et al. (2023), figshare ------------------------------------------

def potash() -> tuple[dict, dict[str, bytes]]:
    """The stored article record, and the bytes of the current version of
    each file it lists."""
    doc = F.stored_article()
    if doc is None:
        raise NotPulled("the figshare article record: python -m src.ingest.figshare_client")
    files = {}
    for meta in doc["files"]:
        current = raw_store.current_version(F.file_item(meta["name"]))
        if current is None:
            raise NotPulled(f"{meta['name']}: python -m src.ingest.figshare_client")
        files[meta["name"]] = current.read_bytes()
    return doc, files


def potash_tables(files: dict[str, bytes]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both CSVs, read from the stored bytes; the keys are kept as text."""
    ids = {"site": str, "location_id": str}
    return (pd.read_csv(io.BytesIO(files["locations.csv"]), dtype=ids),
            pd.read_csv(io.BytesIO(files["measurements.csv"]), dtype=ids))


def p1_provenance(doc: dict, files: dict[str, bytes]) -> Rule:
    """The pinned version, under CC BY 4.0, and every file the article lists,
    with the size and MD5 the article states."""
    problems = []
    if doc.get("version") != F.VERSION:
        problems.append(f"version {doc.get('version')}, pinned {F.VERSION}")
    licence = (doc.get("license") or {}).get("name")
    if licence != LICENCE:
        problems.append(f"licence {licence!r}, expected {LICENCE!r}")
    for meta in doc["files"]:
        if not F.file_matches(meta, files[meta["name"]]):
            problems.append(f"{meta['name']}: size or MD5 differs from the article")
    return Rule("P1 provenance", not problems, "; ".join(problems) or
                f"version {F.VERSION}, {licence}; {', '.join(sorted(files))} "
                "match the article's size and MD5")


def p2_schema(loc: pd.DataFrame, meas: pd.DataFrame) -> Rule:
    """Both files carry the columns the figshare description names."""
    problems = []
    for name, df in (("locations.csv", loc), ("measurements.csv", meas)):
        missing = POTASH_COLUMNS[name] - set(df.columns)
        if missing:
            problems.append(f"{name} lacks {sorted(missing)}")
    return Rule("P2 schema", not problems, "; ".join(problems) or
                f"locations.csv {len(loc)} rows, measurements.csv {len(meas)} rows; "
                "every described column present")


def p3_sites(loc: pd.DataFrame, meas: pd.DataFrame) -> Rule:
    """Eight sites, the same eight in both files."""
    s_loc, s_meas = set(loc["site"].dropna()), set(meas["site"].dropna())
    problems = []
    if len(s_loc) != POTASH_SITES:
        problems.append(f"{len(s_loc)} sites in locations.csv, the description says {POTASH_SITES}")
    if s_loc != s_meas:
        problems.append(f"sites in one file only: {sorted(s_loc ^ s_meas)}")
    return Rule("P3 sites", not problems, "; ".join(problems) or
                f"{len(s_loc)} sites in both files: {', '.join(sorted(s_loc))}")


def p4_keys(loc: pd.DataFrame, meas: pd.DataFrame) -> Rule:
    """(site, location_id) is filled and unique in locations.csv, and every
    measurement row has its location."""
    problems = []
    empty = int(loc[KEY].isna().any(axis=1).sum() + meas[KEY].isna().any(axis=1).sum())
    if empty:
        problems.append(f"{empty} rows with an empty site or location_id")
    dup = int(loc.duplicated(KEY).sum())
    if dup:
        problems.append(f"{dup} repeated (site, location_id) in locations.csv")
    known = set(loc[KEY].dropna().itertuples(index=False, name=None))
    orphans = sum(k not in known for k in meas[KEY].dropna().itertuples(index=False, name=None))
    if orphans:
        problems.append(f"{orphans} measurement rows without a location")
    return Rule("P4 keys", not problems, "; ".join(problems) or
                f"{len(loc)} unique locations; all {len(meas)} measurement rows have one")


def p5_coordinates(loc: pd.DataFrame) -> Rule:
    """Every location has numeric X/Y inside the envelope of config/extents.csv."""
    e = extent("potash2023")
    x, y = pd.to_numeric(loc["X"], errors="coerce"), pd.to_numeric(loc["Y"], errors="coerce")
    problems = []
    missing = int((x.isna() | y.isna()).sum())
    if missing:
        problems.append(f"{missing} locations without numeric X and Y")
    have = x.notna() & y.notna()
    outside = int((~(x.between(e["xmin"], e["xmax"]) & y.between(e["ymin"], e["ymax"])) & have).sum())
    if outside:
        problems.append(f"{outside} locations outside the envelope in config/extents.csv")
    return Rule("P5 coordinates", not problems, "; ".join(problems) or
                f"all {len(loc)} inside lon {e['xmin']:g}..{e['xmax']:g}, lat {e['ymin']:g}..{e['ymax']:g}; "
                f"observed lon {x.min():.4f}..{x.max():.4f}, lat {y.min():.4f}..{y.max():.4f}")


def p6_values(meas: pd.DataFrame) -> Rule:
    """Depth segments ordered from a non-negative top; SOCc in [0, 100) % and
    BD in (0, 2.65] g/cm3 where given. An empty value is counted, not failed."""
    top = pd.to_numeric(meas["sample_depth_min"], errors="coerce")
    bottom = pd.to_numeric(meas["sample_depth_max"], errors="coerce")
    soc = pd.to_numeric(meas["SOCc"], errors="coerce")
    bd = pd.to_numeric(meas["BD"], errors="coerce")
    problems = []
    depth = int((top.isna() | bottom.isna() | (top < 0) | (top >= bottom)).sum())
    if depth:
        problems.append(f"{depth} segments with a missing, negative or inverted depth")
    text = int(((meas["SOCc"].notna() & soc.isna()) | (meas["BD"].notna() & bd.isna())).sum())
    if text:
        problems.append(f"{text} rows with a non-numeric SOCc or BD")
    if int(((soc < 0) | (soc >= 100)).sum()):
        problems.append(f"{int(((soc < 0) | (soc >= 100)).sum())} SOCc outside [0, 100) %")
    if int(((bd <= 0) | (bd > BD_MAX)).sum()):
        problems.append(f"{int(((bd <= 0) | (bd > BD_MAX)).sum())} BD outside (0, {BD_MAX}] g/cm3")
    return Rule("P6 values", not problems, "; ".join(problems) or
                f"{len(meas)} segments ordered; SOCc and BD within bounds "
                f"({int(soc.isna().sum())} SOCc and {int(bd.isna().sum())} BD empty)")


def check_potash(doc: dict, files: dict[str, bytes]):
    """P1-P6 in order; a rule that cannot be evaluated stops the rest.
    Returns (rules, locations or None, measurements or None)."""
    rules = [p1_provenance(doc, files)]
    absent = [n for n in POTASH_COLUMNS if n not in files]
    if absent:
        return rules + [Rule("P2 schema", False, f"not in the article: {', '.join(absent)}")], None, None
    loc, meas = potash_tables(files)
    rules.append(p2_schema(loc, meas))
    if not rules[-1].passed:
        return rules, loc, meas
    rules += [p3_sites(loc, meas), p4_keys(loc, meas), p5_coordinates(loc), p6_values(meas)]
    return rules, loc, meas


def local_metres(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Equirectangular metres about the centre: within about 0.5 % across a
    field, which is enough for spacing and extent."""
    lat0 = math.radians(float(np.mean(lat)))
    return np.c_[(lon - np.mean(lon)) * 111_320 * math.cos(lat0),
                 (lat - np.mean(lat)) * 110_574]


def potash_diagnostics(loc: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    """Per site: locations against Webster and Oliver (1992), extent, spacing,
    and how many locations reach 30 cm."""
    bottom = pd.to_numeric(meas["sample_depth_max"], errors="coerce")
    deepest = bottom.groupby([meas["site"], meas["location_id"]]).max()
    rows = []
    for site, g in loc.groupby("site"):
        xy = local_metres(pd.to_numeric(g["X"]).to_numpy(), pd.to_numeric(g["Y"]).to_numpy())
        nn = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(g) > 1 else np.array([np.nan])
        span = xy.max(axis=0) - xy.min(axis=0)
        d = deepest.xs(site, level=0) if site in deepest.index.get_level_values(0) else pd.Series(dtype=float)
        rows.append({
            "site": site,
            "locations": len(g),
            "segments": int((meas["site"] == site).sum()),
            "variogram (Webster & Oliver 1992)": reading(len(g)),
            "extent E x N (m)": f"{span[0]:.0f} x {span[1]:.0f}",
            "median spacing (m)": round(float(np.nanmedian(nn)), 1),
            f"reach {STOCK_DEPTH_CM} cm": f"{int((d >= STOCK_DEPTH_CM).sum())}/{len(d)}",
            "deepest (cm)": float(d.max()) if len(d) else float("nan"),
        })
    return pd.DataFrame(rows)


# --- BZE-LW, Thünen Atlas WFS ------------------------------------------------

def bze_pages() -> list[dict]:
    """The current version of each stored page, following the chain from
    startIndex 0 as the client does."""
    pages, start = [], 0
    while True:
        current = raw_store.current_version(W.page_item(start))
        if current is None:
            break
        doc = json.loads(current.read_bytes())
        pages.append(doc)
        n = len(doc.get("features") or [])
        start += n
        if n == 0 or start >= (doc.get("numberMatched") or 0):
            break
    if not pages:
        raise NotPulled("the BZE-LW site layer: python -m src.ingest.thuenen_wfs_client")
    return pages


def bze_table(pages: list[dict]) -> pd.DataFrame:
    """One row per feature: its attributes, plus the geometry's type and
    coordinates as _geom_type, _gx, _gy."""
    rows = []
    for doc in pages:
        for f in doc.get("features") or []:
            g = f.get("geometry") or {}
            c = (g.get("coordinates") or [None, None]) + [None, None]
            rows.append({**(f.get("properties") or {}),
                         "_geom_type": g.get("type"), "_gx": c[0], "_gy": c[1]})
    return pd.DataFrame(rows)


def b1_completeness(pages: list[dict], t: pd.DataFrame) -> Rule:
    """The pages add up to the total every page states, which is the
    inventory's 3,104 sites, each pointid once, with the attributes read here."""
    problems = []
    missing = BZE_ATTRIBUTES - set(t.columns)
    if missing:
        problems.append(f"attributes missing: {sorted(missing)}")
    stated = {doc.get("numberMatched") for doc in pages}
    if len(stated) != 1:
        problems.append(f"the pages state different totals: {sorted(map(str, stated))}")
    else:
        total = stated.pop()
        if len(t) != total:
            problems.append(f"{len(t)} features in {len(pages)} pages, the server states {total}")
        if total != BZE_SITES:
            problems.append(f"the server states {total}, the inventory {BZE_SITES}")
    if "pointid" in t and int(t["pointid"].duplicated().sum()):
        problems.append(f"{int(t['pointid'].duplicated().sum())} repeated pointid")
    return Rule("B1 completeness", not problems, "; ".join(problems) or
                f"{len(t)} features in {len(pages)} pages = numberMatched = {BZE_SITES}; pointid unique")


def b2_geometry(pages: list[dict], t: pd.DataFrame) -> Rule:
    """EPSG:25832 points whose geometry equals xcoord/ycoord, all inside the
    layer extent published in the Atlas record (config/extents.csv)."""
    e = extent("bzelw2020")
    problems = []
    crs = {((doc.get("crs") or {}).get("properties") or {}).get("name") for doc in pages}
    if crs != {BZE_CRS}:
        problems.append(f"CRS {sorted(map(str, crs))}, expected {BZE_CRS}")
    not_point = int((t["_geom_type"] != "Point").sum())
    if not_point:
        problems.append(f"{not_point} geometries are not points")
    gx, gy = pd.to_numeric(t["_gx"], errors="coerce"), pd.to_numeric(t["_gy"], errors="coerce")
    ax, ay = pd.to_numeric(t["xcoord"], errors="coerce"), pd.to_numeric(t["ycoord"], errors="coerce")
    differ = int((~(((gx - ax).abs() <= TOL_M) & ((gy - ay).abs() <= TOL_M))).sum())
    if differ:
        problems.append(f"{differ} geometries differ from xcoord/ycoord by more than {TOL_M} m, or lack one")
    outside = int((~(gx.between(e["xmin"], e["xmax"]) & gy.between(e["ymin"], e["ymax"]))).sum())
    if outside:
        problems.append(f"{outside} points outside the published extent in config/extents.csv")
    return Rule("B2 CRS and geometry", not problems, "; ".join(problems) or
                f"EPSG:25832 points; geometry = xcoord/ycoord within {TOL_M} m; all inside the published extent")


def b3_values(t: pd.DataFrame) -> Rule:
    """kv_0_30 numeric and within [0, 1000] Mg/ha where given; land use codes
    from the attribute description. An empty value is counted, not failed."""
    kv = pd.to_numeric(t["kv_0_30"], errors="coerce")
    problems = []
    text = int((t["kv_0_30"].notna() & kv.isna()).sum())
    if text:
        problems.append(f"{text} non-numeric kv_0_30")
    out = int(((kv < 0) | (kv > KV_MAX)).sum())
    if out:
        problems.append(f"{out} kv_0_30 outside [0, {KV_MAX:.0f}] Mg/ha")
    codes = set(t["landnutzung"].dropna()) - BZE_LAND_USE
    if codes:
        problems.append(f"land use codes outside {sorted(BZE_LAND_USE)}: {sorted(codes)}")
    return Rule("B3 values", not problems, "; ".join(problems) or
                f"kv_0_30 within [0, {KV_MAX:.0f}] Mg/ha ({int(kv.isna().sum())} empty); "
                f"land use codes in {sorted(BZE_LAND_USE)}")


def check_bze(pages: list[dict], t: pd.DataFrame) -> list[Rule]:
    """B1-B3; without the attributes B1 names, B2 and B3 are not evaluated."""
    rules = [b1_completeness(pages, t)]
    if BZE_ATTRIBUTES - set(t.columns):
        return rules
    return rules + [b2_geometry(pages, t), b3_values(t)]


def bze_diagnostics(t: pd.DataFrame) -> dict:
    """Stocks by land use, the peat flag, the text in kv_30_100, the years,
    and the spacing of the PUBLISHED points (each up to 4 km from its site)."""
    kv = pd.to_numeric(t["kv_0_30"], errors="coerce")
    kv2 = pd.to_numeric(t["kv_30_100"], errors="coerce")
    years = pd.to_numeric(t["probenahme_jahr"], errors="coerce").dropna()
    xy = np.c_[pd.to_numeric(t["_gx"]).to_numpy(), pd.to_numeric(t["_gy"]).to_numpy()]
    nn_km = cKDTree(xy).query(xy, k=2)[0][:, 1] / 1000
    as_text = t["kv_30_100"][t["kv_30_100"].notna() & kv2.isna()]
    return {
        "sites": len(t),
        "kv_0_30 quantiles 5/25/50/75/95 (Mg/ha)": [round(float(v), 1) for v in np.nanpercentile(kv, QUANTILES)],
        "sites by land use": {str(k): int(v) for k, v in t["landnutzung"].value_counts(dropna=False).items()},
        "kv_0_30 median by land use (Mg/ha)": {str(k): round(float(v), 1) for k, v in kv.groupby(t["landnutzung"]).median().items()},
        "bze_moor values": {str(k): int(v) for k, v in t["bze_moor"].value_counts(dropna=False).sort_index().items()},
        "kv_30_100 numeric / text / empty": f"{int(kv2.notna().sum())} / {len(as_text)} / {int(t['kv_30_100'].isna().sum())}",
        "kv_30_100 text values": {str(k): int(v) for k, v in as_text.value_counts().items()},
        "sampling years": f"{int(years.min())}-{int(years.max())}" if len(years) else "none",
        "nearest published neighbour 5/25/50/75/95 (km)": [round(float(v), 2) for v in np.percentile(nn_km, QUANTILES)],
    }


# --- the checkpoint -----------------------------------------------------------

def run() -> tuple[list[Rule], dict]:
    """Every rule, and the diagnostics of each dataset whose rules all passed.
    Raises NotPulled when a dataset is not in raw/."""
    doc, files = potash()
    pages = bze_pages()
    p_rules, loc, meas = check_potash(doc, files)
    t = bze_table(pages)
    b_rules = check_bze(pages, t)
    diagnostics = {
        "potash": potash_diagnostics(loc, meas) if all(r.passed for r in p_rules) else None,
        "bze": bze_diagnostics(t) if all(r.passed for r in b_rules) else None,
        "locations": loc, "measurements": meas, "bze_sites": t,
    }
    return p_rules + b_rules, diagnostics


def report(rules: list[Rule], diagnostics: dict, when: Optional[dt.datetime] = None) -> str:
    """The text Checkpoint 1 prints, and the RUNBOOK records."""
    when = when or dt.datetime.now(dt.timezone.utc)
    passed = sum(r.passed for r in rules)
    lines = [f"Checkpoint 1, run {when:%Y-%m-%dT%H:%M:%SZ}", ""]
    lines += [f"{'PASS' if r.passed else 'FAIL'}  {r.rule}: {r.detail}" for r in rules]
    if diagnostics.get("potash") is not None:
        lines += ["", "Potash et al. (2023), per site (reported, not judged):",
                  diagnostics["potash"].to_string(index=False)]
    if diagnostics.get("bze") is not None:
        lines += ["", "BZE-LW site layer (reported, not judged):"]
        lines += [f"  {k}: {v}" for k, v in diagnostics["bze"].items()]
    verdict = "PASSED" if passed == len(rules) else "FAILED"
    lines += ["", f"Checkpoint 1 {verdict}: {passed}/{len(rules)} rules"]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """python -m src.checks.checkpoint1"""
    if argv:
        print(main.__doc__)
        return 2
    try:
        rules, diagnostics = run()
    except NotPulled as e:
        print(f"NOT PULLED: {e}")
        return 2
    print(report(rules, diagnostics))
    return 0 if all(r.passed for r in rules) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
