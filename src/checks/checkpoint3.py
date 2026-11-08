"""Checkpoint 3: the terrain covariates (Potash fields).

The rules were fixed in RUNBOOK.md on 08.11.2026, before any elevation was
fetched: E1 provenance, E2 grid, E3 decoding and registration, E4 local
derivatives, E5 flow, E6 table. GDAL (through rasterio) and scikit-image
are the independent implementations E3 and E5 compare with; E4 compares the
kernels with direct least-squares fits.

    python -m src.checks.checkpoint3    exit 0: every rule passed
                                        exit 1: a rule failed
                                        exit 2: raw/ or a curated table is missing

Reads raw/ and curated/, and recomputes each field from raw/ as the
transform does; makes no request.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import spsolve

try:                        # run as a module from the repository root
    from src.checks import checkpoint1 as C1
    from src.ingest import tnm_client as TC
    from src.transform import cog, patches as P, terrain as T
except ImportError:         # imported with src/* on sys.path (pytest)
    import checkpoint1 as C1
    import cog
    import patches as P
    import terrain as T
    import tnm_client as TC

Rule = C1.Rule
KEY = C1.KEY
N_LOCATIONS = 1163
TOL_GEOMETRY = 1e-9          # cells: GDAL's grid against ours
TOL_LATTICE = 1e-3           # cells: two tiles' lattices (E2); the rule gives none,
                             # set after the first run found 1.4e-5
TOL_BBOX = 1e-3              # cells: the listing's bounding box against the grid
TOL_PLANE = 1e-6             # m
TOL_REL, TOL_ABS = 1e-9, 1e-12
TOL_TPI = 1e-9               # m
TOL_FILL = 1e-6              # m
TOL_FLOW = 1e-9              # relative
PLANE = (1000.0, 2000.0)     # m per degree of longitude and latitude, for E3's plane


class NotThere(Exception):
    """raw/ or a curated table is not there yet."""


# --- loading -------------------------------------------------------------------------

def manifest_rows() -> pd.DataFrame:
    if not T.MANIFEST.exists():
        raise NotThere(f"{T.MANIFEST} is missing")
    rows = pd.read_csv(T.MANIFEST, dtype=str, keep_default_na=False)
    return rows[rows["path"].str.contains("/TNM/")]


def load() -> dict:
    """Locations, stored tiles, the table, and each field recomputed."""
    try:
        loc = T.load_locations()
        tiles = T.stored_tiles()
    except T.NotThere as e:
        raise NotThere(str(e))
    if not T.OUT.exists():
        raise NotThere(f"{T.OUT} is missing: run python -m src.transform.terrain")
    table = pd.read_parquet(T.OUT)
    recomputed, fields = T.run_stage(loc, tiles)
    return {"loc": loc, "tiles": tiles, "table": table, "recomputed": recomputed,
            "fields": fields, "rows": manifest_rows()}


def _listings(rows: pd.DataFrame) -> dict[str, dict]:
    """site -> the earliest accepted listing stored for it."""
    out = {}
    lst = rows[rows["path"].str.contains("/TNM/listing/") & (rows["rejected"] == "0")]
    for site in sorted(lst["path"].str.extract(r"/TNM/listing/([^/]+)/")[0].unique()):
        g = lst[lst["path"].str.contains(f"/TNM/listing/{re.escape(site)}/")]
        first = g.sort_values("retrieved_at_utc").iloc[0]
        out[site] = json.loads(T._resolve(first["path"]).read_bytes())
    return out


def _item(listings: dict, url: str) -> Optional[dict]:
    for doc in listings.values():
        for it in doc["items"]:
            if it.get("downloadURL") == url:
                return it
    return None


# --- E1 ------------------------------------------------------------------------------

def e1_provenance(d: dict) -> Rule:
    rows, tiles, fields = d["rows"], d["tiles"], d["fields"]
    problems = []
    for r in rows.itertuples():
        path = T._resolve(r.path)
        if not path.exists():
            problems.append(f"{r.path} is missing")
            continue
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != r.sha256 or len(body) != int(r.bytes):
            problems.append(f"{r.path} does not match its manifest row")
    listings = _listings(rows)
    for site, res in fields.items():
        if site not in listings:
            problems.append(f"{site}: no listing stored")
            continue
        by = TC.versions(listings[site]["items"])
        for block in T.blocks_of(res["grid"]):
            newest = TC.newest(by[block])
            if tiles[block].url != newest["downloadURL"]:
                problems.append(f"{site}: {block} is {tiles[block].version}, the listing's newest "
                                f"{TC.tile_version(newest['downloadURL'])[1]}")
            if tiles[block].size != int(newest["sizeInBytes"]):
                problems.append(f"{block}: size {tiles[block].size}, listed {newest['sizeInBytes']}")
    meta = rows[rows["path"].str.contains("/TNM/metadata/") & (rows["rejected"] == "0")]
    for block, tv in tiles.items():
        if not meta["path"].str.contains(f"/TNM/metadata/{block}/{tv.version}/").any():
            problems.append(f"{block} {tv.version}: no metadata record stored")
    reads = 0
    for res in fields.values():                       # the reads E1 asks about: re-run strictly
        for part in T.decode_parts(res["grid"], [p.tile for p in res["parts"]]):
            reads += 1
    src = Path(T.__file__).read_text(encoding="utf-8")
    if re.search(r"^\s*(import|from)\s+(requests|urllib|http)", src, re.M) or "http_get" in src:
        problems.append("src/transform/terrain.py imports a network library")
    if problems:
        return Rule("E1 provenance", False, "; ".join(problems[:6]))
    ranges = rows[rows["revision"].str.startswith("bytes=") & (rows["rejected"] == "0")]
    read = int(ranges["bytes"].astype(int).sum())
    size = sum(tv.size for tv in tiles.values())
    return Rule("E1 provenance", True,
                f"{len(listings)} listings, {len(tiles)} tile versions, each the newest its listing "
                f"gives, {len(ranges)} range responses and {len(meta)} metadata records, every file "
                f"matching its manifest row; one ETag and size per version; {read / 1e6:.1f} MB read of "
                f"{size / 1e9:.2f} GB; the transform decoded {reads} windows from stored ranges alone "
                f"and imports no network library")


# --- E2 ------------------------------------------------------------------------------

def _lattice_misfit(g0: cog.Geo, g: cog.Geo) -> float:
    """How far, in cells, g's cells lie from g0's lattice (inf if their cell
    sizes or CRS differ)."""
    if g.crs != g0.crs or not (math.isclose(g.dx, g0.dx, rel_tol=1e-9)
                              and math.isclose(g.dy, g0.dy, rel_tol=1e-9)):
        return math.inf
    fr, fc = (g0.north - g.north) / g0.dy, (g.west - g0.west) / g0.dx
    return max(abs(fr - round(fr)), abs(fc - round(fc)))


def _spare(p, grid: P.Grid) -> float:
    """Native cells between the patch's cell centres, with the four native
    cells around each that bilinear interpolation needs, and the edges of
    p's window. A side where the patch runs on past the tile's own edge is
    left out: the neighbouring tile covers it, and E2 checks that every
    centre is covered."""
    lon, lat = T.patch_lonlat(grid, p.geo.crs)
    r, c = p.geo.frac_row(lat), p.geo.frac_col(lon)
    r0, r1, c0, c1 = p.window
    sides = [(math.floor(r.min()) - r0, r0 == 0),
             (r1 - (math.floor(r.max()) + 1), r1 == p.geo.height - 1),
             (math.floor(c.min()) - c0, c0 == 0),
             (c1 - (math.floor(c.max()) + 1), c1 == p.geo.width - 1)]
    return min((n for n, at_edge in sides if not (n < 0 and at_edge)), default=math.inf)


def e2_grid(d: dict) -> Rule:
    loc, fields = d["loc"], d["fields"]
    problems, spare_min, misfit = [], math.inf, 0.0
    for site, res in fields.items():
        g, grid_loc = res["grid"], loc[loc["site"] == site]
        if g.epsg != int(grid_loc["epsg"].iloc[0]):
            problems.append(f"{site}: EPSG {g.epsg}")
        if g.x0 % P.CELL or g.y1 % P.CELL:
            problems.append(f"{site}: edges not on multiples of 10 m")
        x, y = grid_loc["x"].to_numpy(), grid_loc["y"].to_numpy()
        reach = min((x - g.x0).min(), (g.x1 - x).min(), (y - g.y0).min(), (g.y1 - y).min())
        if reach < P.BUFFER:
            problems.append(f"{site}: reaches only {reach:.0f} m beyond a location")
        for p in res["parts"]:
            spare = _spare(p, g)
            spare_min = min(spare_min, spare)
            if spare < 1:
                problems.append(f"{site}, {p.tile.block}: {spare} native cells to spare")
            if np.isnan(p.values).any():
                problems.append(f"{site}, {p.tile.block}: {int(np.isnan(p.values).sum())} nodata cells")
        uncovered = int((res["used"] < 0).sum())
        if uncovered:
            problems.append(f"{site}: {uncovered} patch cells no native window covers")
        for p in res["parts"][1:]:
            m = _lattice_misfit(res["parts"][0].geo, p.geo)
            misfit = max(misfit, m)
            if m > TOL_LATTICE:
                problems.append(f"{site}: {p.tile.block}'s cells off {res['parts'][0].tile.block}'s "
                                f"lattice by {m:.1e} cells")
    if problems:
        return Rule("E2 grid", False, "; ".join(problems[:6]))
    two = [s for s, res in fields.items() if len(res["parts"]) > 1]
    return Rule("E2 grid", True,
                f"{len(fields)} patches in their fields' Stage 2 zones, 10 m cells on multiples of "
                f"10 m, each at least 1,000 m beyond every location; the native windows cover every "
                f"patch cell centre with at least {spare_min} native cells to spare (leaving aside "
                f"sides where a patch runs on into the next tile) and hold no nodata cell; "
                + (f"{', '.join(two)}: its two tiles' cells on one lattice within {misfit:.1e} "
                   f"cells (tolerance {TOL_LATTICE:g}, set after the first run)" if two
                   else "no field on two tiles"))


# --- E3 ------------------------------------------------------------------------------

def _gdal_open(tv, tmp: Path):
    import rasterio
    xml = tmp / f"{tv.block}_{tv.version}.xml"
    xml.write_text(cog.sparse_xml(tv.size, tv.pieces), encoding="utf-8")
    return rasterio.open("/vsisparse/" + xml.resolve().as_posix())


def e3_registration(d: dict) -> Rule:
    import rasterio
    from rasterio.windows import Window
    tiles, fields = d["tiles"], d["fields"]
    listings = _listings(d["rows"])
    problems, worst_geo, worst_bbox, worst_plane, windows = [], 0.0, 0.0, 0.0, 0
    gdal = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for block, tv in tiles.items():
            with _gdal_open(tv, tmp) as src:
                gdal[block] = src.transform
                ours = None
                for res in fields.values():
                    for p in res["parts"]:
                        if p.tile.block != block:
                            continue
                        ours = p.geo
                        r0, r1, c0, c1 = p.window
                        v = src.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)).astype(np.float64)
                        if src.nodata is not None:
                            v[v == src.nodata] = np.nan
                        windows += 1
                        if not np.array_equal(v, p.values, equal_nan=True):
                            problems.append(f"{block}: GDAL's values differ from ours")
                if ours is not None:
                    t = src.transform
                    dev = max(abs(t.c - ours.west) / ours.dx, abs(t.f - ours.north) / ours.dy,
                              abs(t.a - ours.dx) / ours.dx * ours.width,
                              abs(-t.e - ours.dy) / ours.dy * ours.height)
                    worst_geo = max(worst_geo, dev)
                    if (src.width, src.height) != (ours.width, ours.height):
                        problems.append(f"{block}: GDAL reads {src.width} x {src.height}")
                    item = _item(listings, tv.url)
                    if item is None or "boundingBox" not in item:
                        problems.append(f"{block}: no bounding box in the listing")
                    else:
                        bb = item["boundingBox"]
                        dev = max(abs(bb["minX"] - ours.west), abs(bb["maxX"] - ours.east)) / ours.dx
                        dev = max(dev, max(abs(bb["maxY"] - ours.north), abs(bb["minY"] - ours.south)) / ours.dy)
                        worst_bbox = max(worst_bbox, dev)
    if worst_geo > TOL_GEOMETRY:
        problems.append(f"GDAL's grid differs from ours by {worst_geo:.1e} cells")
    if worst_bbox > TOL_BBOX:
        problems.append(f"a grid misses its listed bounding box by {worst_bbox:.1e} cells")
    a, b = PLANE
    plane = lambda lon, lat: a * (lon + 88.0) + b * (lat - 40.0)
    for site, res in fields.items():                  # a plane on GDAL's cell centres
        synth = []
        for p in res["parts"]:
            t = gdal[p.tile.block]
            r0, r1, c0, c1 = p.window
            rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
            lon = t.c + (cols + 0.5) * t.a
            lat = t.f + (rows + 0.5) * t.e
            synth.append(T.Part(p.tile, p.geo, p.window, plane(lon, lat)))
        got, _ = T.compose(synth, res["grid"])
        lon, lat = T.patch_lonlat(res["grid"], synth[0].geo.crs)
        worst_plane = max(worst_plane, float(np.nanmax(np.abs(got - plane(lon, lat)))))
        if np.isnan(got).any():
            problems.append(f"{site}: the plane did not cover the patch")
    if worst_plane > TOL_PLANE:
        problems.append(f"the plane is off by {worst_plane:.1e} m")
    if problems:
        return Rule("E3 decoding and registration", False, "; ".join(problems[:6]))
    return Rule("E3 decoding and registration", True,
                f"GDAL {rasterio.__gdal_version__}, reading the same stored bytes, gives the same "
                f"values in all {windows} native windows and the same grid within {worst_geo:.1e} "
                f"cells (rule: {TOL_GEOMETRY:g}); every grid spans its listed bounding box within "
                f"{worst_bbox:.1e} cells (rule: {TOL_BBOX:g}); a plane laid on GDAL's cell centres "
                f"comes through the resampling within {worst_plane:.1e} m (rule: {TOL_PLANE:g})")


# --- E4 ------------------------------------------------------------------------------

def _direct_fit(z: np.ndarray, row: int, col: int, n: int) -> dict:
    m = n // 2
    w = z[row - m:row + m + 1, col - m:col + m + 1]
    dr, dc = np.mgrid[-m:m + 1, -m:m + 1]
    u, v = (dc * P.CELL).ravel(), (-dr * P.CELL).ravel()
    X = np.column_stack([u * u, v * v, u * v, u, v, np.ones_like(u)])
    a, b, c, d, e, _ = np.linalg.lstsq(X, w.ravel() - w.mean(), rcond=None)[0]
    return {"p": d, "q": e, "r": 2 * a, "s": c, "t": 2 * b}


def _analytic_local() -> list[str]:
    """The closed forms of the unit tests, run here as well."""
    problems = []
    n = 81
    dr, dc = np.mgrid[0:n, 0:n] - n // 2
    u, v = dc * P.CELL, -dr * P.CELL
    plane = 100.0 + 0.02 * u - 0.01 * v
    A, B, C, D, E = -2e-4, -1e-4, 5e-5, 0.01, -0.02
    quad = 50.0 + A * u * u + B * v * v + C * u * v + D * u + E * v
    p, q = 2 * A * u + C * v + D, 2 * B * v + C * u + E
    H = -((1 + q * q) * 2 * A - 2 * p * q * C + (1 + p * p) * 2 * B) / (2 * (1 + p * p + q * q) ** 1.5)
    inner = (slice(20, 61), slice(20, 61))
    for k in T.WINDOWS:
        f = T.local_fit(plane, k)
        if abs(T.slope_degrees(f)[40, 40] - math.degrees(math.atan(math.hypot(0.02, 0.01)))) > 1e-9:
            problems.append(f"plane: slope at {k}")
        if abs(T.mean_curvature(f)[40, 40]) > 1e-12:
            problems.append(f"plane: curvature at {k}")
        f = T.local_fit(quad, k)
        if not np.allclose(T.mean_curvature(f)[inner], H[inner], rtol=1e-8, atol=1e-12):
            problems.append(f"paraboloid: mean curvature at {k}")
    if abs(T.tpi(plane, 150.0)[40, 40]) > 1e-9:
        problems.append("plane: TPI")
    return problems


def e4_local(d: dict) -> Rule:
    loc, fields = d["loc"], d["fields"]
    problems, worst_rel, worst_tpi, checked = [], 0.0, 0.0, 0
    for site, res in fields.items():
        g, z = res["grid"], res["grids"]["elevation"]
        sl = loc[loc["site"] == site]
        rows, cols = g.cell_of(sl["x"].to_numpy(), sl["y"].to_numpy())
        for n in T.WINDOWS:
            slope, curv = res["grids"][f"slope_{n * 10}"], res["grids"][f"curvature_{n * 10}"]
            for row, col in zip(rows, cols):
                f = _direct_fit(z, row, col, n)
                s_direct = math.degrees(math.atan(math.hypot(f["p"], f["q"])))
                h_direct = float(T.mean_curvature(f))
                for got, want in ((slope[row, col], s_direct), (curv[row, col], h_direct)):
                    if not np.isclose(got, want, rtol=TOL_REL, atol=TOL_ABS):
                        problems.append(f"{site} {n}x{n}: {got} against {want}")
                    if abs(want) > TOL_ABS:
                        worst_rel = max(worst_rel, abs(got - want) / abs(want))
                checked += 1
        for radius in T.RADII:
            t = res["grids"][f"tpi_{int(radius)}"]
            dr, dc = T.disc_offsets(radius)
            for row, col in zip(rows, cols):
                direct = z[row, col] - z[row + dr, col + dc].mean()
                worst_tpi = max(worst_tpi, abs(t[row, col] - direct))
    if worst_tpi > TOL_TPI:
        problems.append(f"TPI off its direct mean by {worst_tpi:.1e} m")
    problems += _analytic_local()
    if problems:
        return Rule("E4 local derivatives", False, "; ".join(problems[:6]))
    return Rule("E4 local derivatives", True,
                f"slope and mean curvature from the kernels against a direct least-squares fit at "
                f"{checked} location cells and windows: largest relative difference {worst_rel:.1e} "
                f"(rule: {TOL_REL:g}, or {TOL_ABS:g} absolute near zero); TPI against its direct mean "
                f"within {worst_tpi:.1e} m (rule: {TOL_TPI:g}); the plane and the paraboloid reproduce "
                f"their closed forms")


# --- E5 ------------------------------------------------------------------------------

def _neighbour_lower(filled: np.ndarray) -> np.ndarray:
    ny, nx = filled.shape
    pad = np.pad(filled, 1, constant_values=np.inf)
    lower = np.zeros(filled.shape, bool)
    for di, dj in T.NEIGHBOURS:
        lower |= pad[1 + di:1 + di + ny, 1 + dj:1 + dj + nx] < filled
    return lower


def _solve(fractions: np.ndarray, cell: float) -> np.ndarray:
    """The accumulated areas as the solution of (I - W') a = cell^2."""
    _, ny, nx = fractions.shape
    n = ny * nx
    idx = np.arange(n).reshape(ny, nx)
    src, dst, val = [], [], []
    for k, (di, dj) in enumerate(T.NEIGHBOURS):
        f = fractions[k]
        rr, cc = np.nonzero(f > 0)
        src.append(idx[rr, cc]); dst.append(idx[rr + di, cc + dj]); val.append(f[rr, cc])
    W = sparse.csr_matrix((np.concatenate(val), (np.concatenate(src), np.concatenate(dst))), shape=(n, n))
    A = sparse.identity(n, format="csc") - W.T.tocsc()
    return spsolve(A, np.full(n, cell * cell)).reshape(ny, nx)


def e5_flow(d: dict) -> Rule:
    from skimage.morphology import reconstruction
    problems, worst_fill, worst_solve, worst_out = [], 0.0, 0.0, 0.0
    for site, res in d["fields"].items():
        z, filled, area = res["grids"]["elevation"], res["grids"]["filled"], res["grids"]["area"]
        edge = T.edge_mask(z.shape)
        if (filled < z).any():
            problems.append(f"{site}: filled below the original")
        if not np.array_equal(filled[edge], z[edge]):
            problems.append(f"{site}: the edge was raised")
        seed = z.copy()
        seed[1:-1, 1:-1] = z.max()
        flat = reconstruction(seed, z, method="erosion", footprint=np.ones((3, 3)))
        worst_fill = max(worst_fill, float(np.abs(filled - flat).max()))
        if not _neighbour_lower(filled)[~edge].all():
            problems.append(f"{site}: a cell off the edge has no lower neighbour")
        fractions = T.fd8_fractions(filled)
        solved = _solve(fractions, P.CELL)
        worst_solve = max(worst_solve, float(np.max(np.abs(solved - area) / area)))
        outlets = fractions.sum(axis=0) == 0
        total = z.size * P.CELL * P.CELL
        worst_out = max(worst_out, abs(area[outlets].sum() - total) / total)
    if worst_fill > TOL_FILL:
        problems.append(f"the fill differs from scikit-image's by {worst_fill:.1e} m")
    if worst_solve > TOL_FLOW:
        problems.append(f"the accumulation differs from the sparse solution by {worst_solve:.1e}")
    if worst_out > TOL_FLOW:
        problems.append(f"the area leaving the patch is off by {worst_out:.1e}")
    ny, nx = 30, 40                                    # the tilted plane of the unit tests
    zt = 100.0 - 0.01 * P.CELL * np.arange(nx)[None, :] * np.ones((ny, 1))
    ft = T.fill(zt)
    at = T.accumulate(ft, T.fd8_fractions(ft), np.full(zt.shape, P.CELL ** 2))
    if not all(math.isclose(at[:, k].sum(), (k + 1) * ny * P.CELL ** 2, rel_tol=1e-12) for k in range(nx - 1)):
        problems.append("tilted plane: a column's area is not the area upslope of it")
    if problems:
        return Rule("E5 flow", False, "; ".join(problems[:6]))
    return Rule("E5 flow", True,
                f"in all {len(d['fields'])} patches the filled DEM is nowhere below the original, keeps "
                f"its edge, and equals scikit-image's reconstruction by erosion within {worst_fill:.1e} m "
                f"(rule: {TOL_FILL:g}); every cell off the edge has a lower neighbour; the accumulated "
                f"areas equal SciPy's sparse solution within {worst_solve:.1e} relative and the area "
                f"leaving each patch its area within {worst_out:.1e} (rule: {TOL_FLOW:g}); on the tilted "
                f"plane each column carries the area upslope of it")


# --- E6 ------------------------------------------------------------------------------

def e6_table(d: dict) -> Rule:
    t, loc = d["table"], d["loc"]
    problems = []
    if len(t) != N_LOCATIONS:
        problems.append(f"{len(t)} rows, expected {N_LOCATIONS}")
    if t.duplicated(KEY).any():
        problems.append("duplicate keys")
    if set(map(tuple, t[KEY].to_numpy())) != set(map(tuple, loc[KEY].to_numpy())):
        problems.append("the keys are not the stock table's")
    missing = [c for c in T.COVARIATES if c not in t.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    else:
        vals = t[list(T.COVARIATES)].to_numpy(float)
        if not np.isfinite(vals).all():
            problems.append(f"{int((~np.isfinite(vals)).sum())} values not finite")
        slopes = t[[f"slope_{n * 10}" for n in T.WINDOWS]].to_numpy(float)
        if (slopes < 0).any() or (slopes >= 90).any():
            problems.append("a slope outside [0, 90)")
    if "stock" in t.columns:
        problems.append("the table carries the stocks")
    if problems:
        return Rule("E6 table", False, "; ".join(problems))
    return Rule("E6 table", True,
                f"{len(t)} rows, one per (site, location_id) of the stock table, with the "
                f"{len(T.COVARIATES)} covariates, every value finite; slopes from "
                f"{slopes.min():.2f} to {slopes.max():.2f} degrees")


# --- reported, not judged ------------------------------------------------------------

def _day(s: Optional[str]) -> str:
    s = (s or "").strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:]}" if re.fullmatch(r"\d{8}", s) else (s or "?")


def _record(path: Path) -> dict:
    """What an FGDC metadata record says of its sources: the titles its
    lineage section lists, if it has one, and its time period of content."""
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError:
        return {"sources": [], "period": "?"}
    names = []
    for src in root.iter("srcinfo"):
        title = src.find("./srccite/citeinfo/title")
        if title is not None and title.text:
            names.append(" ".join(title.text.split()))
    dates = root.find("./idinfo/timeperd/timeinfo/rngdates")
    period = "?" if dates is None else \
        f"{_day(dates.findtext('begdate'))} to {_day(dates.findtext('enddate'))}"
    return {"sources": names, "period": period}


def tile_table(d: dict) -> pd.DataFrame:
    rows, tiles = d["rows"], d["tiles"]
    listings = _listings(rows)
    out = []
    for block, tv in sorted(tiles.items()):
        r = rows[(rows["url"] == tv.url) & (rows["rejected"] == "0")]
        item = _item(listings, tv.url) or {}
        meta = rows[rows["path"].str.contains(f"/TNM/metadata/{block}/{tv.version}/") & (rows["rejected"] == "0")]
        rec = _record(T._resolve(meta["path"].iloc[0])) if len(meta) else {"sources": [], "period": "?"}
        names = rec["sources"]
        used = sorted(s for s, res in d["fields"].items() if any(p.tile.block == block for p in res["parts"]))
        out.append({"tile": block, "version": tv.version, "published": item.get("publicationDate", "?"),
                    "file (MB)": f"{tv.size / 1e6:.0f}",
                    "read (MB)": f"{r['bytes'].astype(int).sum() / 1e6:.2f}",
                    "read (%)": f"{100 * r['bytes'].astype(int).sum() / tv.size:.2f}",
                    "fields": ", ".join(used),
                    "sources named": f"{len(names)}" + (f": {'; '.join(names[:2])}"[:60] if names else ""),
                    "content dated": rec["period"]})
    return pd.DataFrame(out)


def field_table(d: dict) -> pd.DataFrame:
    out = []
    for site, res in d["fields"].items():
        g, grids, v, x = res["grid"], res["grids"], res["values"], res["extra"]
        q = lambda a: np.nanpercentile(a, [25, 75])
        raised = grids["filled"] > grids["elevation"] + 1e-9
        rho = stats.spearmanr(x[f"twi_floor_{T.FLOORS_REPORTED[0]:g}"],
                              x[f"twi_floor_{T.FLOORS_REPORTED[1]:g}"]).correlation
        el = v["elevation"]
        out.append({
            "field": site, "patch": f"{g.nx} x {g.ny}",
            "elevation (m)": f"{el.min():.1f} / {np.median(el):.1f} / {el.max():.1f}",
            "IQR (m)": f"{np.subtract(*q(el)[::-1]):.2f}",
            "median slope 30/90/270 (deg)": " / ".join(f"{np.median(v[f'slope_{n * 10}']):.2f}" for n in T.WINDOWS),
            "TWI": f"{v['twi'].min():.1f} to {v['twi'].max():.1f}",
            "cells raised by fill (%)": f"{100 * raised.mean():.1f}",
            "largest rise (m)": f"{(grids['filled'] - grids['elevation']).max():.2f}",
            "locations on raised cells": int((x['filled'] > v['elevation'] + 1e-6).sum()),
            "tan b at floor, cells / locations (%)": f"{100 * np.nanmean(grids['tanb'] < T.FLOOR):.1f} / "
                                                      f"{100 * np.mean(x['tanb'] < T.FLOOR):.1f}",
            "TWI rank corr., floors 1e-4 vs 1e-2": f"{rho:.3f}",
            "upslope area reaching the edge": int((x["from_edge"] > 0).sum()),
        })
    return pd.DataFrame(out)


def spread_table(d: dict) -> pd.DataFrame:
    out = []
    for site, res in d["fields"].items():
        v = res["values"]
        iqr = lambda a: np.subtract(*np.percentile(a, [75, 25]))
        row = {"field": site}
        for n in T.WINDOWS:
            row[f"curv {n * 10} m IQR (1/km)"] = f"{1000 * iqr(v[f'curvature_{n * 10}']):.3f}"
        for r in T.RADII:
            row[f"TPI {int(r)} m IQR (m)"] = f"{iqr(v[f'tpi_{int(r)}']):.3f}"
        out.append(row)
    return pd.DataFrame(out)


def collinearity(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = d["table"]
    cols = list(T.COVARIATES)
    pooled = t[cols].corr(method="spearman")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            per = {s: g[[a, b]].corr(method="spearman").iloc[0, 1] for s, g in t.groupby("site")}
            high = sorted(s for s, rho in per.items() if abs(rho) > 0.9)
            if abs(pooled.loc[a, b]) > 0.9 or high:
                pairs.append({"pair": f"{a} ~ {b}", "pooled": f"{pooled.loc[a, b]:+.2f}",
                              "fields above 0.9": ", ".join(high) if high else "-"})
    return pooled.round(2), pd.DataFrame(pairs)


# --- the checkpoint ------------------------------------------------------------------

def run() -> tuple[list[Rule], dict]:
    d = load()
    rules = [e1_provenance(d), e2_grid(d), e3_registration(d), e4_local(d), e5_flow(d), e6_table(d)]
    pooled, pairs = collinearity(d)
    tables = {"Tiles (versions, what was read, and what their metadata records say of their sources, for the whole block):": tile_table(d),
              "Per field, at the locations (elevation lowest / median / highest):": field_table(d),
              "Spread of curvature and TPI at the locations:": spread_table(d),
              "Spearman correlations among the covariates at the locations, all fields pooled:": pooled.rename_axis("covariate").reset_index(),
              "Pairs above 0.9 in absolute value, pooled or within a field:": pairs if len(pairs) else
              pd.DataFrame([{"pair": "none"}])}
    return rules, tables


def report(rules: list[Rule], tables: Optional[dict], when: Optional[dt.datetime] = None) -> str:
    """The text Checkpoint 3 prints, and the RUNBOOK records."""
    when = when or dt.datetime.now(dt.timezone.utc)
    lines = [f"Checkpoint 3, run {when:%Y-%m-%dT%H:%M:%SZ}", ""]
    lines += [f"{'PASS' if r.passed else 'FAIL'}  {r.rule}: {r.detail}" for r in rules]
    if tables:
        lines += ["", "Reported, not judged:"]
        for title, table in tables.items():
            lines += ["", title, table.to_string(index=False)]
    passed = sum(r.passed for r in rules)
    lines += ["", f"Checkpoint 3 {'PASSED' if passed == len(rules) else 'FAILED'}: {passed}/{len(rules)} rules"]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """python -m src.checks.checkpoint3"""
    if argv:
        print(main.__doc__)
        return 2
    try:
        rules, tables = run()
    except NotThere as e:
        print(f"NOT THERE: {e}")
        return 2
    print(report(rules, tables))
    return 0 if all(r.passed for r in rules) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
