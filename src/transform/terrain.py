"""Stage 3: the terrain covariates at the Potash locations (RUNBOOK, Stage 3).

    python -m src.transform.terrain   writes curated/potash_terrain.parquet
                                      and curated/terrain/<site>.npz

For each field:
  D2  The native cells of every tile under the patch are decoded from the
      ranges the client stored (src/transform/cog.py) and resampled
      bilinearly, each on its own tile's grid, to the patch's 10 m cell
      centres, which PROJ takes from UTM to the tiles' longitude and
      latitude; where two tiles cover a centre (IL-RS straddles 40 N), the
      newer version's value is used.
  D4  A quadratic fitted by least squares to every n x n window, n = 3, 9 and
      27, as fixed convolution kernels: slope (degrees) and mean curvature
      (1/m, positive on convex ground).
  D5  TPI: elevation minus the mean of the cells whose centres lie within
      50, 150 and 450 m, the cell itself excluded.
  D6  TWI = ln(a / tan b): the DEM filled by Priority-Flood with epsilon
      (Barnes et al. 2014); FD8 with an exponent of 1, each cell's flow
      shared among its lower neighbours in proportion to the slope towards
      each, leaving the patch where a cell has no lower neighbour; a, the
      area draining through a cell, itself included, over the cell width;
      tan b, the 3 x 3 fit's slope on the filled DEM, at least 0.001.
  D3  Every grid interpolated bilinearly at the locations.

Reads only the stock table's keys and coordinates (D7) and the responses
stored in raw/TNM/; makes no request.
"""
from __future__ import annotations

import hashlib
import heapq
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

try:                        # run as a module from the repository root
    from src.transform import cog, patches as P
except ImportError:         # imported with src/transform on sys.path (pytest)
    import cog
    import patches as P

ROOT = Path(__file__).resolve().parents[2]
STOCKS = ROOT / "curated" / "potash_stocks_0_30.parquet"
MANIFEST = ROOT / "raw" / "manifest.csv"
OUT = ROOT / "curated" / "potash_terrain.parquet"
OUT_GRIDS = ROOT / "curated" / "terrain"

CELL = P.CELL
WINDOWS = (3, 9, 27)                     # cells: 30, 90 and 270 m
RADII = (50.0, 150.0, 450.0)             # m
FLOOR = 0.001                            # tan b where the filled DEM is level
FLOORS_REPORTED = (1e-4, 1e-2)           # the floor's influence, reported
COVARIATES = (["elevation"] + [f"slope_{n * 10}" for n in WINDOWS]
              + [f"curvature_{n * 10}" for n in WINDOWS]
              + [f"tpi_{int(r)}" for r in RADII] + ["twi"])
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


class NotThere(Exception):
    """The stock table or the stored tile parts are not there yet."""


# --- the stored tiles ----------------------------------------------------------------

@dataclass
class TileVersion:
    """One tile version as stored: its ranges and where each came from."""
    url: str
    block: str
    version: str
    size: int
    etag: str
    ranges: dict = field(default_factory=dict)        # first byte -> bytes
    pieces: list = field(default_factory=list)        # (first byte, path, length)

    def file(self) -> cog.RangeFile:
        return cog.RangeFile(self.ranges, self.size)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def stored_tiles(manifest: Path = None) -> dict[str, TileVersion]:
    """block -> the tile version whose ranges the manifest records, each
    range checked against its SHA-256."""
    manifest = MANIFEST if manifest is None else manifest
    if not manifest.exists():
        raise NotThere(f"{manifest} is missing: run python -m src.ingest.tnm_client")
    rows = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    out: dict[str, TileVersion] = {}
    for row in rows.itertuples():
        if row.rejected != "0" or not cog.TILE_URL.search(row.url):
            continue
        block, version = cog.tile_version(row.url)
        rev = cog.parse_revision(row.revision)
        tv = out.get(block)
        if tv is None:
            tv = out[block] = TileVersion(row.url, block, version, rev["size"], rev["etag"])
        elif tv.url != row.url:
            raise ValueError(f"two versions of {block} in the manifest")
        if (rev["size"], rev["etag"]) != (tv.size, tv.etag):
            raise ValueError(f"{row.url}: ranges disagree on the file's size or ETag")
        path = _resolve(row.path)
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != row.sha256:
            raise ValueError(f"{row.path} does not match its manifest row")
        if len(body) != rev["last"] - rev["first"] + 1:
            raise ValueError(f"{row.path}: {len(body)} bytes for range {row.revision}")
        tv.ranges[rev["first"]] = body
        tv.pieces.append((rev["first"], path, len(body)))
    if not out:
        raise NotThere("no tile ranges in the manifest: run python -m src.ingest.tnm_client")
    return out


def load_locations(path: Path = None) -> pd.DataFrame:
    """The stock table's keys and coordinates, never its stocks (D7)."""
    path = STOCKS if path is None else path
    if not path.exists():
        raise NotThere(f"{path} is missing: run python -m src.transform.stocks")
    return pd.read_parquet(path, columns=P.LOCATION_COLUMNS)


# --- D2: native cells to the patch ---------------------------------------------------

@dataclass
class Part:
    """One tile's share of a patch: its grid, its window and its values."""
    tile: TileVersion
    geo: cog.Geo
    window: tuple
    values: np.ndarray


def decode_parts(grid: P.Grid, tiles: list[TileVersion]) -> list[Part]:
    """The native window of each tile under the patch, oldest version first."""
    parts = []
    for tv in sorted(tiles, key=lambda t: (t.version, t.block)):
        f = tv.file()
        with tifffile.TiffFile(f) as tif:
            page = tif.pages[0]
            geo = cog.geometry(page)
            window = cog.window_for(grid, geo)
            if window is None:
                continue
            parts.append(Part(tv, geo, window, cog.decode_window(page, geo, f, window)))
    if not parts:
        raise ValueError(f"{grid.site}: no stored tile covers the patch")
    return parts


def patch_lonlat(grid: P.Grid, crs: str) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of every patch cell centre."""
    x, y = grid.centres()
    X, Y = np.meshgrid(x, y)
    lon, lat = P.transformer(grid.epsg, crs).transform(X, Y)
    return np.asarray(lon), np.asarray(lat)


def resample(part: Part, grid: P.Grid) -> np.ndarray:
    """Bilinear interpolation of a part's native cells at the patch's cell
    centres, on the part's own grid; NaN where the four native cells around
    a centre are not all in the part's window."""
    lon, lat = patch_lonlat(grid, part.geo.crs)
    r0, _, c0, _ = part.window
    return ndimage.map_coordinates(part.values,
                                   [part.geo.frac_row(lat) - r0, part.geo.frac_col(lon) - c0],
                                   order=1, mode="constant", cval=np.nan, prefilter=False)


def compose(parts: list[Part], grid: P.Grid) -> tuple[np.ndarray, np.ndarray]:
    """The patch's DEM, and for every cell the index in `parts` of the
    version that gave it (-1 where none covers it). Each part is resampled
    on its own tile's grid, and the newest version is taken wherever its
    native cells surround a centre. Two tiles' lattices meet to about 1e-5
    of a cell, not exactly (their tiepoints and cell sizes are stored to
    about 1e-9), so moving one tile's cells onto the other's lattice would
    shift them by that much."""
    dem = np.full((grid.ny, grid.nx), np.nan)
    used = np.full((grid.ny, grid.nx), -1, dtype=np.int8)
    for i, p in enumerate(parts):                # oldest first, so the newest ends on top
        v = resample(p, grid)
        ok = np.isfinite(v)
        dem[ok], used[ok] = v[ok], i
    return dem, used


# --- D4 and D5: local derivatives ----------------------------------------------------

def quadratic_kernels(n: int, cell: float = CELL) -> np.ndarray:
    """(6, n, n): the least-squares coefficients a..f of
    z = a u^2 + b v^2 + c u v + d u + e v + f over an n x n window, as
    kernels for ndimage.correlate; u east, v north, in metres."""
    m = n // 2
    dr, dc = np.mgrid[-m:m + 1, -m:m + 1]
    u, v = (dc * cell).ravel(), (-dr * cell).ravel()
    X = np.column_stack([u * u, v * v, u * v, u, v, np.ones_like(u)])
    return np.linalg.pinv(X).reshape(6, n, n)


def local_fit(z: np.ndarray, n: int, cell: float = CELL) -> dict[str, np.ndarray]:
    """p, q (first derivatives east and north) and r, s, t (second
    derivatives) of the fitted quadratic at every cell; NaN where the window
    leaves the patch."""
    K = quadratic_kernels(n, cell)
    zc = z - np.nanmean(z)                        # derivatives do not change
    a, b, c, d, e = (ndimage.correlate(zc, K[k], mode="constant", cval=np.nan) for k in range(5))
    return {"p": d, "q": e, "r": 2.0 * a, "s": c, "t": 2.0 * b}


def slope_degrees(fit: dict) -> np.ndarray:
    return np.degrees(np.arctan(np.hypot(fit["p"], fit["q"])))


def mean_curvature(fit: dict) -> np.ndarray:
    """The mean of the principal curvatures, 1/m, positive on convex ground."""
    p, q, r, s, t = (fit[k] for k in "pqrst")
    return -((1 + q * q) * r - 2 * p * q * s + (1 + p * p) * t) / (2 * (1 + p * p + q * q) ** 1.5)


def disc_offsets(radius: float, cell: float = CELL) -> tuple[np.ndarray, np.ndarray]:
    """Row and column offsets of the cells whose centres lie within radius
    of a cell's centre, the cell itself excluded."""
    m = int(radius // cell)
    dr, dc = np.mgrid[-m:m + 1, -m:m + 1]
    keep = ((dr * cell) ** 2 + (dc * cell) ** 2 <= radius ** 2) & ~((dr == 0) & (dc == 0))
    return dr[keep], dc[keep]


def tpi(z: np.ndarray, radius: float, cell: float = CELL) -> np.ndarray:
    """Elevation minus the mean elevation within radius (D5)."""
    dr, dc = disc_offsets(radius, cell)
    m = int(radius // cell)
    K = np.zeros((2 * m + 1, 2 * m + 1))
    K[dr + m, dc + m] = 1.0 / len(dr)
    zc = z - np.nanmean(z)
    return zc - ndimage.correlate(zc, K, mode="constant", cval=np.nan)


# --- D6: wetness ---------------------------------------------------------------------

def fill(z: np.ndarray) -> np.ndarray:
    """Priority-Flood with epsilon (Barnes et al. 2014): flooding inwards
    from the patch's edge, a cell no higher than the one it is reached from
    is raised to the next float above it, so that every cell drains."""
    ny, nx = z.shape
    h = [float(v) for v in z.ravel()]
    seen = bytearray(ny * nx)
    heap: list = []
    for i in range(ny):
        for j in range(nx):
            if i in (0, ny - 1) or j in (0, nx - 1):
                k = i * nx + j
                seen[k] = 1
                heap.append((h[k], k))
    heapq.heapify(heap)
    while heap:
        hc, k = heapq.heappop(heap)
        i, j = divmod(k, nx)
        for di, dj in NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx:
                kn = a * nx + b
                if not seen[kn]:
                    seen[kn] = 1
                    if h[kn] <= hc:
                        h[kn] = math.nextafter(hc, math.inf)
                    heapq.heappush(heap, (h[kn], kn))
    return np.array(h).reshape(ny, nx)


def fd8_fractions(filled: np.ndarray, cell: float = CELL) -> np.ndarray:
    """(8, ny, nx): the share of each cell's flow sent to each neighbour,
    in proportion to the slope towards it (FD8 with an exponent of 1). A
    cell with no lower neighbour inside the patch sends nothing: its flow
    leaves the patch."""
    ny, nx = filled.shape
    pad = np.pad(filled, 1, constant_values=np.nan)
    s = np.zeros((8, ny, nx))
    for k, (di, dj) in enumerate(NEIGHBOURS):
        nb = pad[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        drop = filled - nb
        s[k] = np.where(drop > 0, drop / (cell * math.hypot(di, dj)), 0.0)
    total = s.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, s / total, 0.0)


def accumulate(filled: np.ndarray, fractions: np.ndarray, source: np.ndarray) -> np.ndarray:
    """What each cell receives from upslope plus its own source, routed from
    the highest cell down."""
    ny, nx = filled.shape
    acc = [float(v) for v in source.ravel()]
    fr = [f.ravel().tolist() for f in fractions]
    steps = [di * nx + dj for di, dj in NEIGHBOURS]
    for k in np.argsort(filled.ravel(), kind="stable")[::-1].tolist():
        a = acc[k]
        if a == 0.0:
            continue
        for d in range(8):
            f = fr[d][k]
            if f > 0.0:
                acc[k + steps[d]] += a * f
    return np.array(acc).reshape(ny, nx)


def edge_mask(shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m


def wetness(z: np.ndarray, cell: float = CELL, floors=(FLOOR,) + FLOORS_REPORTED) -> dict:
    """D6: the filled DEM, the fractions, the area draining through each
    cell, the part of it that comes from the patch's edge, and TWI for each
    floor of tan b (the first is the rule's)."""
    filled = fill(z)
    fractions = fd8_fractions(filled, cell)
    area = accumulate(filled, fractions, np.full(z.shape, cell * cell))
    from_edge = accumulate(filled, fractions, np.where(edge_mask(z.shape), cell * cell, 0.0))
    fit = local_fit(filled, 3, cell)
    tanb = np.hypot(fit["p"], fit["q"])
    twi = {fl: np.log((area / cell) / np.maximum(tanb, fl)) for fl in floors}
    return {"filled": filled, "fractions": fractions, "area": area, "from_edge": from_edge,
            "tanb": tanb, "twi": twi}


# --- D3 and the stage ----------------------------------------------------------------

def at_locations(grid: P.Grid, values: np.ndarray, x, y) -> np.ndarray:
    """Bilinear interpolation between cell centres at the locations."""
    r, c = grid.fractional(x, y)
    return ndimage.map_coordinates(values, [r, c], order=1, mode="constant", cval=np.nan,
                                   prefilter=False)


def covariate_grids(dem: np.ndarray, cell: float = CELL) -> dict[str, np.ndarray]:
    """D4-D6 on a patch's DEM, with the wetness intermediates."""
    out: dict[str, np.ndarray] = {"elevation": dem}
    for n in WINDOWS:
        fit = local_fit(dem, n, cell)
        out[f"slope_{n * 10}"] = slope_degrees(fit)
        out[f"curvature_{n * 10}"] = mean_curvature(fit)
    for r in RADII:
        out[f"tpi_{int(r)}"] = tpi(dem, r, cell)
    w = wetness(dem, cell)
    out["twi"] = w["twi"][FLOOR]
    for fl in FLOORS_REPORTED:
        out[f"twi_floor_{fl:g}"] = w["twi"][fl]
    for k in ("filled", "area", "from_edge", "tanb"):
        out[k] = w[k]
    return out


def field(grid: P.Grid, tiles: list[TileVersion], loc: pd.DataFrame) -> dict:
    """Everything Stage 3 computes for one field."""
    parts = decode_parts(grid, tiles)
    dem, used = compose(parts, grid)
    grids = covariate_grids(dem)
    x, y = loc["x"].to_numpy(), loc["y"].to_numpy()
    values = {k: at_locations(grid, grids[k], x, y) for k in COVARIATES}
    extra = {k: at_locations(grid, grids[k], x, y) for k in grids if k not in COVARIATES}
    return {"grid": grid, "parts": parts, "used": used,
            "grids": grids, "values": values, "extra": extra}


def blocks_of(grid: P.Grid) -> list[str]:
    return P.block_names(P.lonlat_bbox(grid))


def run_stage(loc: pd.DataFrame, tiles: dict[str, TileVersion]) -> tuple[pd.DataFrame, dict]:
    """The covariate table, one row per location, and each field's results."""
    grids = P.patch_grids(loc)
    rows, fields = [], {}
    for site, grid in grids.items():
        missing = [b for b in blocks_of(grid) if b not in tiles]
        if missing:
            raise NotThere(f"{site}: no stored ranges of {', '.join(missing)}")
        g = loc[loc["site"] == site]
        res = field(grid, [tiles[b] for b in blocks_of(grid)], g)
        fields[site] = res
        t = g[["site", "location_id", "x", "y"]].copy()
        for k in COVARIATES:
            t[k] = res["values"][k]
        rows.append(t)
    table = pd.concat(rows, ignore_index=True)
    return table, fields


def write(table: pd.DataFrame, fields: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT, index=False)
    OUT_GRIDS.mkdir(parents=True, exist_ok=True)
    for site, res in fields.items():
        g = res["grid"]
        np.savez_compressed(
            OUT_GRIDS / f"{site}.npz", epsg=g.epsg, x0=g.x0, y1=g.y1, cell=CELL,
            versions=np.array([p.tile.url for p in res["parts"]]), version_used=res["used"],
            **{k: v.astype(np.float32) if k != "filled" else v for k, v in res["grids"].items()})


def main(argv: list[str]) -> int:
    """python -m src.transform.terrain"""
    if argv:
        print(main.__doc__)
        return 2
    try:
        loc = load_locations()
        tiles = stored_tiles()
        table, fields = run_stage(loc, tiles)
    except NotThere as e:
        print(f"NOT THERE: {e}")
        return 2
    write(table, fields)
    print(f"{OUT.relative_to(ROOT)}: {len(table)} locations, {len(COVARIATES)} covariates")
    for site, res in fields.items():
        g = res["grid"]
        print(f"  {site}: patch {g.nx} x {g.ny} cells, tiles "
              + ", ".join(f"{p.tile.block} {p.tile.version}" for p in res["parts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
