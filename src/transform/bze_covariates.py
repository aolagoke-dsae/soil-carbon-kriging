"""Stage 6, D2 and D3: BZE-LW's covariates, read at each published point and
averaged over the disc of 4 km around it (RUNBOOK, Stage 6; rules fixed
15.11.2026 before any covariate was pulled).

    python -m src.transform.bze_covariates
        builds the rasters in curated/bze_rasters/ (each step cached, so a
        run stopped by the time limit continues), then writes
        curated/bze_covariates.parquet

Rasters, each in its source's own grid and CRS:
    temperature, precipitation   DWD 1991-2020, 1 km, EPSG:31467, as stored
    elevation                    BKG DGM200, 200 m, EPSG:25832 (m)
    slope                        degrees, from the elevation's central differences
    tpi1, tpi5                   elevation minus the mean of the cells within 1
                                 and 5 km, the cell itself left out (as Stage 3's D5)
    organic                      1 where a polygon of the map of organic soils
                                 (Wittnebel et al. 2023) covers the cell's centre,
                                 0 elsewhere on the DGM's data, on the DGM's grid

D3 (i): the value of the cell that holds the point. D3 (ii): the mean over the
cells whose centres lie within 4 km of the point, the cells without data
left out and their share recorded.
"""
from __future__ import annotations

import glob
import gzip
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT / "curated"
RASTERS = CURATED / "bze_rasters"
OUT = CURATED / "bze_covariates.parquet"
RADIUS = 4000.0                     # m (D3)
MIN_SHARE = 0.5                     # C2, amended 15.11.2026: below it, or without a point value, a site is left out
TPI_RADII = {"tpi1": 1000.0, "tpi5": 5000.0}
NAMES = ("temperature", "precipitation", "elevation", "slope", "tpi1", "tpi5", "organic")
CHUNK = 50_000                      # polygons rasterised at a time


@dataclass
class Raster:
    """A grid: values (NaN where there are none), its upper-left corner, cell
    size and CRS. Cell (r, c) has its centre at (x0 + (c + 0.5) d, y0 - (r + 0.5) d)."""
    values: np.ndarray
    x0: float
    y0: float
    d: float
    crs: str

    def save(self, name: str) -> None:
        RASTERS.mkdir(parents=True, exist_ok=True)
        np.save(RASTERS / f"{name}.npy", self.values.astype(np.float32))
        (RASTERS / f"{name}.json").write_text(json.dumps({"x0": self.x0, "y0": self.y0, "d": self.d, "crs": self.crs}))

    @classmethod
    def load(cls, name: str) -> "Raster":
        meta = json.loads((RASTERS / f"{name}.json").read_text())
        return cls(np.load(RASTERS / f"{name}.npy").astype(np.float64), meta["x0"], meta["y0"], meta["d"], meta["crs"])


def read_ascii_grid(raw: bytes, crs: str) -> Raster:
    """An ESRI ASCII grid (DWD's), its corner given as XLLCORNER/YLLCORNER."""
    text = gzip.decompress(raw).decode("ascii")
    lines = text.split("\n", 6)
    head = {k.lower(): float(v) for k, v in (l.split() for l in lines[:6])}
    v = np.loadtxt(io.StringIO(lines[6]), dtype=np.float64)
    v[v == head["nodata_value"]] = np.nan
    n = int(head["nrows"])
    if v.shape != (n, int(head["ncols"])):
        raise ValueError("grid shape differs from its header")
    return Raster(v, head["xllcorner"], head["yllcorner"] + n * head["cellsize"], head["cellsize"], crs)


def disc_offsets(radius: float, d: float, exclude_centre: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Row and column offsets of the cells whose centres lie within radius of
    a cell centre."""
    k = int(math.floor(radius / d))
    r, c = np.mgrid[-k:k + 1, -k:k + 1]
    inside = (r * d) ** 2 + (c * d) ** 2 <= radius ** 2 + 1e-9
    if exclude_centre:
        inside &= ~((r == 0) & (c == 0))
    return r[inside], c[inside]


def neighbourhood_mean(values: np.ndarray, radius: float, d: float) -> np.ndarray:
    """Each cell's mean over the cells within radius, itself left out and the
    cells without data skipped (by FFT convolution of the values and of the
    data mask)."""
    from scipy.signal import fftconvolve
    k = int(math.floor(radius / d))
    r, c = np.mgrid[-k:k + 1, -k:k + 1]
    kern = ((r * d) ** 2 + (c * d) ** 2 <= radius ** 2 + 1e-9).astype(np.float64)
    kern[k, k] = 0.0
    ok = np.isfinite(values)
    s = fftconvolve(np.where(ok, values, 0.0), kern, mode="same")
    n = fftconvolve(ok.astype(np.float64), kern, mode="same")
    out = np.full(values.shape, np.nan)
    good = ok & (n > 0.5)
    out[good] = s[good] / np.round(n[good])
    return out


def slope_degrees(z: np.ndarray, d: float) -> np.ndarray:
    gy, gx = np.gradient(z, d)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def _newest(pattern: str) -> Path:
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return Path(files[-1])


def build(step: str) -> None:
    """One raster, from raw/ or from the rasters already built."""
    if step in ("temperature", "precipitation"):
        var = {"temperature": "air_temperature_mean", "precipitation": "precipitation"}[step]
        read_ascii_grid(_newest(f"raw/DWD/{var}/*_17.asc/*.gz").read_bytes(), "EPSG:31467").save(step)
    elif step == "elevation":
        import rasterio
        z = _newest("raw/BKG/dgm200/dgm200.utm32s.tiff/*.zip")
        with rasterio.open(f"/vsizip/{z.as_posix()}/dgm200/dgm200_utm32s.tif") as src:
            v = src.read(1).astype(np.float64)
            if src.nodata is not None:
                v[v == src.nodata] = np.nan
            t = src.transform
            if abs(t.a + t.e) > 1e-9 or t.b or t.d:
                raise ValueError("DGM200 is not on a square, north-up grid")
            Raster(v, t.c, t.f, t.a, str(src.crs)).save("elevation")
    elif step == "slope":
        e = Raster.load("elevation")
        Raster(slope_degrees(e.values, e.d), e.x0, e.y0, e.d, e.crs).save("slope")
    elif step in TPI_RADII:
        e = Raster.load("elevation")
        Raster(e.values - neighbourhood_mean(e.values, TPI_RADII[step], e.d), e.x0, e.y0, e.d, e.crs).save(step)
    elif step == "organic":
        import pyogrio
        from rasterio import features
        from rasterio.transform import from_origin
        e = Raster.load("elevation")
        z = _newest("raw/THUENEN/organic/Thuenen_Kulisse_OrgBoeden_v1/*.zip")
        path = f"/vsizip/{z.as_posix()}/Thuenen_Kulisse_OrgBoeden_v1/thuenen_kulisse_orgboeden_v1.shp"
        info = pyogrio.read_info(path)
        if str(info["crs"]) != e.crs:
            raise ValueError(f"the map is in {info['crs']}, the DGM in {e.crs}")
        mask = np.zeros(e.values.shape, dtype=np.uint8)
        transform = from_origin(e.x0, e.y0, e.d, e.d)
        for start in range(0, int(info["features"]), CHUNK):
            g = pyogrio.read_dataframe(path, skip_features=start, max_features=CHUNK, columns=[])
            part = features.rasterize(((geom, 1) for geom in g.geometry if geom is not None), out_shape=mask.shape,
                                      transform=transform, fill=0, all_touched=False, dtype="uint8")
            mask |= part
        v = mask.astype(np.float64)
        v[~np.isfinite(e.values)] = np.nan
        Raster(v, e.x0, e.y0, e.d, e.crs).save("organic")
    else:
        raise ValueError(step)


def read(r: Raster, x: np.ndarray, y: np.ndarray, radius: float = RADIUS) -> pd.DataFrame:
    """D3 at points given in the raster's CRS: the value at the point, the disc
    mean and the share of the disc's cells that have data."""
    rows, cols = r.values.shape
    ci = np.floor((x - r.x0) / r.d).astype(int)
    ri = np.floor((r.y0 - y) / r.d).astype(int)
    point = np.full(len(x), np.nan)
    inside = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
    point[inside] = r.values[ri[inside], ci[inside]]
    k = int(math.ceil(radius / r.d)) + 1
    dr, dc = np.mgrid[-k:k + 1, -k:k + 1]
    mean, share = np.full(len(x), np.nan), np.zeros(len(x))
    for i in range(len(x)):
        rr, cc = ri[i] + dr, ci[i] + dc
        cx = r.x0 + (cc + 0.5) * r.d
        cy = r.y0 - (rr + 0.5) * r.d
        disc = (cx - x[i]) ** 2 + (cy - y[i]) ** 2 <= radius ** 2
        rr, cc = rr[disc], cc[disc]
        ok = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        vals = np.full(len(rr), np.nan)
        vals[ok] = r.values[rr[ok], cc[ok]]
        good = np.isfinite(vals)
        share[i] = good.mean() if len(vals) else 0.0
        if good.any():
            mean[i] = vals[good].mean()
    return pd.DataFrame({"point": point, "disc": mean, "share": share})


def sites() -> pd.DataFrame:
    try:
        from src.checks import checkpoint1 as C1
    except ImportError:
        import checkpoint1 as C1
    t = C1.bze_table(C1.bze_pages())
    return t[["pointid", "xcoord", "ycoord"]].astype({"xcoord": float, "ycoord": float})


def covariates(s: pd.DataFrame) -> pd.DataFrame:
    from pyproj import Transformer
    out = s[["pointid"]].copy()
    for name in NAMES:
        r = Raster.load(name)
        x, y = s["xcoord"].to_numpy(), s["ycoord"].to_numpy()
        if r.crs != "EPSG:25832":
            x, y = Transformer.from_crs("EPSG:25832", r.crs, always_xy=True).transform(x, y)
        got = read(r, np.asarray(x, float), np.asarray(y, float))
        out[f"{name}_point"], out[f"{name}_disc"], out[f"{name}_share"] = got["point"], got["disc"], got["share"]
    return out


def kept(cov: pd.DataFrame) -> pd.Series:
    """C2 as amended: a site is kept when every covariate has a value under
    both readings and every disc has at least half its cells inside the data."""
    values = [c for c in cov if c.endswith(("_point", "_disc"))]
    shares = [c for c in cov if c.endswith("_share")]
    return cov[values].notna().all(axis=1) & (cov[shares].min(axis=1) >= MIN_SHARE)


def main(argv: list[str]) -> int:
    if argv:
        print(__doc__)
        return 2
    t0 = time.monotonic()
    for step in ("temperature", "precipitation", "elevation", "slope", "tpi1", "tpi5", "organic"):
        if not (RASTERS / f"{step}.npy").exists():
            if time.monotonic() - t0 > 60:
                print("Stopped before the next step; run again to continue.")
                return 4
            build(step)
            print(f"{step}: built, {time.monotonic() - t0:.0f} s")
    cov = covariates(sites())
    cov["kept"] = kept(cov)
    cov.to_parquet(OUT, index=False)
    print(f"{len(cov)} sites -> {OUT.relative_to(ROOT).as_posix()}  ({time.monotonic() - t0:.0f} s)")
    print(cov.drop(columns="pointid").describe().T[["min", "50%", "max"]].round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
