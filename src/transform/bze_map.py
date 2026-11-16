"""Stage 6, D8: the maps of mineral soils' stock in 0-30 cm, at the centre of
every 1 km cell of Germany, if the cell were cropland and if it were
grassland (RUNBOOK, Stage 6).

The model is the one D6 scored best under reading (ii): QRF on the disc
readings (CRPS 11.99 against RF-RK's 12.62), refitted on all 2,937 mineral
sites C2 keeps. Each cell's covariates are D3's disc means around its centre.
The 1 km centres coincide with DGM200 cell centres, so the 200 m rasters'
disc means are taken from one convolution with D3's disc (C3's definition);
DWD's grids are read at each centre by D3's reader.

    python -m src.transform.bze_map [--budget SECONDS]
        writes curated/bze_map.parquet; every step is cached in
        curated/bze_cache/, so a run stopped by --budget (exit 4) continues
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
import time

import numpy as np
import pandas as pd

try:                        # run as a module from the repository root
    from src.transform import bze_covariates as BC, bze_models as BM, kriging as K
except ImportError:         # imported with src/transform on sys.path (pytest)
    import bze_covariates as BC
    import bze_models as BM
    import kriging as K

OUT = BC.CURATED / "bze_map.parquet"
STEP = 1000.0                               # m
CHUNK = 4000                                # cells per cached step
SCENARIOS = ("A", "G")                      # cropland, grassland
LEVELS = (0.05, 0.5, 0.95)
SEED = BM.SEED + 9000
ORCHESTRATION = ("main", "run")
_CODE = K.code_hash(__file__, BM.__file__, BC.__file__, skip=ORCHESTRATION)


def centres(e: BC.Raster) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The 1 km cell centres over the DGM's extent, (x, y) in EPSG:25832, and
    the row and column of the DGM cell each sits at the centre of."""
    k = int(round(STEP / e.d))
    rows, cols = e.values.shape
    r = np.arange(k // 2, rows, k)
    c = np.arange(k // 2, cols, k)
    R, C = np.meshgrid(r, c, indexing="ij")
    R, C = R.ravel(), C.ravel()
    return e.x0 + (C + 0.5) * e.d, e.y0 - (R + 0.5) * e.d, R, C


def disc_mean_raster(v: np.ndarray, d: float, radius: float = BC.RADIUS) -> tuple[np.ndarray, np.ndarray]:
    """Every cell's mean over the cells whose centres lie within radius of its
    centre, itself included and the cells without data skipped, and the share
    of the disc's cells that have data."""
    from scipy.signal import fftconvolve
    k = int(math.floor(radius / d))
    r, c = np.mgrid[-k:k + 1, -k:k + 1]
    kern = ((r * d) ** 2 + (c * d) ** 2 <= radius ** 2 + 1e-9).astype(np.float64)
    ok = np.isfinite(v)
    s = fftconvolve(np.where(ok, v, 0.0), kern, mode="same")
    n = np.round(fftconvolve(ok.astype(np.float64), kern, mode="same"))
    # the share counts the disc's cells outside the raster as without data, as D3's reader does
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan), n / kern.sum()


def cell_covariates(cache: K.Cache) -> pd.DataFrame:
    e = BC.Raster.load("elevation")
    x, y, R, C = centres(e)
    out = pd.DataFrame({"x": x, "y": y})
    for name in BC.NAMES:
        r = BC.Raster.load(name)
        if r.crs == e.crs and r.d == e.d and (r.x0, r.y0) == (e.x0, e.y0):
            got = cache.get(["bze-map-raster", _CODE, name], lambda: _disc_at(r, R, C))
            out[f"{name}_disc"], out[f"{name}_share"] = got["disc"], got["share"]
        else:
            from pyproj import Transformer
            xx, yy = Transformer.from_crs(e.crs, r.crs, always_xy=True).transform(x, y)
            got = cache.get(["bze-map-read", _CODE, name], lambda: BC.read(r, np.asarray(xx), np.asarray(yy)).to_dict(orient="list"))
            out[f"{name}_disc"], out[f"{name}_share"], out[f"{name}_point"] = got["disc"], got["share"], got["point"]
    return out


def _disc_at(r: BC.Raster, R: np.ndarray, C: np.ndarray) -> dict:
    m, share = disc_mean_raster(r.values, r.d)
    return {"disc": [None if not np.isfinite(v) else float(v) for v in m[R, C]], "share": share[R, C].tolist()}


def mapped(cells: pd.DataFrame) -> pd.Series:
    """The cells mapped: on DWD's land grid at their centre, and every disc
    with at least half its cells inside the data (C2, as amended)."""
    discs = [f"{n}_disc" for n in BC.NAMES]
    shares = [f"{n}_share" for n in BC.NAMES]
    return (cells["temperature_point"].notna() & cells[discs].notna().all(axis=1)
            & (cells[shares].min(axis=1) >= BC.MIN_SHARE))


def weights_dense(rf, X_train: np.ndarray, X_cells: np.ndarray) -> np.ndarray:
    """Meinshausen's weights (forests.qrf_weights), as one sparse product over
    all the trees: cells x (tree, leaf), times (tree, leaf) x training sites
    with 1 / the leaf's size."""
    from scipy import sparse
    lt, lc = rf.apply(X_train), rf.apply(X_cells)
    n, m, T = len(X_train), len(X_cells), lt.shape[1]
    rows_a, cols_a, vals_a, cols_b = [], [], [], []
    offset = 0
    for t in range(T):
        leaves, inv, counts = np.unique(lt[:, t], return_inverse=True, return_counts=True)
        rows_a.append(offset + inv)
        cols_a.append(np.arange(n))
        vals_a.append(1.0 / counts[inv])
        cols_b.append(offset + np.searchsorted(leaves, lc[:, t]))
        offset += len(leaves)
    A = sparse.csr_matrix((np.concatenate(vals_a), (np.concatenate(rows_a), np.concatenate(cols_a))), shape=(offset, n))
    B = sparse.csr_matrix((np.ones(m * T), (np.repeat(np.arange(m), T), np.stack(cols_b, axis=1).ravel())),
                          shape=(m, offset))
    return (B @ A).toarray() / T


def summarise(W: np.ndarray, y: np.ndarray) -> dict:
    """Each cell's predictive mean and quantiles at LEVELS (Meinshausen 2006, eq. 1)."""
    order = np.argsort(y, kind="stable")
    ys, cw = y[order], np.cumsum(W[:, order], axis=1)
    out = {"mean": (W @ y).tolist()}
    for lev in LEVELS:
        i = (cw < lev - 1e-12).sum(axis=1)
        out[f"q{lev:.2f}"] = ys[np.minimum(i, len(ys) - 1)].tolist()
    return out


def run(cache: K.Cache, log=lambda m: None) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor
    t = BM.load()
    p = t[t["bze_moor"].astype(int) == 0].reset_index(drop=True)
    X, cols = BM.features(p, "disc")
    y = p["kv_0_30"].to_numpy(float)
    rf = RandomForestRegressor(n_estimators=BM.F.TREES, max_features=BM.max_features(X.shape[1]),
                               min_samples_leaf=BM.F.MIN_LEAF, bootstrap=True, random_state=SEED, n_jobs=-1).fit(X, y)
    cells = cell_covariates(cache)
    keep = mapped(cells)
    c = cells[keep].reset_index(drop=True)
    log(f"{int(keep.sum())} of {len(cells)} cells mapped")
    parts = []
    for lu in SCENARIOS:
        q = c.rename(columns={"x": "xcoord", "y": "ycoord"}).assign(landnutzung=lu)
        Xc, cc = BM.features(q, "disc")
        assert cc == cols
        for start in range(0, len(q), CHUNK):
            t0 = time.monotonic()
            sl = slice(start, start + CHUNK)
            res = cache.get(["bze-map", _CODE, lu, start, K.data_key(Xc[sl], X, y)],
                            lambda: summarise(weights_dense(rf, X, Xc[sl]), y))
            parts.append(pd.DataFrame({"x": c["x"].iloc[sl].to_numpy(), "y": c["y"].iloc[sl].to_numpy(),
                                       "land_use": lu, **res}))
            log(f"{lu} {start}: {time.monotonic() - t0:.0f} s")
    return pd.concat(parts, ignore_index=True)


def main(argv: list[str]) -> int:
    budget = None
    if argv[:1] == ["--budget"] and len(argv) == 2:
        budget = float(argv[1])
    elif argv:
        print(__doc__)
        return 2
    cache = K.Cache(BM.CACHE, None if budget is None else time.monotonic() + budget)
    try:
        out = run(cache, log=print)
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget; run again to continue from the cache.")
        return 4
    out.to_parquet(OUT, index=False)
    print(f"{len(out)} rows -> {OUT.relative_to(BC.ROOT).as_posix()}")
    print(out.groupby("land_use")[["mean", "q0.05", "q0.50", "q0.95"]].median().round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
