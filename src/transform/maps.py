"""Stage 4, D10 as amended on 10.11.2026 (RUNBOOK, Stage 4, Addendum): the maps.

At each of Stage 3's 10 m cells whose centre lies inside a field's location
hull and whose covariates are all defined, two predictive distributions of
the 0-30 cm stock of a new observation there, nugget included, as in the
cross-validation:

    kriging   the selected mean model's Bayesian predictive (D3, D5) from all
              the field's stocks. Every grid point's correlation matrix is
              diagonalised once, R = U diag(l) U', so that C = (1 - tau) R +
              tau I has the eigenvalues (1 - tau) l + tau for every
              nugget-to-sill ratio at the cost of one: the algebra of
              kriging.evaluate, organised for thousands of targets. Grid
              points whose posterior weight is below 1e-12 of the largest are
              left out. Under (c), the drift the AIC chooses from all the
              stocks (D4); under (b) at the small fields, the shared posterior
              from all eight fields' stocks
    QRF       D7's forest trained on all 1,163 stocks, its weights at the cell

For each, the mean, the standard deviation and the 5 and 95 % quantiles.
Notebook 04 computed the same on 08.11.2026, before this module existed;
rule K7 checks the kriging against kriging.evaluate and kriging.predictive.

    python -m src.transform.maps [--budget SECONDS] [--sites A,B]
        writes curated/potash_maps.parquet. Every step (a field's posterior,
        its kriging in blocks of 1,200 cells, its QRF) is cached in
        curated/maps_cache/, keyed by the code and the data, so a run stopped
        by --budget (exit 4) continues where it stopped; --sites splits the
        work between runs, and a run without it writes the table.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

try:                        # run as a module from the repository root
    from src.transform import forests as F, kriging as K, patches as P, reml as M, scores as SC
except ImportError:         # imported with src/transform on sys.path (pytest)
    import forests as F
    import kriging as K
    import patches as P
    import reml as M
    import scores as SC

OUT = K.CURATED / "potash_maps.parquet"
CACHE = K.CURATED / "maps_cache"
GRIDS = K.CURATED / "terrain"
KEEP = 1e-12                    # grid points lighter than this, relative to the heaviest, are left out
LEVELS = (0.05, 0.95)
NEWTON_STEPS = 60
NEWTON_TOL = 1e-12
CHUNK = 400                     # cells per block in the mixtures' quantiles
BLOCK = 3 * CHUNK               # cells per cached step of the kriging
SUMMARY = ("mean", "sd", "q05", "q95")
COLUMNS = (["site", "row", "col", "x", "y", "model", "drift"] + [f"kriging_{s}" for s in SUMMARY]
           + [f"qrf_{s}" for s in SUMMARY])
ORCHESTRATION = ("main", "run_stage", "load_inputs")
_CODE = K.code_hash(__file__, F.__file__, P.__file__, skip=ORCHESTRATION)


def load_grid(site: str) -> dict:
    """Stage 3's terrain grid of a field: elevation, every covariate, the cell
    size, the upper-left corner (x0, y1) and the EPSG code."""
    with np.load(GRIDS / f"{site}.npz") as z:
        return {k: z[k] for k in z.files}


def hull_cells(xy: np.ndarray, g: dict) -> tuple:
    """The 10 m cells whose centres lie inside the hull of the locations xy,
    with every covariate defined: rows, columns, coordinates and covariates."""
    ny, nx = g["elevation"].shape
    cell = float(g["cell"])
    xs = float(g["x0"]) + (np.arange(nx) + 0.5) * cell
    ys = float(g["y1"]) - (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    inside = Delaunay(xy).find_simplex(np.c_[X.ravel(), Y.ravel()]) >= 0
    rows, cols = np.divmod(np.flatnonzero(inside), nx)
    cov = np.column_stack([g[c][rows, cols].astype(float) for c in K.COVARIATES])
    ok = np.all(np.isfinite(cov), axis=1)
    return rows[ok], cols[ok], np.c_[X[rows[ok], cols[ok]], Y[rows[ok], cols[ok]]], cov[ok]


def all_stocks_drift(fd: K.FieldData, model: K.Model) -> tuple:
    """(c)'s drift as the AIC chooses it from all the field's stocks (D4)."""
    return K.drift_candidates()[K.choose(K.ml_table(model.grid(), fd.xy, fd.z, fd.cov))]


def shared_others_all(fields: dict, site: str, model: K.Model) -> np.ndarray:
    """Under (b) at a small field: the other seven fields' integrated
    likelihoods on the shared grid, from all their stocks."""
    total = 0.0
    for other in sorted(fields):
        if other != site:
            o = fields[other]
            X = M.design(o.xy, "quadratic", o.xy.mean(axis=0))
            total = total + K.evaluate(model.grid(), o.xy, o.z, X).logpost
    return total


def mixture_summary(w: np.ndarray, locs: np.ndarray, scales: np.ndarray, df: float,
                    chunk: int = CHUNK) -> np.ndarray:
    """Each cell's mixture of t (cells x components): its mean, standard
    deviation and 5 and 95 % quantiles, the quantiles by Newton's method from
    the normal with the mixture's mean and variance."""
    table, out = SC.t_table(df), []
    for s in range(0, len(locs), chunk):
        L, S = locs[s:s + chunk], scales[s:s + chunk]
        mean = L @ w
        start = np.sqrt((S ** 2 * df / (df - 2.0) + L ** 2) @ w - mean ** 2)     # Newton's start only
        qs = []
        for level in LEVELS:
            x = mean + start * SC.t_ppf(level, df)
            for _ in range(NEWTON_STEPS):
                z = (x[:, None] - L) / S
                err = table.cdf(z) @ w - level
                if np.max(np.abs(err)) < NEWTON_TOL:
                    break
                x = x - err / ((SC.t_pdf(z, df) / S) @ w)
            qs.append(x)
        sd = np.sqrt((S ** 2 * df / (df - 2.0) + (L - mean[:, None]) ** 2) @ w)
        out.append(np.c_[mean, sd, qs[0], qs[1]])
    return np.vstack(out)


def log_posterior(fd: K.FieldData, model: K.Model, drift: tuple = (),
                  others: Optional[np.ndarray] = None) -> np.ndarray:
    """The model's log posterior on its grid from all the field's stocks;
    others, under (b) at a small field, the other fields' integrated
    likelihoods."""
    X = K.design(model.mean, fd.xy, fd.xy[:1], fd.cov, fd.cov[:1], drift)[0]
    lp = K.evaluate(model.grid(), fd.xy, fd.z, X, angle=model.angle, ratio=model.ratio).logpost
    return lp if others is None else lp + others


def kriging_cells(fd: K.FieldData, model: K.Model, xy0: np.ndarray, cov0: np.ndarray, drift: tuple = (),
                  lp: Optional[np.ndarray] = None) -> np.ndarray:
    """At each cell: the mean, standard deviation and 5 and 95 % quantiles of
    the model's Bayesian predictive from all the field's stocks, given its log
    posterior lp (log_posterior's, by default)."""
    X, X0 = K.design(model.mean, fd.xy, xy0, fd.cov, cov0, drift)
    n, p = X.shape
    D, D0 = K.distances(fd.xy, fd.xy, model.angle, model.ratio), K.distances(xy0, fd.xy, model.angle, model.ratio)
    grid = model.grid()
    lp = log_posterior(fd, model, drift) if lp is None else lp
    keep = lp >= lp.max() + np.log(KEEP)
    w = np.exp(lp[keep] - lp.max())
    w /= w.sum()
    locs, scales, g = [], [], 0
    for a, nu, taus in grid.points():
        if not keep[g:g + len(taus)].any():
            g += len(taus)
            continue
        lam, U = np.linalg.eigh(M.correlation(model.family, D, a, nu))
        Uz, UX, P0 = U.T @ fd.z, U.T @ X, U.T @ M.correlation(model.family, D0, a, nu).T
        for tau in taus:
            if keep[g]:
                dinv = 1.0 / ((1.0 - tau) * lam + tau)
                XCX = UX.T @ (dinv[:, None] * UX)
                beta = np.linalg.solve(XCX, UX.T @ (dinv * Uz))
                r = Uz - UX @ beta
                s2 = float(r @ (dinv * r)) / (n - p)
                Kw = (1.0 - tau) * P0
                pred = X0 @ beta + Kw.T @ (dinv * r)
                G = X0.T - UX.T @ (dinv[:, None] * Kw)
                v1 = 1.0 - ((Kw * Kw) * dinv[:, None]).sum(axis=0) + (G * np.linalg.solve(XCX, G)).sum(axis=0)
                locs.append(pred)
                scales.append(np.sqrt(s2 * np.maximum(v1, 1e-12)))
            g += 1
    locs, scales = np.array(locs).T, np.array(scales).T          # cells x components
    return mixture_summary(w, locs, scales, float(n - p))


def fit_forest(stacked: F.Stacked):
    """D7's forest on all 1,163 stocks."""
    rf = F.forest(F.SEED)
    rf.set_params(oob_score=False, n_jobs=-1)
    return rf.fit(stacked.X, stacked.y)


def qrf_cells(rf, stacked: F.Stacked, xy0: np.ndarray, cov0: np.ndarray, epsg: int) -> np.ndarray:
    """At each cell: QRF's mean, standard deviation and 5 and 95 % quantiles."""
    lon, lat = P.transformer(int(epsg), "EPSG:4326").transform(xy0[:, 0], xy0[:, 1])
    W = F.qrf_weights(rf, stacked.X, np.column_stack([lon, lat, cov0]))
    out = []
    for w in W:
        d = SC.Weighted(stacked.y, w)
        q = d.quantiles(list(LEVELS))
        out.append([d.mean(), math.sqrt(F.weighted_var(d)), q[0], q[1]])
    return np.array(out)


def field_map(fields: dict, site: str, model: K.Model, forest, stacked: F.Stacked, cache: K.Cache) -> pd.DataFrame:
    """One field's map, in cached steps: the posterior (and (c)'s drift), the
    kriging in blocks of cells, QRF."""
    fd = fields[site]
    g = load_grid(site)
    rows, cols, xy0, cov0 = hull_cells(fd.xy, g)
    data = [K.data_key(fd.xy, fd.z, fd.cov)]
    if model.shared:
        data += [K.data_key(fields[o].xy, fields[o].z) for o in sorted(fields) if o != site]

    def posterior():
        drift = all_stocks_drift(fd, model) if model.mean == "drift" else ()
        others = shared_others_all(fields, site, model) if model.shared else None
        return {"drift": list(drift), "logpost": log_posterior(fd, model, drift, others).tolist()}

    post = cache.get(["maps-posterior", _CODE, site, model.key()] + data, posterior)
    drift, lp = tuple(post["drift"]), np.array(post["logpost"], dtype=float)
    k = []
    for b in range(0, len(rows), BLOCK):
        cells = K.data_key(xy0[b:b + BLOCK], cov0[b:b + BLOCK])
        k += cache.get(["maps-kriging", _CODE, site, model.key(), cells] + data,
                       lambda: kriging_cells(fd, model, xy0[b:b + BLOCK], cov0[b:b + BLOCK], drift, lp).tolist())
    q = cache.get(["maps-qrf", _CODE, site, K.data_key(xy0, cov0), K.data_key(stacked.X, stacked.y)],
                  lambda: qrf_cells(forest(), stacked, xy0, cov0, g["epsg"]).tolist())
    k, q = np.array(k), np.array(q)
    out = pd.DataFrame({"site": site, "row": rows, "col": cols, "x": xy0[:, 0], "y": xy0[:, 1],
                        "model": model.name, "drift": K.drift_name(drift) if model.mean == "drift" else ""})
    for i, s in enumerate(SUMMARY):
        out[f"kriging_{s}"] = k[:, i]
        out[f"qrf_{s}"] = q[:, i]
    return out[COLUMNS]


def run_stage(fields: dict, models: dict, selected, stacked: F.Stacked, cache: K.Cache, sites=None,
              log=lambda message: None) -> pd.DataFrame:
    """Every field's map (or those in sites), from the cache where it is there."""
    fitted = []

    def forest():
        if not fitted:
            fitted.append(fit_forest(stacked))
        return fitted[0]

    parts = []
    for site in (sites or sorted(fields)):
        model = next(m for m in models[site] if m.name == selected[site])
        t0 = time.monotonic()
        parts.append(field_map(fields, site, model, forest, stacked, cache))
        log(f"{site} ({model.name}): {len(parts[-1]):,} cells, {time.monotonic() - t0:.0f} s")
    return pd.concat(parts, ignore_index=True)


def load_inputs() -> tuple:
    fields, record, aniso, boxcox = K.load_inputs()
    models = {s: K.models_for(s, record, aniso, boxcox) for s in sorted(fields)}
    selected = pd.read_parquet(K.OUT_SELECTION).set_index("site")["selected"]
    stacked = F.stack(pd.read_parquet(K.STOCKS), pd.read_parquet(K.TERRAIN), pd.read_parquet(F.FO.OUT))
    return fields, models, selected, stacked


def main(argv: list[str]) -> int:
    """python -m src.transform.maps [--budget SECONDS] [--sites A,B]"""
    budget, sites = None, None
    args = list(argv)
    while args:
        if len(args) >= 2 and args[0] == "--budget":
            budget = float(args[1])
        elif len(args) >= 2 and args[0] == "--sites":
            sites = args[1].split(",")
        else:
            print(main.__doc__)
            return 2
        args = args[2:]
    fields, models, selected, stacked = load_inputs()
    if sites and not set(sites) <= set(fields):
        print(f"unknown field in {sites}")
        return 2
    cache = K.Cache(CACHE, None if budget is None else time.monotonic() + budget)
    try:
        table = run_stage(fields, models, selected, stacked, cache, sites, log=print)
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget; run again to continue from the cache.")
        return 4
    if sites:
        print(f"{', '.join(sites)}: in the cache")
        return 0
    table.to_parquet(OUT, index=False)
    print(f"{len(table):,} cells in {table['site'].nunique()} fields -> {OUT.relative_to(K.ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
