"""Review of v1.0, M4: the square grid with the sample mean (RUNBOOK, Stage 5,
Addendum on M4, G1-G4; rules fixed 15.11.2026 before this was written).

For D4's planning grids, the model-expected squared error of the grid's
sample mean as an estimate of the field mean, averaged over the posterior,
beside D4's two cases (simple random sampling with the sample mean, the grid
with block kriging).

    python -m src.transform.grid_mean [--budget SECONDS]
        writes curated/potash_grid_mean.parquet and prints G2. Every step
        is cached, so a run stopped by --budget (exit 4) continues.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np
import pandas as pd

try:                        # run as a module from the repository root
    from src.transform import field_means as FM, kriging as K, reml as M
except ImportError:         # imported with src/transform on sys.path (pytest)
    import field_means as FM
    import kriging as K
    import reml as M

OUT = K.CURATED / "potash_grid_mean.parquet"
ORCHESTRATION = ("main", "run", "summary_table")
_CODE = K.code_hash(__file__, FM.__file__, skip=ORCHESTRATION)


def posterior_points(model: K.Model, res: dict, lp: np.ndarray, mass: float = FM.MASS):
    """The grid points carrying `mass` of the posterior (as D4's heavy_points):
    their (a, nu, tau), renormalised weights and E[sigma^2 | theta]."""
    g = np.array(res["g"])
    w = np.exp(lp[g] - lp[g].max())
    w /= w.sum()
    order = np.argsort(-w, kind="stable")
    k = int(np.searchsorted(np.cumsum(w[order]), mass) + 1)
    chosen = np.sort(order[:k])
    a, t, nu = model.grid().arrays()
    df = float(res["n"] - res["p"])
    e = np.array(res["s2"])[chosen] * df / (df - 2.0)
    thetas = [(float(a[g[i]]), None if np.isnan(nu[g[i]]) else float(nu[g[i]]), float(t[g[i]])) for i in chosen]
    return thetas, w[chosen] / w[chosen].sum(), e


def cross_histogram(d: np.ndarray, width: float = FM.BIN) -> tuple[np.ndarray, np.ndarray]:
    """The distances between the grid's points and the cells, binned as D2
    bins the cells' own pairs: each bin's mean distance and its count."""
    d = d.ravel()
    k = np.floor(d / width).astype(np.int64)
    counts = np.bincount(k)
    sums = np.bincount(k, weights=d)
    used = counts > 0
    return sums[used] / counts[used], counts[used].astype(float)


def sample_mean_mse(model: K.Model, xy, z, cov, pts, pts_cov, cells_xy, cells_cov, drift: tuple,
                    thetas: list, rbb_of) -> list:
    """G1 at each theta, per unit sill: the grid's sample mean's variance term
    (1 - tau)(R_ss - 2 R_sB + R_BB) + tau / n, and its bias (x_s - x_B)' beta(theta),
    beta the GLS coefficients from the field's stocks."""
    X, X0 = K.design(model.mean, xy, cells_xy, cov, cells_cov, drift)
    Xs = K.design(model.mean, xy, pts, cov, pts_cov, drift)[1]
    dx = Xs.mean(axis=0) - X0.mean(axis=0)
    D = K.distances(xy, xy, model.angle, model.ratio)
    Dss = K.distances(pts, pts, model.angle, model.ratio)
    sb = cross_histogram(K.distances(cells_xy, pts, model.angle, model.ratio))
    n = len(pts)
    out, cache = [], {}
    for a, nu, tau in thetas:
        if (a, nu) not in cache:
            R = M.correlation(model.family, D, a, nu)
            lam, U = np.linalg.eigh(R)
            cache = {(a, nu): (lam, U, U.T @ z, U.T @ X,
                               float(M.correlation(model.family, Dss, a, nu).mean()),
                               float(sb[1] @ M.correlation(model.family, sb[0], a, nu) / sb[1].sum()),
                               rbb_of(a, nu))}
        lam, U, Uz, UX, rss, rsb, rbb = cache[(a, nu)]
        dinv = 1.0 / ((1.0 - tau) * lam + tau)
        beta = np.linalg.solve(UX.T @ (dinv[:, None] * UX), UX.T @ (dinv * Uz))
        out.append(((1.0 - tau) * (rss - 2.0 * rsb + rbb) + tau / n, float(dx @ beta)))
    return out


def run(fields: dict, models: dict, selected, cache: K.Cache, log=lambda m: None) -> pd.DataFrame:
    rows = []
    for site in sorted(fields):
        t0 = time.monotonic()
        model = next(m for m in models[site] if m.name == selected[site])
        f, res = FM.field_mean(fields, site, model, cache)
        thetas, w, e = posterior_points(model, res, f["lp"])
        fd = f["fd"]
        starts = [i for i in range(len(thetas)) if i == 0 or thetas[i][:2] != thetas[i - 1][:2]]
        bounds = starts[::FM.PAIRS] + [len(thetas)]
        for n in FM.EFFORT_N:
            pts = FM.planning_grid(f["cells_xy"], fd.xy, n, f["cell"])
            pts_cov = f["cells_cov"][FM.nearest_cells(pts, f["cells_xy"])]
            Xp = K.design(model.mean, pts, pts, pts_cov, pts_cov, f["drift"])[0]
            if len(pts) <= Xp.shape[1] or np.linalg.matrix_rank(Xp) < Xp.shape[1]:
                continue                    # as D4: the grids D4 leaves out
            parts = []
            for lo, hi in zip(bounds[:-1], bounds[1:]):     # steps of at most FM.PAIRS correlation functions
                parts += cache.get(["grid-mean", _CODE, site, model.key(), n, lo, hi] + f["data"],
                                   lambda: sample_mean_mse(model, fd.xy, fd.z, fd.cov, pts, pts_cov, f["cells_xy"],
                                                           f["cells_cov"], f["drift"], thetas[lo:hi],
                                                           lambda a, nu: FM.mean_correlation(
                                                               f["hist"], len(f["cells_xy"]), model.family, a, nu)))
            v = np.array([p[0] for p in parts])
            b = np.array([p[1] for p in parts])
            rows.append({"site": site, "n_target": n, "n": len(pts), "V_mean": float(w @ (e * v + b * b)),
                         "bias_share": float(w @ (b * b)) / float(w @ (e * v + b * b))})
        log(f"{site} ({model.name}): {time.monotonic() - t0:.0f} s")
    return pd.DataFrame(rows)


def summary_table(grid_mean: pd.DataFrame, effort: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    """G2: n for each target and case, as D4's effort_summary computes it."""
    gm = grid_mean.rename(columns={"V_mean": "V"})
    a = FM.effort_summary(gm, means).set_index("site")
    k = FM.effort_summary(effort, means).set_index("site")
    out = pd.DataFrame(index=k.index)
    for name in ("pm5", "mdc10", "mdc5"):
        out[f"{name}_srs"] = k[f"n_{name}_srs"]
        out[f"{name}_grid_mean"] = a[f"n_{name}_grid"]
        out[f"{name}_grid_kriging"] = k[f"n_{name}_grid"]
    return out


def main(argv: list[str]) -> int:
    budget = None
    if argv[:1] == ["--budget"] and len(argv) == 2:
        budget = float(argv[1])
    elif argv:
        print(__doc__)
        return 2
    fields, models, selected = FM.load_inputs()
    cache = K.Cache(FM.CACHE, None if budget is None else time.monotonic() + budget)
    try:
        rows = run(fields, models, selected, cache, log=print)
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget; run again to continue from the cache.")
        return 4
    rows.to_parquet(OUT, index=False)
    means, effort = pd.read_parquet(FM.OUT), pd.read_parquet(FM.OUT_EFFORT)
    pd.set_option("display.width", 220)
    print(summary_table(rows, effort, means).to_string())
    print("\nlargest share of the grid mean's error from its bias, per field:")
    print(rows.groupby("site")["bias_share"].max().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
