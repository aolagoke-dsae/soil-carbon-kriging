"""Stage 5, D1 to D6: each field's mean stock, and what it means for monitoring.

A field is the set of its 10 m cells in Stage 4's maps (D1). Its mean stock
is predicted by Bayesian block kriging (D2): under the selected mean model
and its posterior from all the field's stocks, at every grid point the
universal block-kriging predictor of the cells' mean,

    m_B = x_B' b + k_B' C^-1 (z - X b),    k_B = (1 - tau) mean over cells of R(x_i, cell),
    v_B = (1 - tau) R_BB - k_B' C^-1 k_B + g' (X'C^-1 X)^-1 g,    g = x_B - X' C^-1 k_B,

per unit sill, R_BB the correlation averaged over all pairs of cells (the
nugget averages away over a field), by a histogram of the pairs' distances
in bins of 5 cm. The predictive of the field mean is the posterior mixture
of t with n - p degrees of freedom, as in Stage 4. Beside it: the sample
mean with its simple-random-sampling standard error and the model-based
error of that sample mean (D3); and, for square grids of n samples, the
variance of the block-kriged field mean averaged over the posterior, with
simple random sampling's D^2 / n beside it (D4), and the minimum detectable
change of two campaigns (D5).

    python -m src.transform.field_means [--budget SECONDS] [--sites A,B]
        writes curated/potash_field_means.parquet and
        curated/potash_effort.parquet. Every step is cached in
        curated/field_means_cache/, so a run stopped by --budget (exit 4)
        continues where it stopped; a run without --sites writes the tables.
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
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

try:                        # run as a module from the repository root
    from src.transform import kriging as K, maps as MP, reml as M, scores as SC
except ImportError:         # imported with src/transform on sys.path (pytest)
    import kriging as K
    import maps as MP
    import reml as M
    import scores as SC

OUT = K.CURATED / "potash_field_means.parquet"
OUT_EFFORT = K.CURATED / "potash_effort.parquet"
CACHE = K.CURATED / "field_means_cache"
BIN = 0.05                      # m: the pairs' distances are binned this finely (D2)
KEEP = MP.KEEP                  # grid points lighter than this, relative to the heaviest, are left out
MASS = 0.999                    # the effort curves use the grid points that carry this much weight (D4)
LEVELS = (0.025, 0.5, 0.975)
EFFORT_N = (4, 6, 9, 12, 16, 25, 36, 50, 70, 100, 140, 200, 280, 400)
Z_MDC = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)       # 1.960 + 0.842 (D5)
RUNNER_UP = {"IL-RT": "b", "NE": "a"}                       # within 1 % of CRPS of the selected (D6)
PAIRS = 40                      # correlation functions (range and nu) per cached step
ORCHESTRATION = ("main", "run_stage", "load_inputs", "effort_summary")
_CODE = K.code_hash(__file__, MP.__file__, skip=ORCHESTRATION)


# --- the block's own correlation ----------------------------------------------------------------

def transformed(xy: np.ndarray, angle: Optional[float] = None, ratio: Optional[float] = None) -> np.ndarray:
    """Coordinates in which Euclidean distance is Stage 2's anisotropic distance."""
    if angle is None:
        return xy
    t = math.radians(angle)
    return np.c_[xy[:, 0] * math.sin(t) + xy[:, 1] * math.cos(t),
                 (xy[:, 0] * math.cos(t) - xy[:, 1] * math.sin(t)) / ratio]


def pair_histogram(xy: np.ndarray, angle=None, ratio=None, width: float = BIN) -> tuple[np.ndarray, np.ndarray]:
    """The distances between all pairs of points, binned: each bin's mean
    distance and its number of (unordered) pairs."""
    d = pdist(transformed(xy - xy.mean(axis=0), angle, ratio))
    k = np.floor(d / width).astype(np.int64)
    counts = np.bincount(k)
    sums = np.bincount(k, weights=d)
    used = counts > 0
    return sums[used] / counts[used], counts[used]


def mean_correlation(hist: tuple, n_points: int, family: str, a: float, nu) -> float:
    """R averaged over all ordered pairs of the points, each with itself included."""
    dist, counts = hist
    return float((n_points + 2.0 * counts @ M.correlation(family, dist, a, nu)) / n_points ** 2)


def exact_mean_correlation(xy: np.ndarray, family: str, a: float, nu, angle=None, ratio=None) -> float:
    """The same without binning (B1, E1)."""
    d = pdist(transformed(xy - xy.mean(axis=0), angle, ratio))
    return float((len(xy) + 2.0 * M.correlation(family, d, a, nu).sum()) / len(xy) ** 2)


# --- block kriging on the grid -----------------------------------------------------------------

def block_grid(fd: K.FieldData, model: K.Model, cells_xy: np.ndarray, cells_cov: np.ndarray, drift: tuple,
               lp: np.ndarray, hist: tuple, keep: Optional[np.ndarray] = None) -> dict:
    """At every kept grid point: the block predictor and variance per unit
    sill, S^2/(n - p), the GLS coefficients, and the sample mean's model-based
    squared error per unit sill (D3)."""
    X, X0 = K.design(model.mean, fd.xy, cells_xy, fd.cov, cells_cov, drift)
    n, p = X.shape
    xB = X0.mean(axis=0)
    xs = X.mean(axis=0)
    D, D0 = K.distances(fd.xy, fd.xy, model.angle, model.ratio), K.distances(cells_xy, fd.xy, model.angle, model.ratio)
    grid = model.grid()
    keep = (lp >= lp.max() + np.log(KEEP)) if keep is None else keep
    out = {k: [] for k in ("g", "pred", "v", "s2", "sample_sq", "trend_var", "rbb")}
    g = 0
    for a, nu, taus in grid.points():
        if not keep[g:g + len(taus)].any():
            g += len(taus)
            continue
        R = M.correlation(model.family, D, a, nu)
        lam, U = np.linalg.eigh(R)
        rb = M.correlation(model.family, D0, a, nu).mean(axis=0)        # (n,): mean over the cells
        rbb = mean_correlation(hist, len(cells_xy), model.family, a, nu)
        rss = float(R.mean())
        rsb = float(rb.mean())
        Uz, UX, Ub = U.T @ fd.z, U.T @ X, U.T @ rb
        for tau in taus:
            if keep[g]:
                dinv = 1.0 / ((1.0 - tau) * lam + tau)
                XCX = UX.T @ (dinv[:, None] * UX)
                beta = np.linalg.solve(XCX, UX.T @ (dinv * Uz))
                r = Uz - UX @ beta
                s2 = float(r @ (dinv * r)) / (n - p)
                kw = (1.0 - tau) * Ub
                gv = xB - UX.T @ (dinv * kw)
                v = (1.0 - tau) * rbb - float(kw @ (dinv * kw)) + float(gv @ np.linalg.solve(XCX, gv))
                trend = X0 @ beta
                bias = float((xs - xB) @ beta)
                out["g"].append(g)
                out["pred"].append(float(xB @ beta + kw @ (dinv * r)))
                out["v"].append(max(v, 0.0))
                out["s2"].append(s2)
                out["sample_sq"].append(((1.0 - tau) * (rss - 2.0 * rsb + rbb) + tau / n, bias))
                out["trend_var"].append(float(trend.var()))
                out["rbb"].append((1.0 - tau) * rbb)
            g += 1
    out["n"], out["p"] = n, p
    return out


def summary(res: dict, lp: np.ndarray, fd: K.FieldData) -> dict:
    """The field mean's Bayesian and plug-in predictives, the design-based
    estimate and the sample mean's model-based error."""
    g = np.array(res["g"])
    w = np.exp(lp[g] - lp[g].max())
    w /= w.sum()
    df = float(res["n"] - res["p"])
    locs, s2, v = np.array(res["pred"]), np.array(res["s2"]), np.array(res["v"])
    mix = SC.MixtureT(w, locs, np.sqrt(s2 * v), df)
    q = mix.quantiles(list(LEVELS))
    top = int(np.argmax(w))
    plug = SC.Normal(locs[top], math.sqrt(s2[top] * v[top]))
    pq = plug.quantiles(list(LEVELS))
    e_sigma2 = s2 * df / (df - 2.0)                      # E[sigma^2 | theta] = S^2 / (n - p - 2)
    sq = np.array([e * a + b * b for e, (a, b) in zip(e_sigma2, res["sample_sq"])])
    n = len(fd.z)
    ybar, s = float(fd.z.mean()), float(fd.z.std(ddof=1))
    se = s / math.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    d2 = float(w @ (e_sigma2 * (1.0 - np.array(res["rbb"])) + np.array(res["trend_var"])))
    return {"mean": mix.mean(), "sd": math.sqrt(mix.var()), "q025": q[0], "q50": q[1], "q975": q[2],
            "plugin_mean": plug.mean(), "plugin_sd": plug.scale, "plugin_q025": pq[0], "plugin_q975": pq[2],
            "sample_mean": ybar, "sample_se": se, "sample_lo": ybar - t * se, "sample_hi": ybar + t * se,
            "sample_se_model": math.sqrt(float(w @ sq)), "d2": d2, "components": len(g)}


# --- sampling effort ---------------------------------------------------------------------------

def planning_grid(cells_xy: np.ndarray, hull_xy: np.ndarray, n: int, cell: float) -> np.ndarray:
    """A square grid of about n points over the field: spacing sqrt(area / n),
    the first point half a spacing in from the hull's lower-left corner, the
    points inside the hull kept (D4)."""
    area = len(cells_xy) * cell * cell
    step = math.sqrt(area / n)
    lo, hi = hull_xy.min(axis=0), hull_xy.max(axis=0)
    xs = np.arange(lo[0] + step / 2, hi[0], step)
    ys = np.arange(lo[1] + step / 2, hi[1], step)
    P = np.array([(x, y) for y in ys for x in xs])
    eq = ConvexHull(hull_xy - hull_xy.mean(axis=0)).equations
    inside = np.all((P - hull_xy.mean(axis=0)) @ eq[:, :2].T + eq[:, 2] <= 1e-9, axis=1)
    return P[inside]


def nearest_cells(points: np.ndarray, cells_xy: np.ndarray) -> np.ndarray:
    return np.argmin(((points[:, None, :] - cells_xy[None, :, :]) ** 2).sum(axis=2), axis=1)


def effort_variance(model: K.Model, pts: np.ndarray, pts_cov: np.ndarray, cells_xy: np.ndarray,
                    cells_cov: np.ndarray, drift: tuple, thetas: list, hist: tuple, exact: bool = False) -> list:
    """v(n, theta), the block-kriging variance per unit sill of the field mean
    from the points pts, at each (a, nu, tau) of thetas."""
    X, X0 = K.design(model.mean, pts, cells_xy, pts_cov, cells_cov, drift)
    xB = X0.mean(axis=0)
    D, D0 = K.distances(pts, pts, model.angle, model.ratio), K.distances(cells_xy, pts, model.angle, model.ratio)
    out, cache = [], {}
    for a, nu, tau in thetas:
        key = (a, nu)
        if key not in cache:
            R = M.correlation(model.family, D, a, nu)
            rb = M.correlation(model.family, D0, a, nu).mean(axis=0)
            rbb = (exact_mean_correlation(cells_xy, model.family, a, nu, model.angle, model.ratio) if exact
                   else mean_correlation(hist, len(cells_xy), model.family, a, nu))
            cache = {key: (R, rb, rbb)}
        R, rb, rbb = cache[key]
        C = (1.0 - tau) * R + tau * np.eye(len(pts))
        k = (1.0 - tau) * rb
        Ci_k, Ci_X = np.linalg.solve(C, k), np.linalg.solve(C, X)
        gv = xB - X.T @ Ci_k
        out.append(max((1.0 - tau) * rbb - float(k @ Ci_k) + float(gv @ np.linalg.solve(X.T @ Ci_X, gv)), 0.0))
    return out


def heavy_points(model: K.Model, res: dict, lp: np.ndarray, mass: float = MASS) -> tuple[list, np.ndarray]:
    """The grid points that carry `mass` of the posterior, their (a, nu, tau)
    and E[sigma^2 | theta] times their renormalised weights."""
    g = np.array(res["g"])
    w = np.exp(lp[g] - lp[g].max())
    w /= w.sum()
    order = np.argsort(-w, kind="stable")
    k = int(np.searchsorted(np.cumsum(w[order]), mass) + 1)
    chosen = np.sort(order[:k])
    a, t, nu = model.grid().arrays()
    df = float(res["n"] - res["p"])
    e = np.array(res["s2"])[chosen] * df / (df - 2.0)
    ww = w[chosen] / w[chosen].sum()
    thetas = [(float(a[g[i]]), None if np.isnan(nu[g[i]]) else float(nu[g[i]]), float(t[g[i]])) for i in chosen]
    return thetas, ww * e


# --- the run -----------------------------------------------------------------------------------

def field_inputs(fields: dict, site: str, model: K.Model, cache: K.Cache) -> dict:
    """The field's cells, its posterior from all its stocks (Stage 4's maps'
    own, from their cache where it is there) and its pairs' histogram."""
    fd = fields[site]
    g = MP.load_grid(site)
    rows, cols, cells_xy, cells_cov = MP.hull_cells(fd.xy, g)
    data = [K.data_key(fd.xy, fd.z, fd.cov)]
    if model.shared:
        data += [K.data_key(fields[o].xy, fields[o].z) for o in sorted(fields) if o != site]

    def posterior():
        drift = MP.all_stocks_drift(fd, model) if model.mean == "drift" else ()
        others = MP.shared_others_all(fields, site, model) if model.shared else None
        return {"drift": list(drift), "logpost": MP.log_posterior(fd, model, drift, others).tolist()}

    post = K.Cache(MP.CACHE, cache.deadline).get(["maps-posterior", MP._CODE, site, model.key()] + data, posterior)
    return {"fd": fd, "cells_xy": cells_xy, "cells_cov": cells_cov, "cell": float(g["cell"]),
            "drift": tuple(post["drift"]), "lp": np.array(post["logpost"], dtype=float),
            "hist": pair_histogram(cells_xy, model.angle, model.ratio), "data": data}


def pair_chunks(model: K.Model, keep: np.ndarray, size: int = PAIRS) -> list[np.ndarray]:
    """The kept grid points in steps of at most `size` correlation functions,
    in the grid's order."""
    per = len(model.grid().taus)
    groups = [np.flatnonzero(keep[i:i + per]) + i for i in range(0, len(keep), per)]
    groups = [g for g in groups if len(g)]
    return [np.concatenate(groups[i:i + size]) for i in range(0, len(groups), size)]


def field_mean(fields: dict, site: str, model: K.Model, cache: K.Cache) -> tuple[dict, dict]:
    f = field_inputs(fields, site, model, cache)
    lp = f["lp"]
    res = None
    for chunk in pair_chunks(model, lp >= lp.max() + np.log(KEEP)):
        mask = np.zeros(len(lp), dtype=bool)
        mask[chunk] = True
        part = cache.get(["block", _CODE, site, model.key(), int(chunk[0]), int(chunk[-1])] + f["data"],
                         lambda: block_grid(f["fd"], model, f["cells_xy"], f["cells_cov"], f["drift"], lp,
                                            f["hist"], mask))
        res = part if res is None else {k: (res[k] + part[k] if isinstance(part[k], list) else part[k]) for k in part}
    return f, res


def effort_rows(site: str, model: K.Model, f: dict, res: dict, cache: K.Cache) -> list:
    thetas, we = heavy_points(model, res, f["lp"])
    starts = [i for i in range(len(thetas)) if i == 0 or thetas[i][:2] != thetas[i - 1][:2]]
    bounds = starts[::PAIRS] + [len(thetas)]
    rows = []
    for n in EFFORT_N:
        pts = planning_grid(f["cells_xy"], f["fd"].xy, n, f["cell"])
        pts_cov = f["cells_cov"][nearest_cells(pts, f["cells_xy"])]
        Xp = K.design(model.mean, pts, pts, pts_cov, pts_cov, f["drift"])[0]
        if len(pts) <= Xp.shape[1] or np.linalg.matrix_rank(Xp) < Xp.shape[1]:
            continue                        # too few points, or too few rows and columns, for the mean model
        v = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            v += cache.get(["effort", _CODE, site, model.key(), n, lo, hi] + f["data"],
                           lambda: effort_variance(model, pts, pts_cov, f["cells_xy"], f["cells_cov"], f["drift"],
                                                   thetas[lo:hi], f["hist"]))
        rows.append({"site": site, "n_target": n, "n": len(pts), "V": float(we @ np.array(v))})
    return rows


def effort_summary(table: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    """n for a 95 % half-width of 5 % of the mean, and for an MDC of 10 % and
    5 % of it: by kriging from a grid (interpolated on log n; NaN past the
    largest grid) and by simple random sampling (D^2 / n, closed form)."""
    rows = []
    for site, g in table.groupby("site"):
        m = means.set_index("site").loc[site]
        logn, logse = np.log(g["n"].to_numpy(float)), 0.5 * np.log(g["V"].to_numpy(float))
        row = {"site": site}
        for name, target in (("pm5", 0.05 * m["mean"] / 1.959964), ("mdc10", 0.10 * m["mean"] / (Z_MDC * math.sqrt(2))),
                             ("mdc5", 0.05 * m["mean"] / (Z_MDC * math.sqrt(2)))):
            below = np.flatnonzero(logse <= math.log(target))
            if len(below) == 0:
                row[f"n_{name}_grid"] = np.nan
            elif below[0] == 0:
                row[f"n_{name}_grid"] = float(g["n"].iloc[0])
            else:
                i = below[0]
                x = np.interp(math.log(target), [logse[i], logse[i - 1]], [logn[i], logn[i - 1]])
                row[f"n_{name}_grid"] = float(math.ceil(math.exp(x)))
            row[f"n_{name}_srs"] = float(math.ceil(m["d2"] / target ** 2))
        rows.append(row)
    return pd.DataFrame(rows)


def run_stage(fields: dict, models: dict, selected, cache: K.Cache, sites=None, log=lambda m: None):
    means, effort = [], []
    for site in (sites or sorted(fields)):
        t0 = time.monotonic()
        model = next(m for m in models[site] if m.name == selected[site])
        f, res = field_mean(fields, site, model, cache)
        row = {"site": site, "model": model.name, "n": len(f["fd"].z), "cells": len(f["cells_xy"]),
               "area_ha": len(f["cells_xy"]) * f["cell"] ** 2 / 1e4, **summary(res, f["lp"], f["fd"])}
        row["mdc"] = Z_MDC * math.sqrt(2.0) * row["sd"]
        row["disagree"] = bool(not (row["q025"] <= row["sample_mean"] <= row["q975"])
                               or not (row["sample_lo"] <= row["mean"] <= row["sample_hi"]))
        if site in RUNNER_UP:
            other = next(m for m in models[site] if m.name == RUNNER_UP[site])
            fo, ro = field_mean(fields, site, other, cache)
            so = summary(ro, fo["lp"], fo["fd"])
            row.update(runner_model=other.name, runner_mean=so["mean"], runner_q025=so["q025"], runner_q975=so["q975"])
        means.append(row)
        effort += effort_rows(site, model, f, res, cache)
        log(f"{site} ({model.name}): {time.monotonic() - t0:.0f} s")
    return pd.DataFrame(means), pd.DataFrame(effort)


def load_inputs() -> tuple:
    fields, record, aniso, boxcox = K.load_inputs()
    models = {s: K.models_for(s, record, aniso, boxcox) for s in sorted(fields)}
    selected = pd.read_parquet(K.OUT_SELECTION).set_index("site")["selected"]
    return fields, models, selected


def main(argv: list[str]) -> int:
    """python -m src.transform.field_means [--budget SECONDS] [--sites A,B]"""
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
    fields, models, selected = load_inputs()
    cache = K.Cache(CACHE, None if budget is None else time.monotonic() + budget)
    try:
        means, effort = run_stage(fields, models, selected, cache, sites, log=print)
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget; run again to continue from the cache.")
        return 4
    if sites:
        print(f"{', '.join(sites)}: in the cache")
        return 0
    effort["se_grid"] = np.sqrt(effort["V"])
    effort["se_srs"] = np.sqrt(effort["site"].map(means.set_index("site")["d2"]) / effort["n"])
    effort["mdc_grid"] = Z_MDC * math.sqrt(2.0) * effort["se_grid"]
    effort["mdc_srs"] = Z_MDC * math.sqrt(2.0) * effort["se_srs"]
    means = means.merge(effort_summary(effort, means), on="site")
    means.to_parquet(OUT, index=False)
    effort.to_parquet(OUT_EFFORT, index=False)
    pd.set_option("display.width", 220)
    print(means.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
