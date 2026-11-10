"""Checkpoint 4's addendum of 10.11.2026 (RUNBOOK, Stage 4, Addendum): rule
K6, this repository's kriging against an independent library, GSTools
(Müller et al. 2022); rule K7, the maps against the cross-validation's own
predictive; and D12's deviation ratios.

They live outside checkpoint4.py so that its code hash, which keys the cache
of the eleven rules of 08.11.2026, stays as it was: checkpoint4.run() imports
this module. K7's recomputations are cached in curated/checkpoint4_cache/,
keyed by this module's code and the data they use.

GSTools takes the parameters in its own terms:

    var = (1 - tau) sill, nugget = tau sill     its variance is the partial sill
    len_scale = a / ln 20                       exponential: R(a) = 0.05
              = a                               spherical: R(a) = 0
              = a / r, with cor(r) = 0.05       Matern, by GSTools' own correlation
    anis = ratio, angles = 90 - angle (deg)     minor over major; anticlockwise from east

with the coordinates in kilometres from the training locations' centroid
(the length scale with them) and (c)'s covariates centred and scaled on the
training locations. Neither changes a kriging predictor; both keep GSTools'
equations well scaled.
"""
from __future__ import annotations

import math
from typing import Optional

import gstools as gs
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.spatial import ConvexHull

try:                        # run as a module from the repository root
    from src.checks import checkpoint1 as C1
    from src.transform import folds as FO, forests as F, kriging as K, maps as MP, patches as P, scores as SC
except ImportError:         # imported with src/* on sys.path (pytest)
    import checkpoint1 as C1
    import folds as FO
    import forests as F
    import kriging as K
    import maps as MP
    import patches as P
    import scores as SC

Rule = C1.Rule
FOLD = (0, 0)                   # the repeat and fold K6 checks, as C2, K1 and K5 recompute
MEAN_MODELS = ("a", "b", "c")
METHODS = tuple(f"{kind}_{m}" for kind in ("kriging", "plugin") for m in MEAN_MODELS) + ("qrf", "rfrk", "baseline")
KM = 1000.0                     # GSTools' unit of length here
TOL_K6_PRED = 1e-8              # relative to the predictor
TOL_K6_VAR = 1e-8               # of the sill
TOL_K7 = 1e-8                   # relative (mean, sd); of the 90 % interval's width (quantiles)
TOL_K7_QRF = 1e-12
K7_CELLS = 50
K7_SEED = FO.SEED + 7
HULL_SLACK = 1e-6               # m: a cell centre this close outside a hull edge counts as inside
GS_MODELS = {"exponential": gs.Exponential, "spherical": gs.Spherical, "matern": gs.Matern}
CODE = K.code_hash(__file__, F.__file__, P.__file__)


# --- K6: GSTools ------------------------------------------------------------------------------

def length_scale(family: str, eff_range: float, nu: Optional[float] = None) -> float:
    """GSTools' length scale for the effective range a: the distance at which
    this repository's correlation is 0.05 (exponential, Matern) or 0 (spherical)."""
    if family == "exponential":
        return eff_range / math.log(20.0)
    if family == "spherical":
        return eff_range
    if family == "matern":
        cor = gs.Matern(dim=2, nu=nu).cor
        r = optimize.brentq(lambda h: cor(h) - 0.05, 1e-8, 1e3, xtol=1e-15, rtol=1e-15, maxiter=500)
        return eff_range / r
    raise ValueError(f"unknown family {family!r}")


def gstools_model(family: str, sill: float, tau: float, eff_range: float, nu: Optional[float] = None,
                  angle: Optional[float] = None, ratio: Optional[float] = None, unit: float = KM):
    """A GSTools covariance model with this repository's parameters, lengths in unit."""
    kw = {"dim": 2, "var": (1.0 - tau) * sill, "nugget": tau * sill,
          "len_scale": length_scale(family, eff_range, nu) / unit}
    if family == "matern":
        kw["nu"] = nu
    if angle is not None:
        kw.update(anis=ratio, angles=math.radians(90.0 - angle))
    return GS_MODELS[family](**kw)


def gstools_krige(mean: str, model, xy: np.ndarray, z: np.ndarray, xy0: np.ndarray,
                  cov: Optional[np.ndarray] = None, cov0: Optional[np.ndarray] = None,
                  unit: float = KM) -> tuple[np.ndarray, np.ndarray]:
    """GSTools' kriging predictor and variance at xy0 from the stocks z at xy:
    ordinary kriging for a constant mean, universal kriging with GSTools'
    quadratic drift, or kriging with the columns of cov as external drift
    (cov0 at xy0)."""
    c = xy.mean(axis=0)
    p, p0 = (xy - c) / unit, (xy0 - c) / unit
    pos, pos0 = (p[:, 0], p[:, 1]), (p0[:, 0], p0[:, 1])
    if mean == "constant" or (mean == "drift" and (cov is None or cov.shape[1] == 0)):
        kr = gs.krige.Ordinary(model, cond_pos=pos, cond_val=z, pseudo_inv=False)
        return kr(pos0, mesh_type="unstructured", return_var=True)
    if mean == "quadratic":
        kr = gs.krige.Universal(model, cond_pos=pos, cond_val=z, drift_functions="quadratic", pseudo_inv=False)
        return kr(pos0, mesh_type="unstructured", return_var=True)
    if mean == "drift":
        mu, sd = cov.mean(axis=0), cov.std(axis=0)
        kr = gs.krige.ExtDrift(model, cond_pos=pos, cond_val=z, ext_drift=((cov - mu) / sd).T, pseudo_inv=False)
        return kr(pos0, mesh_type="unstructured", return_var=True, ext_drift=((cov0 - mu) / sd).T)
    raise ValueError(f"unknown mean model {mean!r}")


def run_result(d: dict, site: str, name: str, cache: Optional[K.Cache] = None) -> dict:
    """The kriging run's result for one field and model in repeat 0, fold 0:
    from its cache, under the key kriging.run_stage gave it, or computed
    afresh as run_stage computes it where the cache lacks it."""
    fields, fd = d["fields"], d["fields"][site]
    group = next(g for g in K.group_models(d["models"][site]) if any(m.name == name for m in g))
    base, model = group[0], next(m for m in group if m.name == name)
    r, k = FOLD
    train, test = fd.split(r, k)
    cache = cache or K.Cache(K.CACHE)
    others = K.shared_others(fields, site, base, r, k, cache) if base.shared else None
    data = K.data_key(fd.xy[train], fd.z[train], fd.cov[train], fd.xy[test], fd.z[test])
    key = ["fold", site, model.key(), r, k, data] + ([K.data_key(others)] if base.shared else [])
    return cache.get_many({name: key}, lambda: K.fold_results(fd, group, train, test, others))[name]


def grid_point(model: K.Model, posterior: dict) -> tuple:
    """The grid point the run recorded as its maximum, as the grid holds it:
    (log a, tau, nu), nu None but for the Matern."""
    grid = model.grid()
    a, tau = posterior["eff_range_max"], posterior["tau_max"]
    lr = min(grid.log_ranges, key=lambda v: abs(math.exp(v) - a))
    t = min(grid.taus, key=lambda v: abs(v - tau))
    nu = None
    if model.family == "matern":
        nu = min(grid.nus, key=lambda v: abs(v - posterior["nu_max"]))
        if abs(nu - posterior["nu_max"]) > 1e-12:
            raise ValueError("the recorded maximum's nu is not on the grid")
    if abs(math.exp(lr) - a) > 1e-9 * a or abs(t - tau) > 1e-12:
        raise ValueError("the recorded maximum is not a grid point")
    return lr, t, nu


def k6_case(d: dict, site: str, name: str, cache: Optional[K.Cache] = None) -> dict:
    """GSTools against the record's plug-in predictive, one field and mean model."""
    fd = d["fields"][site]
    model = next(m for m in d["models"][site] if m.name == name)
    res = run_result(d, site, name, cache)
    lr, tau, nu = grid_point(model, res["posterior"])
    a = math.exp(lr)
    train, test = fd.split(*FOLD)
    drift = K.drift_candidates()[res["drift"]["chosen"]] if model.mean == "drift" else ()
    xy, z, cov, xy0, cov0 = fd.xy[train], fd.z[train], fd.cov[train], fd.xy[test], fd.cov[test]
    X = K.design(model.mean, xy, xy0, cov, cov0, drift)[0]
    one = K.evaluate(K.Grid(model.family, (lr,), (tau,), (nu,)), xy, z, X, angle=model.angle, ratio=model.ratio)
    sill = float(one.s2[0]) / one.df
    gm = gstools_model(model.family, sill, tau, a, nu, model.angle, model.ratio)
    cols = list(drift)
    pred, var = gstools_krige(model.mean, gm, xy, z, xy0, cov[:, cols], cov0[:, cols])
    cv = d["cv"]
    rec = cv[(cv["site"] == site) & (cv["method"] == f"plugin_{name}") & (cv["repeat"] == FOLD[0])
             & (cv["fold"] == FOLD[1])].set_index("location_id").loc[fd.ids[test]]
    mean, v = rec["mean"].to_numpy(float), rec["var"].to_numpy(float)
    same_drift = True
    if model.mean == "drift":
        dr = d["drifts"]
        row = dr[(dr["site"] == site) & (dr["repeat"] == FOLD[0]) & (dr["fold"] == FOLD[1]) & dr["chosen"]]
        same_drift = len(row) == 1 and int(row["candidate"].iloc[0]) == int(res["drift"]["chosen"])
    return {"site": site, "model": name, "family": model.family, "held_out": len(test),
            "eff_range": a, "tau": tau, "nu": np.nan if nu is None else nu, "sill": sill,
            "anisotropy": model.angle is not None, "drift": K.drift_name(drift) if model.mean == "drift" else "",
            "pred": float(np.max(np.abs(pred - mean) / np.abs(mean))),
            "var_of_sill": float(np.max(np.abs(var - v)) / sill),
            "var_relative": float(np.max(np.abs(var - v) / v)), "same_drift": same_drift}


def k6_gstools(d: dict, cache: Optional[K.Cache] = None) -> tuple[Rule, pd.DataFrame]:
    rows, problems = [], []
    for site in sorted(d["fields"]):
        for name in MEAN_MODELS:
            row = k6_case(d, site, name, cache)
            rows.append(row)
            if not row["same_drift"]:
                problems.append(f"{site} ({name}): the cache's drift is not the record's")
            if row["pred"] > TOL_K6_PRED:
                problems.append(f"{site} ({name}): predictor {row['pred']:.1e} relative")
            if row["var_of_sill"] > TOL_K6_VAR:
                problems.append(f"{site} ({name}): variance {row['var_of_sill']:.1e} of the sill")
    t = pd.DataFrame(rows)
    detail = (f"repeat 0, fold 0, {int(t['held_out'].sum()):,} held-out predictions under (a), (b) and (c) in all "
              f"{t['site'].nunique()} fields: GSTools {gs.__version__}, given the run's grid maximum and its sill, "
              f"returns the plug-in's mean within {t['pred'].max():.1e} (relative) and its variance within "
              f"{t['var_of_sill'].max():.1e} of the sill ({t['var_relative'].max():.1e} of itself)")
    return Rule("K6 an independent library", not problems, detail if not problems else "; ".join(problems[:6])), t


# --- K7: the maps -----------------------------------------------------------------------------

def map_cells(g: dict, xy: np.ndarray) -> set:
    """K7's own count of a map's cells: every cell whose centre lies inside the
    convex hull of the locations xy, by the hull's half-planes, and whose
    covariates are all finite, as (row, column)."""
    ny, nx = g["elevation"].shape
    cell = float(g["cell"])
    r, c = np.divmod(np.arange(ny * nx), nx)
    centre = xy.mean(axis=0)
    pts = np.c_[float(g["x0"]) + (c + 0.5) * cell, float(g["y1"]) - (r + 0.5) * cell] - centre
    eq = ConvexHull(xy - centre).equations
    inside = np.all(pts @ eq[:, :2].T + eq[:, 2] <= HULL_SLACK, axis=1)
    r, c = r[inside], c[inside]
    cov = np.column_stack([g[k][r, c].astype(float) for k in K.COVARIATES])
    ok = np.all(np.isfinite(cov), axis=1)
    return set(zip(r[ok].tolist(), c[ok].tolist()))


def cells_from_grid(g: dict, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The centres and covariates of the cells (rows, cols), from Stage 3's grid."""
    cell = float(g["cell"])
    xy = np.c_[float(g["x0"]) + (cols + 0.5) * cell, float(g["y1"]) - (rows + 0.5) * cell]
    return xy, np.column_stack([g[k][rows, cols].astype(float) for k in K.COVARIATES])


def _k7_kriging(d: dict, site: str, model: K.Model, xy0: np.ndarray, cov0: np.ndarray, cache: K.Cache) -> dict:
    """The cross-validation's own predictive at the cells xy0, from all the
    field's stocks: kriging.evaluate on the whole grid, kriging.predictive."""
    fd = d["fields"][site]
    drift = (K.drift_candidates()[K.choose(K.ml_table(model.grid(), fd.xy, fd.z, fd.cov))]
             if model.mean == "drift" else ())
    X, X0 = K.design(model.mean, fd.xy, xy0, fd.cov, cov0, drift)
    post = K.evaluate(model.grid(), fd.xy, fd.z, X, xy0, X0, model.angle, model.ratio)
    lp = post.logpost
    if model.shared:
        for other in sorted(d["fields"]):
            if other != site:
                o = d["fields"][other]
                lp = lp + K.shared_loglik(o, model, np.arange(len(o.z)), cache)
    out = []
    for j in range(len(xy0)):
        mix = K.predictive(post, j, lp)
        q = mix.quantiles(list(MP.LEVELS))
        out.append([mix.mean(), math.sqrt(mix.var()), float(q[0]), float(q[1])])
    return {"drift": list(drift), "kriging": out}


def _k7_qrf(stacked: F.Stacked, Xc: np.ndarray) -> list:
    """QRF at the cells from D7's forest on all the stocks, fitted afresh."""
    rf = F.forest(F.SEED)
    rf.set_params(oob_score=False, n_jobs=-1)
    rf.fit(stacked.X, stacked.y)
    out = []
    for w in F.qrf_weights(rf, stacked.X, Xc):
        dist = SC.Weighted(stacked.y, w)
        q = dist.quantiles(list(MP.LEVELS))
        out.append([dist.mean(), float(q[0]), float(q[1]), float(w.min()), float(abs(w.sum() - 1.0))])
    return out


def k7_maps(d: dict, maps: pd.DataFrame, cache: K.Cache) -> tuple[Rule, pd.DataFrame]:
    rng = np.random.default_rng(K7_SEED)
    selected = d["selection"].set_index("site")["selected"]
    problems, rows, picks = [], [], []
    for site in sorted(d["fields"]):
        fd = d["fields"][site]
        g = MP.load_grid(site)
        m = maps[maps["site"] == site].sort_values(["row", "col"]).reset_index(drop=True)
        got, expected = set(zip(m["row"].tolist(), m["col"].tolist())), map_cells(g, fd.xy)
        if got != expected or len(got) != len(m):
            problems.append(f"{site}: the map holds {len(m)} cells, {len(got - expected)} outside the hull or "
                            f"without covariates, {len(expected - got)} missing")
        s = m.iloc[np.sort(rng.choice(len(m), size=min(K7_CELLS, len(m)), replace=False))]
        xy0, cov0 = cells_from_grid(g, s["row"].to_numpy(), s["col"].to_numpy())
        if np.max(np.abs(xy0 - s[["x", "y"]].to_numpy(float))) > 1e-6:
            problems.append(f"{site}: the map's coordinates are not its cells' centres")
        model = next(mm for mm in d["models"][site] if mm.name == selected[site])
        if (m["model"] != model.name).any():
            problems.append(f"{site}: the map's model is not D5's selection")
        res = cache.get(["k7", CODE, site, model.key(), K.data_key(fd.xy, fd.z, fd.cov, xy0, cov0)],
                        lambda: _k7_kriging(d, site, model, xy0, cov0, cache))
        drift = K.drift_name(tuple(res["drift"])) if model.mean == "drift" else ""
        if (m["drift"] != drift).any():
            problems.append(f"{site}: the map's drift is not the one the AIC chooses from all the stocks")
        ref = np.array(res["kriging"])
        mine = s[[f"kriging_{k}" for k in MP.SUMMARY]].to_numpy(float)
        width = ref[:, 3] - ref[:, 2]
        e_mean = float(np.max(np.abs(mine[:, 0] - ref[:, 0]) / np.abs(ref[:, 0])))
        e_sd = float(np.max(np.abs(mine[:, 1] - ref[:, 1]) / ref[:, 1]))
        e_q = float(np.max(np.abs(mine[:, 2:] - ref[:, 2:]) / width[:, None]))
        if max(e_mean, e_sd, e_q) > TOL_K7:
            problems.append(f"{site}: kriging mean {e_mean:.1e}, sd {e_sd:.1e}, quantiles {e_q:.1e}")
        lon, lat = P.transformer(int(g["epsg"]), "EPSG:4326").transform(xy0[:, 0], xy0[:, 1])
        picks.append((site, s, np.column_stack([lon, lat, cov0])))
        rows.append({"site": site, "model": model.name, "drift": drift, "cells": len(m), "sampled": len(s),
                     "kriging_mean": e_mean, "kriging_sd": e_sd, "kriging_quantiles": e_q})
    Xc = np.vstack([x for _, _, x in picks])
    data = d["data"]
    qrf = np.array(cache.get(["k7-qrf", CODE, K.data_key(data.X, data.y, Xc)], lambda: _k7_qrf(data, Xc)))
    start, e_qrf, w_min, w_sum = 0, 0.0, 0.0, 0.0
    for (site, s, _), row in zip(picks, rows):
        ref = qrf[start:start + len(s)]
        start += len(s)
        mine = s[["qrf_mean", "qrf_q05", "qrf_q95"]].to_numpy(float)
        row["qrf"] = float(np.max(np.abs(mine - ref[:, :3])))
        e_qrf, w_min, w_sum = max(e_qrf, row["qrf"]), min(w_min, float(ref[:, 3].min())), max(w_sum, float(ref[:, 4].max()))
        if row["qrf"] > TOL_K7_QRF:
            problems.append(f"{site}: QRF {row['qrf']:.1e}")
    if w_min < 0 or w_sum > TOL_K7_QRF:
        problems.append(f"QRF's weights: smallest {w_min:.1e}, sum off one by {w_sum:.1e}")
    t = pd.DataFrame(rows)
    detail = (f"the maps hold exactly the {int(t['cells'].sum()):,} cells inside the eight hulls with every covariate "
              f"finite; at {int(t['sampled'].sum())} sampled cells the kriging's mean and standard deviation agree "
              f"with kriging.evaluate and kriging.predictive within {t['kriging_mean'].max():.1e} and "
              f"{t['kriging_sd'].max():.1e} (relative), its 5 and 95 % quantiles within "
              f"{t['kriging_quantiles'].max():.1e} of the 90 % width; QRF's mean and quantiles within {e_qrf:.0e} "
              f"of the forest refitted, its weights non-negative and summing to one within {w_sum:.0e}")
    return Rule("K7 the maps", not problems, detail if not problems else "; ".join(problems[:6])), t


# --- D12 ---------------------------------------------------------------------------------------

def deviation_ratios(cv: pd.DataFrame) -> pd.DataFrame:
    """D12, per field and method: MSDR, the mean of (y - mean)^2 / var, and its
    median, each per repeat and averaged; PICP at 80 and 95 %, the repeats pooled."""
    i80, i95 = SC.CENTRAL.tolist().index(0.8), SC.CENTRAL.tolist().index(0.95)
    rows = []
    for (site, method), g in cv[cv["method"].isin(METHODS)].groupby(["site", "method"]):
        ratio = (g["y"] - g["mean"]) ** 2 / g["var"]
        per = ratio.groupby(g["repeat"]).agg(["mean", "median"]).mean()
        _, picp = SC.coverage(g[K.QCOLS].to_numpy(), g["y"].to_numpy())
        rows.append({"site": site, "method": method, "MSDR": float(per["mean"]),
                     "MSDR_median": float(per["median"]), "PICP80": picp[i80], "PICP95": picp[i95]})
    return pd.DataFrame(rows)
