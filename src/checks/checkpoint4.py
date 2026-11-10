"""Checkpoint 4: kriging and its uncertainty, against two forests (Potash fields).

The rules were fixed in RUNBOOK.md on 08.11.2026, before any kriging,
cross-validation or forest was run: C1 folds, C2 no leakage, K1 kriging
algebra, K2 posterior, K3 predictive distribution, K4 calibration by
simulation, K5 the drift of (c), M1 forests, S1 scores, R1 record. The
second paths are direct: the bordered kriging system with its Lagrange
multipliers (K1), explicit determinants and inverses and Stage 2's own REML
criterion (K2, K5), the trees' leaves and bootstrap samples one by one (M1),
quadrature (S1). The addendum of 10.11.2026 adds K6, the kriging against an
independent library, GSTools, and K7, the maps against the cross-validation's
own predictive, with D12's deviation ratios; they live in independent.py, so
that this module's code hash, which keys the cache, stays as it was.

    python -m src.checks.checkpoint4 [--budget SECONDS] [--only PARTS]
        exit 0: every rule passed
        exit 1: a rule failed
        exit 2: a curated table is missing
        exit 4: stopped at the budget; run again to continue from the cache

Reads curated/ only. The recomputations are cached in
curated/checkpoint4_cache/, keyed by the code and the data they use. BLAS
runs on one thread, as the kriging's run did, so that C2 compares like with
like.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dataclasses
import datetime as dt
import math
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy import integrate, linalg, stats
from scipy.spatial import Delaunay, cKDTree

try:                        # run as a module from the repository root
    from src.checks import checkpoint1 as C1
    from src.transform import anisotropy as A, folds as FO, forests as F, kriging as K, reml as M, scores as SC
except ImportError:         # imported with src/* on sys.path (pytest)
    import anisotropy as A
    import checkpoint1 as C1
    import folds as FO
    import forests as F
    import kriging as K
    import reml as M
    import scores as SC

Rule = C1.Rule
FOLD = (0, 0)                   # the repeat and fold that C2, K1, K2's refined grid, K5 and M1 recompute
RAISE = 1000.0                  # Mg C/ha added to the held-out stocks (C2)
TOL_K1 = 1e-9
TOL_K2_REML = 1e-8              # the spread of log p - l_R
TOL_K2_DIRECT = 1e-8            # relative
TOL_K2_WEIGHTS = 1e-12
TOL_K2_REFINE = 0.02            # of the 90 % interval's width
TOL_K3 = 1e-9
TOL_K4 = 0.03
K4_FIELDS = 100
K4_SITE = "IL-MC"
K4_STEP = 10                    # simulated fields per cached step
K4_SEED = FO.SEED + 4
TOL_K5 = 1e-8
TOL_M1 = 1e-12
TOL_M1_QCP = 0.05
M1_N = 2000
M1_SEED = FO.SEED + 1
TOL_S1_CLOSED = 1e-9
TOL_S1_ENERGY = 1e-12
TOL_S1_MIXTURE = 1e-6
MEAN_MODELS = ("a", "b", "c")
BASE_METHODS = tuple(f"{kind}_{m}" for kind in ("kriging", "plugin") for m in MEAN_MODELS) + ("qrf", "rfrk", "baseline")
BAYESIAN = ("kriging_a", "kriging_b", "kriging_c", "boxcox_a", "log_a", "rfrk")
CACHE = K.CURATED / "checkpoint4_cache"
# What only schedules the cached steps or judges and reports their results; the
# cache keys carry each step's model and data, so these stay out of the hash.
ORCHESTRATION = ("main", "run", "report", "load", "warm", "heavy_first", "recompute", "c1_folds", "c2_leakage",
                 "k1_algebra", "k2_all_stocks", "k2_refined", "k3_predictive", "k4_calibration", "k5_drift",
                 "m1_forests", "s1_scores", "r1_record", "scores_table", "selection_table", "drift_table",
                 "uncertainty_table", "transform_table", "forest_table", "importance_table", "distance_table",
                 "_hull_cells", "_quad_crps")
CODE = K.code_hash(__file__, skip=ORCHESTRATION)


class NotThere(Exception):
    """A curated table Checkpoint 4 reads is missing."""


def load() -> dict:
    try:                    # the addendum's maps, imported here so that they stay out of CODE
        from src.transform import maps as MP
    except ImportError:
        import maps as MP
    need = [K.STOCKS, K.TERRAIN, FO.OUT, M.OUT, A.OUT, M.OUT_BOXCOX, K.OUT, K.OUT_DRIFTS, K.OUT_SELECTION,
            F.OUT, F.OUT_IMPORTANCE, F.OUT_TRAINING, F.OUT_CV, MP.OUT]
    for p in need:
        if not p.exists():
            raise NotThere(p.relative_to(K.ROOT).as_posix())
    fields, record, aniso, boxcox = K.load_inputs()
    stocks, terrain, folds = pd.read_parquet(K.STOCKS), pd.read_parquet(K.TERRAIN), pd.read_parquet(FO.OUT)
    return {"fields": fields, "record": record, "aniso": aniso, "boxcox": boxcox, "stocks": stocks,
            "terrain": terrain, "folds": folds, "data": F.stack(stocks, terrain, folds),
            "cv": pd.read_parquet(F.OUT_CV), "drifts": pd.read_parquet(K.OUT_DRIFTS),
            "selection": pd.read_parquet(K.OUT_SELECTION), "forests": pd.read_parquet(F.OUT),
            "importance": pd.read_parquet(F.OUT_IMPORTANCE), "training": pd.read_parquet(F.OUT_TRAINING),
            "models": {s: K.models_for(s, record, aniso, boxcox) for s in sorted(fields)},
            "bounds": F.own_bounds(record), "maps": pd.read_parquet(MP.OUT)}


def heavy_first(sites) -> list:
    """IL-RT, whose Matern grid is the largest, before the others."""
    return sorted(sites, key=lambda s: (s != "IL-RT", s))


def _key(*parts) -> list:
    return ["checkpoint4", CODE] + list(parts)


def _model(d: dict, site: str, name: str) -> K.Model:
    return next(m for m in d["models"][site] if m.name == name)


def _chosen_drift(d: dict, site: str, repeat: int, fold: int) -> tuple:
    dr = d["drifts"]
    row = dr[(dr["site"] == site) & (dr["repeat"] == repeat) & (dr["fold"] == fold) & dr["chosen"]]
    return K.drift_candidates()[int(row["candidate"].iloc[0])]


# --- C1 ---------------------------------------------------------------------------------------

def c1_folds(d: dict) -> Rule:
    folds, stocks, cv = d["folds"], d["stocks"], d["cv"]
    problems = []
    again = FO.assign(stocks)
    cols = ["site", "location_id", "repeat", "fold", "seed"]
    same = (again[cols].sort_values(cols[:3]).reset_index(drop=True)
            .equals(folds[cols].sort_values(cols[:3]).reset_index(drop=True)))
    if not same:
        problems.append("the recorded folds are not the assignment the seeds give")
    spread = 0
    for site, g in folds.groupby("site"):
        ids = set(stocks.loc[stocks["site"] == site, "location_id"])
        parts = {}
        for r, gr in g.groupby("repeat"):
            if set(gr["location_id"]) != ids or gr["location_id"].duplicated().any():
                problems.append(f"{site} repeat {r}: not every location once")
            sizes = gr["fold"].value_counts().reindex(range(FO.FOLDS), fill_value=0)
            spread = max(spread, int(sizes.max() - sizes.min()))
            if sizes.max() - sizes.min() > 1 or (sizes == 0).any():
                problems.append(f"{site} repeat {r}: fold sizes {sizes.min()} to {sizes.max()}")
            parts[r] = frozenset(frozenset(x["location_id"]) for _, x in gr.groupby("fold"))
        reps = sorted(parts)
        for i, r1 in enumerate(reps):
            for r2 in reps[i + 1:]:
                if parts[r1] == parts[r2]:
                    problems.append(f"{site}: repeats {r1} and {r2} share an assignment")
    m = cv.merge(folds[["site", "location_id", "repeat", "fold"]], on=["site", "location_id", "repeat"],
                 how="left", suffixes=("", "_recorded"))
    wrong = int((m["fold"] != m["fold_recorded"]).sum())
    if wrong:
        problems.append(f"{wrong} predictions not in their recorded fold")
    n_sites = folds["site"].nunique()
    detail = (f"{n_sites} fields x {folds['repeat'].nunique()} repeats: every location in one of ten folds, "
              f"sizes within {spread}; no two repeats share an assignment; the seeds reproduce the record; "
              f"all {len(cv):,} predictions use their recorded fold")
    return Rule("C1 folds", not problems, detail if not problems else "; ".join(problems[:6]))


# --- C2: the fold recomputed, with and without its held-out stocks raised --------------------

def _raised(d: dict, repeat: int, fold: int) -> tuple[dict, F.Stacked]:
    fields = {s: dataclasses.replace(fd, z=fd.z + RAISE * (fd.folds[repeat] == fold))
              for s, fd in d["fields"].items()}
    data = d["data"]
    return fields, dataclasses.replace(data, y=data.y + RAISE * (data.folds[repeat] == fold))


def _kriging_fold(fields: dict, site: str, group: list, repeat: int, fold: int, cache: K.Cache) -> dict:
    fd = fields[site]
    train, test = fd.split(repeat, fold)
    others = K.shared_others(fields, site, group[0], repeat, fold, cache) if group[0].shared else None
    return K.fold_results(fd, group, train, test, others)


def _forest_fold(data: F.Stacked, bounds: dict, repeat: int, fold: int) -> dict:
    res = F.forest_result(data, repeat, fold)
    train, test = data.split(repeat, fold)
    resid = data.y[train] - np.array(res["oob"])
    f_test = np.array(res["prediction"])
    rfrk = {}
    for site in sorted(set(data.site)):
        tr = train[data.site[train] == site]
        pos = np.flatnonzero(data.site[test] == site)
        lo, hi = bounds[site]
        rfrk[site] = F.residual_kriging(data.xy[tr], resid[data.site[train] == site], data.xy[test[pos]],
                                        f_test[pos], data.y[test[pos]], lo, hi)
    base = {}
    for site in sorted(set(data.site)):
        d0 = SC.Weighted(data.y[train][data.site[train] == site])
        base[site] = list(map(float, d0.quantiles()))
    return {"qrf": [q["q"] for q in res["qrf"]], "rfrk": {s: [b["q"] for b in v] for s, v in rfrk.items()},
            "baseline": base, "test": res["test"]}


def recompute(d: dict, cache: K.Cache) -> dict:
    """Fold FOLD of every field, every method, as recorded and with its
    held-out stocks raised by RAISE in every field at once."""
    r, k = FOLD
    out = {}
    for raised in (False, True):
        fields, data = _raised(d, r, k) if raised else (d["fields"], d["data"])
        for site in heavy_first(fields):
            fd = fields[site]
            train, test = fd.split(r, k)
            for group in K.group_models(d["models"][site]):
                key = _key("c2-kriging", site, [m.key() for m in group], raised,
                           K.data_key(fd.xy, fd.z, fd.cov, fd.folds[r]))
                res = cache.get(key, lambda: _kriging_fold(fields, site, group, r, k, cache))
                for name, v in res.items():
                    out[(raised, site, name)] = {"bayes": [b["q"] for b in v["bayes"]],
                                                 "plugin": [p["mean"] for p in v.get("plugin", [])],
                                                 "ids": v["ids"]}
        key = _key("c2-forest", raised, K.data_key(data.X, data.y, data.folds[r]), F._SOURCE)
        out[(raised, "forests")] = cache.get(key, lambda: _forest_fold(data, d["bounds"], r, k))
    return out


def c2_leakage(d: dict, rec: dict) -> Rule:
    r, k = FOLD
    moved, compared = [], 0
    for (raised, site, *rest), v in rec.items():
        if raised or site == "forests":
            continue
        name = rest[0]
        w = rec[(True, site, name)]
        compared += len(v["bayes"]) + len(v["plugin"])
        if v["bayes"] != w["bayes"] or v["plugin"] != w["plugin"]:
            moved.append(f"{site} {name}")
    f0, f1 = rec[(False, "forests")], rec[(True, "forests")]
    compared += 3 * len(f0["qrf"])                      # QRF, RF-RK and the baseline at every held-out location
    for part in ("qrf", "rfrk", "baseline"):
        if f0[part] != f1[part]:
            moved.append(part)
    # the recomputation against the record
    cv = d["cv"]
    worst = 0.0
    for (raised, site, *rest), v in rec.items():
        if raised or site == "forests":
            continue
        name = rest[0]
        method = name if name.startswith(("boxcox", "log")) else f"kriging_{name}"
        sub = cv[(cv["site"] == site) & (cv["method"] == method) & (cv["repeat"] == r)].set_index("location_id")
        q = sub.loc[v["ids"], K.QCOLS].to_numpy()
        worst = max(worst, float(np.max(np.abs(q - np.array(v["bayes"])) / (q[:, -1:] - q[:, :1]))))
    same = "bit for bit" if worst == 0 else f"within {worst:.1e} of the 95 % interval's width"
    detail = (f"repeat {r}, fold {k}, its held-out stocks raised by {RAISE:g} Mg C/ha in all eight fields at "
              f"once: no predictive quantile moved, bit for bit, at {compared:,} predictions (Bayesian and "
              f"plug-in kriging under every model, QRF, RF-RK and the baseline); the recomputation reproduces "
              f"the record {same}")
    return Rule("C2 no leakage", not moved, detail if not moved else "moved: " + ", ".join(moved[:8]))


# --- K1: the bordered system --------------------------------------------------------------------

def bordered(C: np.ndarray, X: np.ndarray, c0: np.ndarray, x0: np.ndarray, z: np.ndarray):
    """The kriging system with its Lagrange multipliers, solved directly."""
    n, p = X.shape
    big = np.block([[C, X], [X.T, np.zeros((p, p))]])
    sol = np.linalg.solve(big, np.concatenate([c0, x0]))
    lam, mu = sol[:n], sol[n:]
    return float(lam @ z), float(1.0 - lam @ c0 - mu @ x0), lam


def _fold_design(d: dict, site: str, model: K.Model, repeat: int, fold: int):
    fd = d["fields"][site]
    train, test = fd.split(repeat, fold)
    drift = _chosen_drift(d, site, repeat, fold) if model.mean == "drift" else ()
    X, X0 = K.design(model.mean, fd.xy[train], fd.xy[test], fd.cov[train], fd.cov[test], drift)
    return fd, train, test, X, X0


def _k1_one(d: dict, site: str, model: K.Model, cache: K.Cache) -> dict:
    r, k = FOLD
    fd, train, test, X, X0 = _fold_design(d, site, model, r, k)
    post = K.evaluate(model.grid(), fd.xy[train], fd.z[train], X, fd.xy[test], X0, model.angle, model.ratio)
    lp = post.logpost
    if model.shared:
        lp = lp + K.shared_others(d["fields"], site, model, r, k, cache)
    g = int(np.argmax(lp))
    a, t, nu = (v[g] for v in post.grid.arrays())
    nu = None if np.isnan(nu) else float(nu)
    D = K.distances(fd.xy[train], fd.xy[train], model.angle, model.ratio)
    C = (1.0 - t) * M.correlation(model.family, D, a, nu)
    np.fill_diagonal(C, 1.0)
    c0 = (1.0 - t) * M.correlation(model.family, K.distances(fd.xy[test], fd.xy[train], model.angle, model.ratio),
                                   a, nu)
    pred_err = var_err = drift_err = 0.0
    for j in range(len(test)):
        p_b, v_b, lam = bordered(C, X, c0[j], X0[j], fd.z[train])
        pred_err = max(pred_err, abs(post.pred[g, j] - p_b) / abs(p_b))
        var_err = max(var_err, abs(post.var1[g, j] - v_b) / abs(v_b))
        drift_err = max(drift_err, float(np.max(np.abs(X.T @ lam - X0[j]))))
    return {"pred": pred_err, "var": var_err, "drift": drift_err, "n": len(test)}


def k1_algebra(d: dict, cache: K.Cache) -> tuple[Rule, pd.DataFrame]:
    rows = []
    for site in heavy_first(d["fields"]):
        for name in MEAN_MODELS:
            model = _model(d, site, name)
            res = cache.get(_key("k1", site, model.key(), FOLD), lambda: _k1_one(d, site, model, cache))
            rows.append({"site": site, "model": name, **res})
    t = pd.DataFrame(rows)
    ok = bool((t["pred"] <= TOL_K1).all() and (t["var"] <= TOL_K1).all() and (t["drift"] <= TOL_K1).all())
    detail = (f"at the grid's maximum in repeat {FOLD[0]}, fold {FOLD[1]}, {int(t['n'].sum())} held-out "
              f"predictions under (a), (b) and (c): predictor within {t['pred'].max():.1e} and variance within "
              f"{t['var'].max():.1e} of the bordered system (relative); the weights reproduce the drift within "
              f"{t['drift'].max():.1e}")
    return Rule("K1 kriging algebra", ok, detail), t


# --- K2: the posterior on every field's grid, from all its stocks -------------------------------

def _all_design(fd: K.FieldData, model: K.Model, drift: tuple) -> np.ndarray:
    if model.mean == "drift":
        return K.design("drift", fd.xy, fd.xy[:1], fd.cov, fd.cov[:1], drift)[0]
    return M.design(fd.xy, model.mean, fd.xy.mean(axis=0))


def _reml_field(fd: K.FieldData, model: K.Model) -> M.Field:
    field = M.Field.of(fd.xy, fd.z, model.mean)
    if model.angle is not None:
        i, j = np.triu_indices(len(fd.z), 1)
        field = dataclasses.replace(field, d=A.rotated_distances(fd.xy[j, 0] - fd.xy[i, 0], fd.xy[j, 1] - fd.xy[i, 1],
                                                                 model.angle, model.ratio))
    return field


def _direct(grid: K.Grid, fd: K.FieldData, X: np.ndarray, model: K.Model) -> list:
    """log p at every grid point with explicit inverses and LU determinants."""
    D = K.distances(fd.xy, fd.xy, model.angle, model.ratio)
    n, p = X.shape
    out = []
    for a, nu, taus in grid.points():
        R = M.correlation(grid.family, D, a, nu)
        for t in taus:
            C = (1.0 - t) * R
            np.fill_diagonal(C, 1.0)
            Ci = np.linalg.inv(C)
            XCX = X.T @ Ci @ X
            b = np.linalg.solve(XCX, X.T @ Ci @ fd.z)
            e = fd.z - X @ b
            out.append(float(-0.5 * np.linalg.slogdet(C)[1] - 0.5 * np.linalg.slogdet(XCX)[1]
                             - 0.5 * (n - p) * math.log(float(e @ Ci @ e))))
    return out


def all_stocks_drift(fd: K.FieldData, model: K.Model) -> tuple:
    return K.drift_candidates()[K.choose(K.ml_table(model.grid(), fd.xy, fd.z, fd.cov))]


def _k2_part(d: dict, site: str, model: K.Model, part: str, cache: K.Cache):
    """One field's and model's posterior from all its stocks ("post"), Stage 2's
    l_R at every grid point ("reml") or the explicit implementation ("direct");
    the last two one value of nu at a time, as cached steps."""
    fd = d["fields"][site]
    data = K.data_key(fd.xy, fd.z, fd.cov)
    drift = cache.get(_key("k2-drift", site, model.key(), data),
                      lambda: list(all_stocks_drift(fd, model))) if model.mean == "drift" else []
    X = _all_design(fd, model, tuple(drift))
    grid = model.grid()
    if part == "post":
        def compute():
            post = K.evaluate(grid, fd.xy, fd.z, X, angle=model.angle, ratio=model.ratio,
                              coefficients=model.mean == "drift")
            out = {"logpost": list(map(float, post.logpost)), "s2": list(map(float, post.s2))}
            if model.mean == "drift":
                out["coefficients"] = K.coefficient_summary(post)
                out["drift"] = list(drift)
            return out
        return cache.get(_key("k2", part, site, model.key(), data), compute)

    def chunk(sub: K.Grid) -> list:
        if part == "reml":
            field = _reml_field(fd, model)
            a, t, nu = sub.arrays()
            return [float(M.reml_profile(field, model.family, ai, ti, None if np.isnan(ni) else ni)[0])
                    for ai, ti, ni in zip(a, t, nu)]
        return _direct(sub, fd, X, model)
    out = []
    for i, nu in enumerate(grid.nus):         # nu outermost: the chunks follow the grid's order
        sub = K.Grid(grid.family, grid.log_ranges, grid.taus, (nu,))
        out += cache.get(_key("k2", part, site, model.key(), data, i), lambda: chunk(sub))
    return out


def k2_all_stocks(d: dict, cache: K.Cache) -> tuple[Rule, dict]:
    """K2's first three parts, and the posterior summaries the report gives."""
    spread_reml, direct_err, weight_err, summaries = 0.0, 0.0, 0.0, {}
    shared_lp = shared_reml = None
    for site in heavy_first(d["fields"]):
        for name in MEAN_MODELS:
            model = _model(d, site, name)
            post = _k2_part(d, site, model, "post", cache)
            lp = np.array(post["logpost"])
            direct = np.array(_k2_part(d, site, model, "direct", cache))
            direct_err = max(direct_err, float(np.max(np.abs(lp - direct) / np.abs(direct))))
            if name in ("a", "b") and not model.shared:
                diff = lp - np.array(_k2_part(d, site, model, "reml", cache))
                spread_reml = max(spread_reml, float(np.ptp(diff)))
            if model.shared:
                continue
            w = np.exp(lp - lp.max())
            weight_err = max(weight_err, abs(float((w / w.sum()).sum()) - 1.0))
            summaries[(site, name)] = (model.grid(), lp, post)
    # the shared posterior under (b): every field's quadratic on the shared grid
    shared = next(m for s in sorted(d["models"]) for m in d["models"][s] if m.shared)
    for site in sorted(d["fields"]):
        model = dataclasses.replace(shared, name="b_shared")
        post = _k2_part(d, site, model, "post", cache)
        lp = np.array(post["logpost"])
        direct = np.array(_k2_part(d, site, model, "direct", cache))
        direct_err = max(direct_err, float(np.max(np.abs(lp - direct) / np.abs(direct))))
        reml = np.array(_k2_part(d, site, model, "reml", cache))
        shared_lp = lp if shared_lp is None else shared_lp + lp
        shared_reml = reml if shared_reml is None else shared_reml + reml
    spread_reml = max(spread_reml, float(np.ptp(shared_lp - shared_reml)))
    w = np.exp(shared_lp - shared_lp.max())
    weight_err = max(weight_err, abs(float((w / w.sum()).sum()) - 1.0))
    for site in sorted(d["fields"]):
        if _model(d, site, "b").shared:
            summaries[(site, "b")] = (shared.grid(), shared_lp, None)
    ok = spread_reml <= TOL_K2_REML and direct_err <= TOL_K2_DIRECT and weight_err <= TOL_K2_WEIGHTS
    detail = (f"from all the stocks, on every grid: log p - Stage 2's l_R spreads {spread_reml:.1e} under (a) "
              f"and (b), the shared posterior included; explicit inverses agree within {direct_err:.1e} "
              f"(relative) under (a), (b) and (c); the weights sum to one within {weight_err:.0e}")
    return Rule("K2 posterior (all stocks)", ok, detail), summaries


def _refined_quantiles(d: dict, site: str, model: K.Model, cache: K.Cache) -> list:
    r, k = FOLD
    fd, train, test, X, X0 = _fold_design(d, site, model, r, k)
    grid = model.grid(refine=2)
    post = K.evaluate(grid, fd.xy[train], fd.z[train], X, fd.xy[test], X0, model.angle, model.ratio)
    lp = post.logpost
    if model.shared:
        for other in sorted(d["fields"]):
            if other == site:
                continue
            o = d["fields"][other]
            tr = o.split(r, k)[0]
            Xo = M.design(o.xy[tr], "quadratic", o.xy[tr].mean(axis=0))
            lp = lp + np.array(cache.get(_key("k2-refined-shared", other, model.key(), FOLD,
                                              K.data_key(o.xy[tr], o.z[tr])),
                                         lambda: list(map(float, K.evaluate(grid, o.xy[tr], o.z[tr], Xo).logpost))))
    return [list(map(float, K.predictive(post, j, lp).quantiles([0.05, 0.5, 0.95]))) for j in range(len(test))]


def k2_refined(d: dict, cache: K.Cache) -> tuple[Rule, pd.DataFrame]:
    r, k = FOLD
    cv, rows = d["cv"], []
    for site in heavy_first(d["fields"]):
        fd = d["fields"][site]
        test = fd.split(r, k)[1]
        for name in MEAN_MODELS:
            model = _model(d, site, name)
            q2 = np.array(cache.get(_key("k2-refined", site, model.key(), FOLD,
                                         K.data_key(fd.xy, fd.z, fd.cov, fd.folds[r])),
                                    lambda: _refined_quantiles(d, site, model, cache)))
            sub = cv[(cv["site"] == site) & (cv["method"] == f"kriging_{name}") & (cv["repeat"] == r)]
            q1 = sub.set_index("location_id").loc[fd.ids[test], ["q0.050", "q0.500", "q0.950"]].to_numpy()
            move = np.abs(q2 - q1).max(axis=1) / (q1[:, 2] - q1[:, 0])
            rows.append({"site": site, "model": name, "grid": model.grid().size,
                         "refined": model.grid(refine=2).size, "largest_move": float(move.max())})
    t = pd.DataFrame(rows)
    ok = bool((t["largest_move"] <= TOL_K2_REFINE).all())
    worst = t.loc[t["largest_move"].idxmax()]
    detail = (f"twice the points along every axis, repeat {r}, fold {k}: the 5, 50 and 95 % quantiles move at "
              f"most {worst['largest_move']:.2%} of the 90 % interval's width ({worst['site']}, "
              f"({worst['model']})), the limit {TOL_K2_REFINE:.0%}")
    return Rule("K2 posterior (grid resolution)", ok, detail), t


# --- K3 ----------------------------------------------------------------------------------------

def k3_predictive(d: dict) -> Rule:
    cv = d["cv"]
    b = cv[cv["method"].isin(BAYESIAN)]
    err = float(b["cdf_error"].max())
    m = cv[cv["method"].isin(["kriging_a", "kriging_b", "kriging_c", "rfrk"])]
    rel = float((np.abs(m["mean"] - m["mean_post"]) / np.abs(m["mean_post"])).max())
    ok = bool(err <= TOL_K3 and rel <= TOL_K3 and b["cdf_error"].notna().all())
    detail = (f"at all {len(b):,} Bayesian predictions (kriging, its transforms and RF-RK) the distribution "
              f"function returns the levels at the quantiles within {err:.1e}; at {len(m):,} the mean is the "
              f"posterior-weighted mean of the kriging predictors within {rel:.1e} (relative); the closed "
              f"Student-t form is a unit test")
    return Rule("K3 predictive distribution", ok, detail)


# --- K4: calibration by simulation ----------------------------------------------------------------

def _k4_step(d: dict, start: int) -> dict:
    fd = d["fields"][K4_SITE]
    lo, hi = d["bounds"][K4_SITE]
    grid = K.make_grid("exponential", lo, hi)
    a_all, t_all, _ = grid.arrays()
    D = K.distances(fd.xy, fd.xy)
    folds = fd.folds[0]
    pit, plug = [], []
    for s in range(start, min(start + K4_STEP, K4_FIELDS)):
        rng = np.random.default_rng(K4_SEED + s)
        g = int(rng.integers(grid.size))
        a, t = a_all[g], t_all[g]
        C = (1.0 - t) * M.correlation("exponential", D, a)
        np.fill_diagonal(C, 1.0)
        z = linalg.cholesky(C, lower=True) @ rng.standard_normal(len(fd.z))
        for k in range(FO.FOLDS):
            train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            X, X0 = K.design("constant", fd.xy[train], fd.xy[test])
            post = K.evaluate(grid, fd.xy[train], z[train], X, fd.xy[test], X0)
            for j, i in enumerate(test):
                pit.append(float(K.predictive(post, j).cdf(z[i])[0]))
                nd = K.plug_in(post, j)
                plug.append(float(nd.cdf(z[i])[0]))
    return {"pit": pit, "plug": plug}


def k4_calibration(d: dict, cache: K.Cache) -> tuple[Rule, pd.DataFrame]:
    pit, plug = [], []
    fd = d["fields"][K4_SITE]
    for start in range(0, K4_FIELDS, K4_STEP):
        res = cache.get(_key("k4", start, K4_STEP, K4_SEED, K.data_key(fd.xy, fd.folds[0]), d["bounds"][K4_SITE]),
                        lambda: _k4_step(d, start))
        pit += res["pit"]
        plug += res["plug"]
    pit, plug = np.array(pit), np.array(plug)

    def cover(u):
        qcp = np.array([np.mean(u <= lev) for lev in SC.LEVELS])
        picp = {p: float(np.mean(np.abs(u - 0.5) <= p / 2)) for p in (0.5, 0.9)}
        return qcp, picp
    qcp, picp = cover(pit)
    qcp_p, picp_p = cover(plug)
    dev = float(np.max(np.abs(qcp - SC.LEVELS)))
    ok = dev <= TOL_K4 and all(abs(picp[p] - p) <= TOL_K4 for p in picp)
    t = pd.DataFrame({"level": SC.LEVELS, "qcp_bayes": qcp, "qcp_plugin": qcp_p})
    detail = (f"{K4_FIELDS} fields at {K4_SITE}'s {len(fd.z)} locations, parameters from D3's prior, "
              f"{len(pit):,} held-out values: the Bayesian QCP within {dev:.3f} of its levels, PICP "
              f"{picp[0.5]:.3f} at 50 % and {picp[0.9]:.3f} at 90 % (computed from the PIT, which is the same "
              f"thing for a continuous predictive); plug-in kriging: QCP within "
              f"{np.max(np.abs(qcp_p - SC.LEVELS)):.3f}, PICP {picp_p[0.5]:.3f} and {picp_p[0.9]:.3f}")
    return Rule("K4 calibration by simulation", ok, detail), t


# --- K5 ----------------------------------------------------------------------------------------

def _k5_direct(d: dict, site: str) -> dict:
    r, k = FOLD
    fd = d["fields"][site]
    train = fd.split(r, k)[0]
    xy, z, cov = fd.xy[train], fd.z[train], fd.cov[train]
    dr = d["drifts"]
    rows = dr[(dr["site"] == site) & (dr["repeat"] == r) & (dr["fold"] == k)].sort_values("candidate")
    D = K.distances(xy, xy)
    n = len(z)
    inverses = {}
    worst = 0.0
    for cand, (_, row) in zip(K.drift_candidates(), rows.iterrows()):
        key = (row["eff_range"], row["tau"])
        if key not in inverses:
            C = (1.0 - row["tau"]) * M.correlation("exponential", D, row["eff_range"])
            np.fill_diagonal(C, 1.0)
            inverses[key] = (np.linalg.inv(C), np.linalg.slogdet(C)[1])
        Ci, logdet = inverses[key]
        X = np.column_stack([np.ones(n)] + [cov[:, i] for i in cand])        # raw covariates
        b = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ z)
        e = z - X @ b
        ll = -0.5 * (n * math.log(2.0 * math.pi * float(e @ Ci @ e) / n) + logdet + n)
        worst = max(worst, abs(ll - row["loglik"]) / abs(ll))
    return {"worst": worst}


def k5_drift(d: dict, cache: K.Cache) -> Rule:
    dr = d["drifts"]
    problems = []
    groups = [set(range(s, s + len(g))) for s, g in zip(np.cumsum([0] + [len(g) for g in K.ATTRIBUTES[:-1]]),
                                                         K.ATTRIBUTES)]
    names = [K.drift_name(c) for c in K.drift_candidates()]
    for c in K.drift_candidates():
        if any(len(set(c) & g) > 1 for g in groups):
            problems.append(f"candidate {K.drift_name(c)} takes two scales of one attribute")
    n_folds = 0
    for (site, r, k), g in dr.groupby(["site", "repeat", "fold"]):
        n_folds += 1
        if len(g) != 256 or list(g.sort_values("candidate")["drift"]) != names:
            problems.append(f"{site} {r}/{k}: not the 256 candidates")
            continue
        best = g.sort_values(["aic", "p", "candidate"], kind="stable").iloc[0]
        if not bool(best["chosen"]) or int(g["chosen"].sum()) != 1:
            problems.append(f"{site} {r}/{k}: the drift used is not the lowest AIC")
    worst = 0.0
    for site in sorted(d["fields"]):
        fd = d["fields"][site]
        res = cache.get(_key("k5", site, FOLD, K.data_key(fd.xy, fd.z, fd.cov, fd.folds[FOLD[0]])),
                        lambda: _k5_direct(d, site))
        worst = max(worst, res["worst"])
    if worst > TOL_K5:
        problems.append(f"the maximum log-likelihoods differ by {worst:.1e} (relative)")
    detail = (f"in all {n_folds} training folds the drift used has the lowest maximum-likelihood AIC of the "
              f"256 candidates, none with two scales of one attribute; in repeat {FOLD[0]}, fold {FOLD[1]} of "
              f"every field, explicit inverses on the raw covariates reproduce every candidate's maximum "
              f"log-likelihood within {worst:.1e} (relative)")
    return Rule("K5 the drift of (c)", not problems, detail if not problems else "; ".join(problems[:6]))


# --- M1 ----------------------------------------------------------------------------------------

def _m1_forest(d: dict) -> dict:
    data = d["data"]
    r, k = FOLD
    train, test = data.split(r, k)
    rf = F.forest(F.seed_of(r, k)).fit(data.X[train], data.y[train])
    W = F.qrf_weights(rf, data.X[train], data.X[test])
    leaf_train, leaf_test = rf.apply(data.X[train]), rf.apply(data.X[test])
    W2 = np.zeros_like(W)
    for t in range(leaf_train.shape[1]):                 # tree by tree, location by location
        for j in range(len(test)):
            members = np.flatnonzero(leaf_train[:, t] == leaf_test[j, t])
            W2[j, members] += 1.0 / len(members)
    W2 /= leaf_train.shape[1]
    oob = F.oob_from_trees(rf, data.X[train])
    resid = data.y[train] - oob
    f_test = rf.predict(data.X[test])
    rfrk = {}
    for site in sorted(set(data.site)):
        tr = train[data.site[train] == site]
        pos = np.flatnonzero(data.site[test] == site)
        lo, hi = d["bounds"][site]
        rfrk[site] = [b["q"] for b in F.residual_kriging(data.xy[tr], resid[data.site[train] == site],
                                                         data.xy[test[pos]], f_test[pos], data.y[test[pos]], lo, hi)]
    return {"weights": float(np.max(np.abs(W - W2))), "oob": float(np.max(np.abs(oob - rf.oob_prediction_))),
            "rfrk": rfrk}


def _m1_simulation() -> dict:
    rng = np.random.default_rng(M1_SEED)
    x = rng.uniform(0.0, 1.0, (2 * M1_N, 1))
    y = 2.0 * x[:, 0] + rng.standard_normal(2 * M1_N)
    rf = F.forest(M1_SEED)
    rf.set_params(max_features=1, oob_score=False)
    rf.fit(x[:M1_N], y[:M1_N])
    W = F.qrf_weights(rf, x[:M1_N], x[M1_N:])
    below = np.array([SC.Weighted(y[:M1_N], w).quantiles() >= y[M1_N + j] for j, w in enumerate(W)])
    return {"qcp": list(map(float, below.mean(axis=0)))}


def m1_forests(d: dict, cache: K.Cache) -> Rule:
    problems = []
    fo = d["forests"]
    q = fo[fo["method"] == "qrf"]
    sum_err = float(np.abs(q["weight_sum"] - 1.0).max())
    if (q["weight_min"] < 0).any() or sum_err > TOL_M1:
        problems.append(f"QRF weights negative or not summing to one ({sum_err:.1e})")
    # the training sets
    tr, folds = d["training"], d["folds"]
    leaks = 0
    for (r, k), g in tr.groupby(["repeat", "fold"]):
        f = folds[folds["repeat"] == r].set_index(["site", "location_id"])["fold"]
        lab = f.loc[list(zip(g["site"], g["location_id"]))].to_numpy()
        leaks += int((lab == k).sum())
        if len(g) != int((f != k).sum()):
            problems.append(f"forest {r}/{k} was not trained on every other location")
    if leaks:
        problems.append(f"{leaks} held-out locations in forests' training sets")
    data = d["data"]
    res = cache.get(_key("m1-forest", FOLD, K.data_key(data.X, data.y, data.folds[FOLD[0]]), F._SOURCE),
                    lambda: _m1_forest(d))
    if res["weights"] > TOL_M1 or res["oob"] > TOL_M1:
        problems.append(f"loop-based weights {res['weights']:.1e}, out-of-bag predictions {res['oob']:.1e}")
    cv = d["cv"]
    worst_rk = 0.0
    r, k = FOLD
    for site, qs in res["rfrk"].items():
        sub = cv[(cv["site"] == site) & (cv["method"] == "rfrk") & (cv["repeat"] == r) & (cv["fold"] == k)]
        fd = d["fields"][site]
        qrec = sub.set_index("location_id").loc[fd.ids[fd.split(r, k)[1]], K.QCOLS].to_numpy()
        worst_rk = max(worst_rk, float(np.max(np.abs(qrec - np.array(qs)) / (qrec[:, -1:] - qrec[:, :1]))))
    if worst_rk > TOL_K3:
        problems.append(f"RF-RK from the trees' own out-of-bag residuals differs by {worst_rk:.1e}")
    sim = cache.get(_key("m1-sim", M1_N, M1_SEED, F._SOURCE), _m1_simulation)
    qcp = np.array(sim["qcp"])
    inside = (SC.LEVELS >= 0.05 - 1e-12) & (SC.LEVELS <= 0.95 + 1e-12)
    dev = float(np.max(np.abs(qcp[inside] - SC.LEVELS[inside])))
    if dev > TOL_M1_QCP:
        problems.append(f"QRF's QCP on the simulated regression is off by {dev:.3f}")
    detail = (f"{len(q):,} QRF predictions: weights non-negative, summing to one within {sum_err:.0e}; in "
              f"repeat {r}, fold {k}, tree-by-tree weights agree within {res['weights']:.0e} and out-of-bag "
              f"predictions from the trees' own bootstrap samples within {res['oob']:.0e}, and RF-RK from them "
              f"reproduces the record within {worst_rk:.1e} of the interval; no forest's training set holds a "
              f"held-out location; on y = 2x + noise ({M1_N:,} + {M1_N:,} points) QRF's QCP is within "
              f"{dev:.3f} of every level from 5 to 95 %")
    return Rule("M1 forests", not problems, detail if not problems else "; ".join(problems[:6]))


# --- S1 ----------------------------------------------------------------------------------------

def _quad_crps(cdf, y, lo, hi):
    left = integrate.quad(lambda x: cdf(x) ** 2, lo, y, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
    right = integrate.quad(lambda x: (1.0 - cdf(x)) ** 2, y, hi, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
    return left + right


def s1_scores(d: Optional[dict] = None) -> Rule:
    errs = {}
    nd = stats.norm(60.0, 8.0)
    errs["normal"] = max(abs(SC.crps_normal(y, 60.0, 8.0) / _quad_crps(nd.cdf, y, -100, 220) - 1)
                         for y in (41.0, 60.0, 83.5))
    td = stats.t(45.0, loc=60.0, scale=8.0)
    errs["t"] = max(abs(SC.crps_t(y, 60.0, 8.0, 45.0) / _quad_crps(td.cdf, y, -1e4, 1e4) - 1)
                    for y in (41.0, 60.0, 83.5))
    rng = np.random.default_rng(3)
    e = SC.Weighted(rng.normal(60, 9, 40), rng.uniform(0, 1, 40))
    errs["energy"] = max(abs(e.crps(y) - e.crps_integral(y)) for y in (30.0, 59.0, 61.5, 95.0))
    m = SC.MixtureT([1.0], [60.0], [8.0], 45.0)
    errs["mixture"] = max(abs(m.crps(y) / SC.crps_t(y, 60.0, 8.0, 45.0) - 1) for y in (41.0, 60.0, 83.5))
    q = np.tile(SC.LEVELS * 10.0, (4, 1))
    qcp, picp = SC.coverage(q, np.array([0.1, 2.0, 5.0, 9.9]))
    hand = qcp[SC.level_index(0.5)] == 0.75 and picp[-1] == 0.5 and qcp[0] == 0.25    # 0.1 <= 0.25
    ok = (errs["normal"] <= TOL_S1_CLOSED and errs["t"] <= TOL_S1_CLOSED and errs["energy"] <= TOL_S1_ENERGY
          and errs["mixture"] <= TOL_S1_MIXTURE and hand)
    detail = (f"the CRPS of a normal and of a Student-t against quadrature within {errs['normal']:.0e} and "
              f"{errs['t']:.0e} (relative), the weighted empirical's energy form against its integral within "
              f"{errs['energy']:.0e}, the one-component t-mixture's numerical CRPS within {errs['mixture']:.0e} "
              f"of the closed form; PICP and QCP reproduce the hand-worked case")
    return Rule("S1 scores", ok, detail)


# --- R1 ----------------------------------------------------------------------------------------

def r1_record(d: dict) -> Rule:
    cv, problems = d["cv"], []
    n = d["stocks"].groupby("site").size()
    for site in sorted(d["fields"]):
        methods = list(BASE_METHODS) + [m.name for m in d["models"][site] if m.lam is not None]
        for method in methods:
            sub = cv[(cv["site"] == site) & (cv["method"] == method)]
            if len(sub) != 3 * n[site] or sub.duplicated(["repeat", "location_id"]).any():
                problems.append(f"{site} {method}: {len(sub)} rows for {3 * n[site]}")
    extra = set(cv["method"]) - set(BASE_METHODS) - {"boxcox_a", "log_a"}
    if extra:
        problems.append(f"unexpected methods {sorted(extra)}")
    q = cv[K.QCOLS].to_numpy()
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(cv[["mean", "crps"]].to_numpy())):
        problems.append("a value is not finite")
    if np.any(np.diff(q, axis=1) < 0):
        problems.append(f"{int((np.diff(q, axis=1) < 0).any(axis=1).sum())} rows with decreasing quantiles")
    if cv["fold"].isna().any():
        problems.append("a row without its fold")
    dr = d["drifts"]
    want = {(s, r, k) for s in d["fields"] for r in range(FO.REPEATS) for k in range(FO.FOLDS)}
    have = {tuple(x) for x in dr.loc[dr["chosen"], ["site", "repeat", "fold"]].itertuples(index=False)}
    if have != want:
        problems.append(f"(c)'s drift recorded for {len(have)} of {len(want)} folds")
    again = K.selection(cv)
    sel = d["selection"].set_index("site")["selected"]
    if not again.set_index("site")["selected"].equals(sel.reindex(again["site"])):
        problems.append("the recorded selection is not D5's")
    detail = (f"curated/potash_cv.parquet: {len(cv):,} rows, one per repeat, method and location for "
              f"{cv['method'].nunique()} methods, every value finite and the quantiles non-decreasing; the drift "
              f"of every field, repeat and fold; D5's selection reproduced")
    return Rule("R1 record", not problems, detail if not problems else "; ".join(problems[:6]))


# --- reported, not judged -----------------------------------------------------------------------

def scores_table(d: dict) -> pd.DataFrame:
    cv, rows = d["cv"], []
    for (site, method), g in cv.groupby(["site", "method"]):
        per = []
        for r, gr in g.groupby("repeat"):
            p = SC.point_scores(gr["mean"], gr["y"])
            per.append({"me": p["me"], "rmse": p["rmse"], "mec": p["mec"], "crps": gr["crps"].mean()})
        per = pd.DataFrame(per).mean()
        _, picp = SC.coverage(g[K.QCOLS].to_numpy(), g["y"].to_numpy())
        lo, hi = SC.central_bounds(0.9)
        rows.append({"site": site, "method": method, "ME": per["me"], "RMSE": per["rmse"], "MEC": per["mec"],
                     "CRPS": per["crps"], "PICP50": picp[SC.CENTRAL.tolist().index(0.5)],
                     "PICP90": picp[SC.CENTRAL.tolist().index(0.9)],
                     "width90": float((g[K.QCOLS[hi]] - g[K.QCOLS[lo]]).mean())})
    t = pd.DataFrame(rows)
    base = t[t["method"] == "baseline"].set_index("site")["CRPS"]
    t["CRPS_baseline"] = t["site"].map(base)
    return t.round(3)


def selection_table(d: dict) -> pd.DataFrame:
    return d["selection"].round(3)


def drift_table(d: dict, summaries: dict, cache: K.Cache) -> tuple[pd.DataFrame, pd.DataFrame]:
    dr = d["drifts"]
    chosen = dr[dr["chosen"]]
    freq = []
    for site, g in chosen.groupby("site"):
        row = {"site": site}
        for c in K.COVARIATES:
            row[c] = int(sum(c in name.split("+") for name in g["drift"]))
        row["constant_alone"] = int((g["drift"] == "constant").sum())
        freq.append(row)
    coef = []
    for site in sorted(d["fields"]):
        post = _k2_part(d, site, _model(d, site, "c"), "post", cache)
        names = ["constant"] + [K.COVARIATES[i] for i in post["drift"]]
        for name, c in zip(names, post["coefficients"]):
            coef.append({"site": site, "term": name, "mean": c["mean"], "q05": c["q05"], "q95": c["q95"]})
    return pd.DataFrame(freq), pd.DataFrame(coef).round(3)


def uncertainty_table(d: dict, summaries: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv, rows = d["cv"], []
    for site in sorted(d["fields"]):
        for name in MEAN_MODELS:
            b = cv[(cv["site"] == site) & (cv["method"] == f"kriging_{name}")].set_index(["repeat", "location_id"])
            p = cv[(cv["site"] == site) & (cv["method"] == f"plugin_{name}")].set_index(["repeat", "location_id"])
            ratio = (b["var"] / p.loc[b.index, "var"]).median()
            _, pb = SC.coverage(b[K.QCOLS].to_numpy(), b["y"].to_numpy())
            _, pp = SC.coverage(p[K.QCOLS].to_numpy(), p["y"].to_numpy())
            i50, i90 = SC.CENTRAL.tolist().index(0.5), SC.CENTRAL.tolist().index(0.9)
            rows.append({"site": site, "model": name, "PICP90_bayes": pb[i90], "PICP90_plugin": pp[i90],
                         "PICP50_bayes": pb[i50], "PICP50_plugin": pp[i50], "variance_ratio": ratio})
    post_rows = []
    rec = d["record"]
    for (site, name), (grid, lp, _) in sorted(summaries.items()):
        w = np.exp(lp - lp.max())
        w /= w.sum()
        a, t, _ = grid.arrays()
        qa, qt = K.weighted_quantiles(a, w, (0.05, 0.5, 0.95)), K.weighted_quantiles(t, w, (0.05, 0.5, 0.95))
        row = {"site": site, "model": name, "range_q05": qa[0], "range_median": qa[1], "range_q95": qa[2],
               "tau_q05": qt[0], "tau_median": qt[1], "tau_q95": qt[2]}
        if name in ("a", "b"):
            mean = K.MEAN_OF[name]
            r2 = rec[rec["for_stage4"].astype(bool) & (rec["site"] == site) & (rec["mean"] == mean)].iloc[0]
            row.update({"stage2_range": r2["eff_range"], "stage2_range_lo": r2["eff_range_lo"],
                        "stage2_range_hi": r2["eff_range_hi"], "stage2_tau": r2["tau"],
                        "stage2_tau_lo": r2["tau_lo"], "stage2_tau_hi": r2["tau_hi"]})
        post_rows.append(row)
    return pd.DataFrame(rows).round(3), pd.DataFrame(post_rows).round(3)


def transform_table(scores: pd.DataFrame, d: dict) -> pd.DataFrame:
    cv = d["cv"]
    t = scores[scores["method"].isin(["kriging_a", "boxcox_a", "log_a"])
               & scores["site"].isin(cv.loc[cv["method"] == "boxcox_a", "site"].unique())]
    out = cv[cv["method"].isin(["boxcox_a", "log_a"])].groupby(["site", "method"])["outside"].max()
    t = t.merge(out.rename("largest_cut").reset_index(), on=["site", "method"], how="left")
    t = t.assign(largest_cut=t["largest_cut"].round(4))
    return t[["site", "method", "CRPS", "RMSE", "MEC", "PICP90", "width90", "largest_cut"]]


def forest_table(d: dict) -> pd.DataFrame:
    cv, sel = d["cv"], d["selection"].set_index("site")["selected"]
    rows = []
    for site in sorted(d["fields"]):
        k = cv[(cv["site"] == site) & (cv["method"] == f"kriging_{sel[site]}")]
        kk = k.groupby("location_id").agg(crps=("crps", "mean"))
        kr = k.assign(se=(k["mean"] - k["y"]) ** 2).groupby("repeat")["se"].mean() ** 0.5
        for method in ("qrf", "rfrk"):
            f = cv[(cv["site"] == site) & (cv["method"] == method)]
            ff = f.groupby("location_id").agg(crps=("crps", "mean"))
            fr = f.assign(se=(f["mean"] - f["y"]) ** 2).groupby("repeat")["se"].mean() ** 0.5
            diff = ff["crps"] - kk.loc[ff.index, "crps"]
            rows.append({"site": site, "method": method, "selected_kriging": sel[site],
                         "CRPS_difference": float(diff.mean()), "RMSE_difference": float((fr - kr).mean()),
                         "share_forest_better": float((diff < 0).mean()),
                         "share_kriging_better": float((diff > 0).mean())})
    return pd.DataFrame(rows).round(3)


def importance_table(d: dict) -> pd.DataFrame:
    imp = d["importance"]
    t = imp.groupby("covariate")["importance"].agg(["mean", "std", "min", "max"]).reset_index()
    return t.sort_values("mean", ascending=False).round(2)


def _hull_cells(site: str, g: dict, pts: np.ndarray) -> np.ndarray:
    ny, nx = g["elevation"].shape
    cell = float(g["cell"])
    xs = float(g["x0"]) + (np.arange(nx) + 0.5) * cell
    ys = float(g["y1"]) - (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    cells = np.c_[X.ravel(), Y.ravel()]
    return cells[Delaunay(pts).find_simplex(cells) >= 0]


def distance_table(d: dict) -> pd.DataFrame:
    rows = []
    for site in sorted(d["fields"]):
        fd = d["fields"][site]
        cv_d = []
        for r in sorted(fd.folds):
            for k in range(FO.FOLDS):
                train, test = fd.split(r, k)
                cv_d.append(cKDTree(fd.xy[train]).query(fd.xy[test])[0])
        cv_d = np.concatenate(cv_d)
        with np.load(K.CURATED / "terrain" / f"{site}.npz") as z:
            g = {k: z[k] for k in z.files}
        cells = _hull_cells(site, g, fd.xy)
        map_d = cKDTree(fd.xy).query(cells)[0]
        rows.append({"site": site, "cv_median_m": np.median(cv_d), "cv_q90_m": np.quantile(cv_d, 0.9),
                     "cells": len(cells), "map_median_m": np.median(map_d), "map_q90_m": np.quantile(map_d, 0.9)})
    return pd.DataFrame(rows).round(1)


# --- the run ------------------------------------------------------------------------------------

PARTS = ("c2", "k1", "k2", "k2r", "k4", "k5", "m1")


def warm(d: dict, cache: K.Cache, parts) -> None:
    """Only the recomputations of some rules, into the cache: two such runs
    can share the work before a full run reads it."""
    try:                    # the addendum's rules, imported here so that they stay out of CODE
        from src.checks import independent as I
    except ImportError:
        import independent as I
    steps = {"c2": lambda: recompute(d, cache), "k1": lambda: k1_algebra(d, cache),
             "k2": lambda: k2_all_stocks(d, cache), "k2r": lambda: k2_refined(d, cache),
             "k4": lambda: k4_calibration(d, cache), "k5": lambda: k5_drift(d, cache),
             "m1": lambda: m1_forests(d, cache), "k7": lambda: I.k7_maps(d, d["maps"], cache)}
    for part in parts:
        steps[part]()


def run(budget: Optional[float] = None, only=None) -> tuple[list[Rule], dict]:
    try:                    # the addendum's rules, imported here so that they stay out of CODE
        from src.checks import independent as I
    except ImportError:
        import independent as I
    d = load()
    cache = K.Cache(CACHE, None if budget is None else time.monotonic() + budget)
    if only:
        warm(d, cache, only)
        return [], {}
    rules = [c1_folds(d)]
    rec = recompute(d, cache)
    rules.append(c2_leakage(d, rec))
    k1, k1_table = k1_algebra(d, cache)
    k2, summaries = k2_all_stocks(d, cache)
    k2r, k2_table = k2_refined(d, cache)
    k4, k4_table = k4_calibration(d, cache)
    k6, k6_table = I.k6_gstools(d)
    k7, k7_table = I.k7_maps(d, d["maps"], cache)
    rules += [k1, k2, k2r, k3_predictive(d), k4, k5_drift(d, cache), k6, k7, m1_forests(d, cache),
              s1_scores(d), r1_record(d)]
    scores = scores_table(d)
    freq, coef = drift_table(d, summaries, cache)
    unc, post = uncertainty_table(d, summaries)
    tables = {
        "Per field and method (scores per repeat, averaged; coverages pooled over the repeats):": scores,
        "D12 per field and method: MSDR, the mean of (y - mean)^2 / var, and its median (1 and 0.455 for "
        "Gaussian errors with the right variance), per repeat and averaged; PICP at 80 and 95 % (repeats "
        "pooled):": I.deviation_ratios(d["cv"]).round(3),
        "D5: each field's CRPS under (a), (b) and (c), and the mean model selected:": selection_table(d),
        "(c): how often each covariate entered the drift, over the 30 folds:": freq,
        "(c): from all the stocks, the drift the AIC chooses: coefficients per standard deviation, posterior "
        "mean and 90 % credible interval:": coef,
        "Bayesian against plug-in kriging: coverage, and the median ratio of their predictive variances:": unc,
        "From all the stocks: the posterior of the effective range (m) and the nugget-to-sill ratio, beside "
        "Stage 2's estimates and profile intervals (the small fields' (b) is the shared posterior):": post,
        "D6: the transformed models beside the original-scale (a) (largest_cut: the largest share of a "
        "predictive cut off below the transform's range):": transform_table(scores, d),
        "The forests against the selected kriging, per field (differences: forest minus kriging; shares of "
        "locations, CRPS averaged over the repeats):": forest_table(d),
        "Permutation importance over the 30 forests (rise in held-out squared error, (Mg C/ha)^2):":
            importance_table(d),
        "How far the cross-validation sits from mapping: distance to the nearest location (m), from each "
        "held-out location (training locations of its field) and from each 10 m cell in the location hull:":
            distance_table(d),
        "K1 per field and mean model (largest relative error of the predictor and the variance, largest "
        "error of the drift reproduced):": k1_table.assign(**{c: k1_table[c].map(lambda v: f"{v:.1e}")
                                                                 for c in ("pred", "var", "drift")}),
        "K2's refined grid per field and mean model:": k2_table,
        "K4: QCP of the Bayesian and the plug-in predictive on the simulated fields:": k4_table.round(3),
        "K6 per field and mean model: GSTools against the plug-in at the run's grid maximum (the parameters; "
        "the largest relative error of the predictor, and of the variance against the sill and against "
        "itself):": k6_table.drop(columns="same_drift").assign(
            **{c: k6_table[c].map(lambda v: f"{v:.1e}") for c in ("pred", "var_of_sill", "var_relative")},
            eff_range=k6_table["eff_range"].round(1), sill=k6_table["sill"].round(3)),
        "K7 per field: the map against kriging.evaluate and kriging.predictive at the sampled cells (largest "
        "relative error of the mean and sd, of the quantiles against the 90 % width), and QRF's largest "
        "difference from the forest refitted:": k7_table.assign(
            **{c: k7_table[c].map(lambda v: f"{v:.1e}")
               for c in ("kriging_mean", "kriging_sd", "kriging_quantiles", "qrf")}),
    }
    return rules, tables


def report(rules: list[Rule], tables: Optional[dict], when: Optional[dt.datetime] = None) -> str:
    """The text Checkpoint 4 prints, and the RUNBOOK records."""
    when = when or dt.datetime.now(dt.timezone.utc)
    lines = [f"Checkpoint 4, run {when:%Y-%m-%dT%H:%M:%SZ}", ""]
    lines += [f"{'PASS' if r.passed else 'FAIL'}  {r.rule}: {r.detail}" for r in rules]
    if tables:
        lines += ["", "Reported, not judged:"]
        for title, table in tables.items():
            lines += ["", title, table.to_string(index=False)]
    passed = sum(r.passed for r in rules)
    lines += ["", f"Checkpoint 4 {'PASSED' if passed == len(rules) else 'FAILED'}: {passed}/{len(rules)} rules"]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """python -m src.checks.checkpoint4 [--budget SECONDS] [--only c2,k1,k2,k2r,k4,k5,m1,k7]"""
    budget, only = None, None
    args = list(argv)
    while args:
        if len(args) >= 2 and args[0] == "--budget":
            budget = float(args[1])
        elif len(args) >= 2 and args[0] == "--only" and set(args[1].split(",")) <= set(PARTS) | {"k7"}:
            only = args[1].split(",")
        else:
            print(main.__doc__)
            return 2
        args = args[2:]
    try:
        rules, tables = run(budget, only)
    except NotThere as e:
        print(f"NOT THERE: {e}")
        return 2
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget; run again to continue from the cache.")
        return 4
    if only:
        print(f"{', '.join(only)}: in the cache")
        return 0
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print(report(rules, tables))
    return 0 if all(r.passed for r in rules) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
