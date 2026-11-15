"""Stage 6, D4-D7: the models of BZE-LW's stocks, cross-validated, and the
cost of a second displacement (RUNBOOK, Stage 6; rules fixed 15.11.2026).

    python -m src.transform.bze_models [--budget SECONDS] [--part cv|d7]
        writes curated/bze_cv.parquet (every model's predictions) and
        curated/bze_d7.parquet (D7's); every forest and residual kriging is
        cached in curated/bze_cache/, so a run stopped by --budget (exit 4)
        continues where it stopped

Populations (D1): mineral (bze_moor 0) and organic (1), the sites C2 keeps.
Feature sets: "point" and "disc" (D3's readings of the seven covariates, the
published coordinates and the site's land use, one-hot), and "desc" (D4 (d):
the site's own description as found in the field, each attribute coded by
its categories in sorted order, and its land use). Models (D4): qrf on each
set; rfrk, the qrf's forest plus the Bayesian kriging of its out-of-bag
residuals, on "point" and "disc"; baseline, the training stocks of the
site's land use.
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

try:                        # run as a module from the repository root
    from src.transform import bze_covariates as BC, forests as F, kriging as K, reml as M, scores as SC
except ImportError:         # imported with src/transform on sys.path (pytest)
    import bze_covariates as BC
    import forests as F
    import kriging as K
    import reml as M
    import scores as SC

OUT = BC.CURATED / "bze_cv.parquet"
OUT_D7 = BC.CURATED / "bze_d7.parquet"
CACHE = BC.CURATED / "bze_cache"
REPEATS, FOLDS, SEED = 3, 10, 20261116
LAND_USES = ("A", "G", "SO")
DESCRIPTION = ("hauptbodentyp", "grundwastufe", "neigung", "exposition", "woelbung", "reliefformtyp",
               "lageimrelief", "bodenklimaraum_name")
POPULATIONS = {"mineral": 0, "organic": 1}
D7_REPLICATES, D7_SEED = 5, 20261117
ORCHESTRATION = ("main", "run_cv", "run_d7", "load")
_CODE = K.code_hash(__file__, F.__file__, K.__file__, SC.__file__, skip=ORCHESTRATION)


def load() -> pd.DataFrame:
    """The site layer with D3's covariates, the sites C2 keeps."""
    try:
        from src.checks import checkpoint1 as C1
    except ImportError:
        import checkpoint1 as C1
    t = C1.bze_table(C1.bze_pages())
    cov = pd.read_parquet(BC.OUT)
    t = t.merge(cov, on="pointid", validate="one_to_one")
    t = t[t["kept"]].copy()
    t["exposition"] = t["exposition"].fillna("flat")
    for c in ("xcoord", "ycoord", "kv_0_30"):
        t[c] = t[c].astype(float)
    return t.sort_values("pointid", kind="stable").reset_index(drop=True)


def features(t: pd.DataFrame, which: str) -> tuple[np.ndarray, list[str]]:
    """The feature matrix of a set, and its column names."""
    lu = np.column_stack([(t["landnutzung"] == u).to_numpy(float) for u in LAND_USES])
    if which in ("point", "disc"):
        cols = [f"{n}_{which}" for n in BC.NAMES]
        X = np.column_stack([t[cols].to_numpy(float), t[["xcoord", "ycoord"]].to_numpy(float), lu])
        return X, cols + ["xcoord", "ycoord"] + [f"lu_{u}" for u in LAND_USES]
    if which == "desc":
        codes = [pd.Categorical(t[c].astype(str), categories=sorted(t[c].astype(str).unique())).codes for c in DESCRIPTION]
        return np.column_stack(codes + [lu]).astype(float), list(DESCRIPTION) + [f"lu_{u}" for u in LAND_USES]
    raise ValueError(which)


def folds(n: int, repeat: int) -> np.ndarray:
    """D5: ten folds at random, sizes differing by at most one."""
    rng = np.random.default_rng(SEED + repeat)
    out = np.empty(n, dtype=int)
    out[rng.permutation(n)] = np.arange(n) % FOLDS
    return out


def max_features(p: int) -> int:
    return max(1, p // 3)                    # a third of the covariates (Stage 4, D7)


def forest_fold(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int) -> dict:
    """QRF at the held-out sites, the forest's predictions there and its
    out-of-bag predictions at the training sites."""
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=F.TREES, max_features=max_features(X.shape[1]),
                               min_samples_leaf=F.MIN_LEAF, bootstrap=True, oob_score=True, random_state=seed,
                               n_jobs=-1).fit(X[train], y[train])
    W = F.qrf_weights(rf, X[train], X[test])
    out = []
    for j, w in enumerate(W):
        d = SC.Weighted(y[train], w)
        out.append({"mean": d.mean(), "var": F.weighted_var(d), "q": list(map(float, d.quantiles())),
                    "crps": d.crps(y[test][j]), "weight_sum": float(w.sum()), "weight_min": float(w.min())})
    return {"qrf": out, "prediction": list(map(float, rf.predict(X[test]))), "oob": list(map(float, rf.oob_prediction_))}


def residual_bounds(xy: np.ndarray) -> tuple[float, float]:
    """D4 (b): Stage 2's rule for the effective range's bounds, on the residuals' own sites."""
    return M.Field.of(xy, np.zeros(len(xy))).range_bounds()


def run_cv(t: pd.DataFrame, cache: K.Cache, log=lambda m: None, populations=None, sets=("point", "disc", "desc")):
    rows = []
    for pop in (populations or POPULATIONS):
        p = t[t["bze_moor"].astype(int) == POPULATIONS[pop]].reset_index(drop=True)
        y = p["kv_0_30"].to_numpy(float)
        xy = p[["xcoord", "ycoord"]].to_numpy(float)
        for r in range(REPEATS):
            fl = folds(len(p), r)
            for k in range(FOLDS):
                train, test = np.flatnonzero(fl != k), np.flatnonzero(fl == k)
                for j, i in enumerate(test):            # baseline: the training stocks of the site's land use
                    same = train[p["landnutzung"].to_numpy()[train] == p["landnutzung"].iloc[i]]
                    d = SC.Weighted(y[same if len(same) else train])
                    rows.append(_row(p, i, pop, "none", "baseline", r, k, y, {"mean": d.mean(), "var": F.weighted_var(d),
                                     "q": list(map(float, d.quantiles())), "crps": d.crps(y[i])}))
                for s in sets:
                    t0 = time.monotonic()
                    X, _ = features(p, s)
                    seed = SEED + 1000 * POPULATIONS[pop] + 100 * r + k
                    key = ["bze-forest", _CODE, pop, s, r, k, seed, K.data_key(X, y, train)]
                    res = cache.get(key, lambda: forest_fold(X, y, train, test, seed))
                    for i, b in zip(test, res["qrf"]):
                        rows.append(_row(p, i, pop, s, "qrf", r, k, y, b))
                    if s == "desc":
                        continue
                    resid = y[train] - np.array(res["oob"])
                    lo, hi = residual_bounds(xy[train])
                    rk_key = ["bze-rfrk", _CODE, pop, s, r, k, lo, hi, K.data_key(xy[train], resid, xy[test])]
                    rk = cache.get(rk_key, lambda: F.residual_kriging(xy[train], resid, xy[test],
                                                                       np.array(res["prediction"]), y[test], lo, hi))
                    for i, b in zip(test, rk):
                        rows.append(_row(p, i, pop, s, "rfrk", r, k, y, b))
                    log(f"{pop} {s} r{r} k{k}: {time.monotonic() - t0:.0f} s")
    return pd.DataFrame(rows)


def _row(p: pd.DataFrame, i: int, pop: str, s: str, method: str, r: int, k: int, y: np.ndarray, b: dict) -> dict:
    row = {"population": pop, "features": s, "method": method, "pointid": int(p["pointid"].iloc[i]),
           "landnutzung": p["landnutzung"].iloc[i], "repeat": r, "fold": k, "y": float(y[i]),
           "mean": b["mean"], "var": b["var"], "crps": b["crps"]}
    row.update(dict(zip(K.QCOLS, b["q"])))
    return row


# --- D7: a second displacement -------------------------------------------------------------------

def displace(xy: np.ndarray, replicate: int, radius: float = BC.RADIUS) -> np.ndarray:
    """Every point moved again, uniformly over the disc of `radius` around it."""
    rng = np.random.default_rng(D7_SEED + replicate)
    rho = radius * np.sqrt(rng.uniform(size=len(xy)))
    theta = rng.uniform(0, 2 * math.pi, size=len(xy))
    return xy + np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def displaced_covariates(t: pd.DataFrame, xy: np.ndarray) -> pd.DataFrame:
    s = pd.DataFrame({"pointid": t["pointid"].to_numpy(), "xcoord": xy[:, 0], "ycoord": xy[:, 1]})
    return BC.covariates(s)


def run_d7(t: pd.DataFrame, cache: K.Cache, log=lambda m: None) -> pd.DataFrame:
    rows = []
    p = t[t["bze_moor"].astype(int) == 0].reset_index(drop=True)
    y = p["kv_0_30"].to_numpy(float)
    for rep in range(D7_REPLICATES):
        xy2 = displace(p[["xcoord", "ycoord"]].to_numpy(float), rep)
        cov2 = cache.get(["bze-d7-cov", _CODE, rep, K.data_key(xy2)],
                         lambda: displaced_covariates(p, xy2).to_dict(orient="list"))
        q = p.copy()
        for c, v in cov2.items():
            if c != "pointid":
                q[c] = v
        q["xcoord"], q["ycoord"] = xy2[:, 0], xy2[:, 1]
        usable = np.isfinite(q[[c for c in q if c.endswith(("_point", "_disc"))]].to_numpy(float)).all(axis=1)
        for s in ("point", "disc"):
            X, _ = features(q, s)
            for r in range(REPEATS):
                fl = folds(len(p), r)
                for k in range(FOLDS):
                    t0 = time.monotonic()
                    train, test = np.flatnonzero((fl != k) & usable), np.flatnonzero((fl == k) & usable)
                    seed = SEED + 100 * r + k
                    key = ["bze-d7-forest", _CODE, rep, s, r, k, seed, K.data_key(X, y, train)]
                    res = cache.get(key, lambda: forest_fold(X, y, train, test, seed))
                    for i, b in zip(test, res["qrf"]):
                        row = _row(p, i, "mineral", s, "qrf", r, k, y, b)
                        row["replicate"] = rep
                        rows.append(row)
                    log(f"d7 rep{rep} {s} r{r} k{k}: {time.monotonic() - t0:.0f} s")
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    budget, part = None, "cv"
    args = list(argv)
    while args:
        if len(args) >= 2 and args[0] == "--budget":
            budget = float(args[1])
        elif len(args) >= 2 and args[0] == "--part" and args[1] in ("cv", "d7"):
            part = args[1]
        else:
            print(__doc__)
            return 2
        args = args[2:]
    t = load()
    cache = K.Cache(CACHE, None if budget is None else time.monotonic() + budget)
    try:
        if part == "cv":
            out = run_cv(t, cache, log=print)
            out.to_parquet(OUT, index=False)
        else:
            out = run_d7(t, cache, log=print)
            out.to_parquet(OUT_D7, index=False)
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget; run again to continue from the cache.")
        return 4
    print(f"{len(out)} rows -> {(OUT if part == 'cv' else OUT_D7).relative_to(BC.ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
