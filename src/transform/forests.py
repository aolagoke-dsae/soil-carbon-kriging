"""Stage 4, D7 and D8: two forests trained across the eight fields, and the baseline.

One random forest per repeat and fold, trained on the training locations of
all eight fields, fold k of every field held out together (D1), on 13
covariates: longitude, latitude and Stage 3's eleven terrain covariates,
none selected. 1,000 trees on bootstrap samples, four covariates tried at
each split, more than ten locations in every leaf, no tuning.

    qrf       the quantile regression forest: a held-out location's
              predictive distribution weights the training stocks by how
              often each shares a leaf with it, each tree's weight divided by
              the number of training locations in that leaf and averaged over
              the trees (Meinshausen 2006, eqs 4-6); its mean is the weighted
              mean
    rfrk      the forest's prediction plus, field by field, the Bayesian
              kriging of D3 (a constant mean, exponential, the field's own
              bounds) of the forest's out-of-bag residuals at that field's
              training locations: that kriging's predictive, shifted
    baseline  the training fold's own distribution of the field's stocks (D8)

The permutation importance of each covariate is the rise in the held-out
squared error of the forest's prediction when that covariate is shuffled
among the held-out locations, once per forest.

    python -m src.transform.forests [--budget SECONDS]
        writes curated/potash_cv_forests.parquet (the three methods'
        predictions), curated/potash_cv_importance.parquet, each forest's
        training locations in curated/potash_cv_forest_training.parquet, and
        R1's record, curated/potash_cv.parquet: these beside the kriging's. Every forest
        and residual kriging is cached in curated/forests_cache/, so a run
        stopped by --budget (exit 4) continues where it stopped.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

try:                        # run as a module from the repository root
    from src.transform import folds as FO, kriging as K, reml as M, scores as SC
except ImportError:         # imported with src/transform on sys.path (pytest)
    import folds as FO
    import kriging as K
    import reml as M
    import scores as SC

TREES = 1000
MAX_FEATURES = 4                    # a third of the 13 covariates
MIN_LEAF = 11                       # more than ten locations in every leaf
SEED = FO.SEED                      # the forest of repeat r and fold k: SEED + 100 r + k
COVARIATES = ("lon", "lat") + K.COVARIATES
RESIDUAL_FAMILY = "exponential"
CURATED = K.CURATED
OUT = CURATED / "potash_cv_forests.parquet"
OUT_IMPORTANCE = CURATED / "potash_cv_importance.parquet"
OUT_TRAINING = CURATED / "potash_cv_forest_training.parquet"
OUT_CV = CURATED / "potash_cv.parquet"
CACHE = CURATED / "forests_cache"
ORCHESTRATION = K.ORCHESTRATION + ("assemble", "baseline_rows", "_row", "own_bounds", "stack")
_SOURCE = K.code_hash(__file__, K.__file__, SC.__file__, M.__file__, K.A.__file__,
                      skip=ORCHESTRATION) + f"/sklearn {sklearn.__version__}"


def seed_of(repeat: int, fold: int) -> int:
    return SEED + 100 * repeat + fold


@dataclass
class Stacked:
    """The eight fields' locations in one table, sorted by field and location."""
    site: np.ndarray
    ids: np.ndarray
    X: np.ndarray                   # the 13 covariates
    y: np.ndarray
    xy: np.ndarray                  # each field's own coordinates (m), for the residual kriging
    folds: dict = dc_field(default_factory=dict)     # repeat -> fold label per location

    def split(self, repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
        f = self.folds[repeat]
        return np.flatnonzero(f != fold), np.flatnonzero(f == fold)


def stack(stocks: pd.DataFrame, terrain: pd.DataFrame, folds: pd.DataFrame) -> Stacked:
    t = stocks.merge(terrain[["site", "location_id"] + list(K.COVARIATES)], on=["site", "location_id"],
                     how="left", validate="one_to_one").sort_values(["site", "location_id"]).reset_index(drop=True)
    if t[list(COVARIATES)].isna().any().any():
        raise ValueError("a location without its covariates")
    out = Stacked(t["site"].to_numpy(), t["location_id"].to_numpy(), t[list(COVARIATES)].to_numpy(float),
                  t["stock"].to_numpy(float), t[["x", "y"]].to_numpy(float))
    f = folds.set_index(["site", "location_id", "repeat"])["fold"]
    for r in sorted(folds["repeat"].unique()):
        out.folds[int(r)] = f.loc[list(zip(out.site, out.ids, [r] * len(out.y)))].to_numpy()
    return out


def forest(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_LEAF,
                                 bootstrap=True, oob_score=True, random_state=seed, n_jobs=1)


def qrf_weights(rf: RandomForestRegressor, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Meinshausen's weights, one row per held-out location: for every tree, the
    training locations in the held-out location's leaf, each 1 / their number;
    averaged over the trees."""
    leaf_train, leaf_test = rf.apply(X_train), rf.apply(X_test)
    W = np.zeros((len(X_test), len(X_train)))
    for t in range(leaf_train.shape[1]):
        lt, ls = leaf_train[:, t], leaf_test[:, t]
        counts = np.bincount(lt, minlength=int(max(lt.max(), ls.max())) + 1)
        W += (ls[:, None] == lt[None, :]) / counts[ls][:, None]
    return W / leaf_train.shape[1]


def oob_from_trees(rf: RandomForestRegressor, X_train: np.ndarray) -> np.ndarray:
    """Each training location's mean prediction over the trees whose bootstrap
    sample left it out, from the trees' own samples (M1's second path)."""
    total, count = np.zeros(len(X_train)), np.zeros(len(X_train))
    for tree, sample in zip(rf.estimators_, rf.estimators_samples_):
        out = np.ones(len(X_train), dtype=bool)
        out[sample] = False
        total[out] += tree.predict(X_train[out])
        count[out] += 1
    return total / count


def importance(rf: RandomForestRegressor, X: np.ndarray, y: np.ndarray, seed: int) -> list[float]:
    """The rise in held-out squared error when each covariate is shuffled."""
    base = float(np.mean((y - rf.predict(X)) ** 2))
    rng = np.random.default_rng(seed)
    out = []
    for c in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, c] = rng.permutation(Xp[:, c])
        out.append(float(np.mean((y - rf.predict(Xp)) ** 2)) - base)
    return out


def weighted_var(d: SC.Weighted) -> float:
    m = d.mean()
    return float(np.dot(d.w, (d.x - m) ** 2))


def forest_result(data: Stacked, repeat: int, fold: int) -> dict:
    """One forest: QRF at the held-out locations, its predictions there, its
    out-of-bag predictions at the training locations, and the importances."""
    train, test = data.split(repeat, fold)
    if np.intersect1d(train, test).size or np.any(data.folds[repeat][train] == fold):
        raise AssertionError("a held-out location in a forest's training set")
    rf = forest(seed_of(repeat, fold)).fit(data.X[train], data.y[train])
    W = qrf_weights(rf, data.X[train], data.X[test])
    qrf = []
    for j, w in enumerate(W):
        d = SC.Weighted(data.y[train], w)
        qrf.append({"mean": d.mean(), "var": weighted_var(d), "q": list(map(float, d.quantiles())),
                    "crps": d.crps(data.y[test][j]), "weight_sum": float(w.sum()), "weight_min": float(w.min())})
    return {"train": list(map(int, train)), "test": list(map(int, test)), "qrf": qrf,
            "prediction": list(map(float, rf.predict(data.X[test]))),
            "oob": list(map(float, rf.oob_prediction_)),
            "importance": importance(rf, data.X[test], data.y[test], seed_of(repeat, fold) + 7)}


def residual_kriging(xy_train: np.ndarray, resid: np.ndarray, xy_test: np.ndarray, shift: np.ndarray,
                     y_test: np.ndarray, lo: float, hi: float) -> list[dict]:
    """RF-RK at one field's held-out locations: the Bayesian kriging (D3, a
    constant mean, exponential) of the out-of-bag residuals, shifted by the
    forest's prediction."""
    grid = K.make_grid(RESIDUAL_FAMILY, lo, hi)
    X, X0 = K.design("constant", xy_train, xy_test)
    post = K.evaluate(grid, xy_train, resid, X, xy_test, X0)
    w = post.weights()
    ok = np.isfinite(post.logpost) & (w > 0)
    out = []
    for j in range(len(xy_test)):
        mix = K.predictive(post, j)
        shifted = SC.MixtureT(mix.w, mix.loc + shift[j], mix.scale, mix.df)
        b = K.summarise(shifted, y_test[j])
        b["mean_post"] = float(shift[j] + np.dot(w[ok], post.pred[ok, j]))
        out.append(b)
    return out


def run_stage(data: Stacked, bounds: dict, cache: Optional[K.Cache] = None, log=lambda m: None,
              repeats=None, folds=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every repeat and fold: the forests' rows, the importances, and each
    forest's training locations."""
    cache = cache or K.Cache()
    rows, imp, sets = [], [], []
    for r in (repeats if repeats is not None else sorted(data.folds)):
        t0 = time.monotonic()
        for k in (folds if folds is not None else range(FO.FOLDS)):
            train, test = data.split(r, k)
            key = ["forest", r, k, seed_of(r, k), TREES, MAX_FEATURES, MIN_LEAF,
                   K.data_key(data.X[train], data.y[train], data.X[test])]
            res = cache.get(key, lambda: forest_result(data, r, k))
            sets.append(pd.DataFrame({"repeat": r, "fold": k, "site": data.site[res["train"]],
                                      "location_id": data.ids[res["train"]]}))
            f_test = np.array(res["prediction"])
            resid = data.y[train] - np.array(res["oob"])
            imp += [{"repeat": r, "fold": k, "covariate": c, "importance": v}
                    for c, v in zip(COVARIATES, res["importance"])]
            for j, i in enumerate(test):
                q = res["qrf"][j]
                rows.append(_row(data, i, r, k, "qrf", q))
            for site in sorted(set(data.site)):
                tr = train[data.site[train] == site]
                te_pos = np.flatnonzero(data.site[test] == site)
                te = test[te_pos]
                e = resid[data.site[train] == site]
                lo, hi = bounds[site]
                rk_key = ["rfrk", site, r, k, lo, hi, K.data_key(data.xy[tr], e, data.xy[te], f_test[te_pos],
                                                                  data.y[te])]
                rk = cache.get(rk_key, lambda: residual_kriging(data.xy[tr], e, data.xy[te], f_test[te_pos],
                                                                data.y[te], lo, hi))
                for i, b in zip(te, rk):
                    rows.append(_row(data, i, r, k, "rfrk", b))
        log(f"repeat {r}: {time.monotonic() - t0:.0f} s")
    return pd.DataFrame(rows), pd.DataFrame(imp), pd.concat(sets, ignore_index=True)


def _row(data: Stacked, i: int, r: int, k: int, method: str, b: dict) -> dict:
    row = {"site": data.site[i], "location_id": data.ids[i], "repeat": r, "fold": k, "method": method,
           "y": data.y[i], "mean": b["mean"], "var": b["var"], "crps": b["crps"],
           "cdf_error": b.get("cdf_error", np.nan), "mean_post": b.get("mean_post", np.nan), "outside": np.nan,
           "weight_sum": b.get("weight_sum", np.nan), "weight_min": b.get("weight_min", np.nan)}
    row.update(dict(zip(K.QCOLS, b["q"])))
    return row


def baseline_rows(data: Stacked) -> pd.DataFrame:
    """D8's baseline: the training fold's own stocks of the field, equally weighted."""
    rows = []
    for r in sorted(data.folds):
        for k in range(FO.FOLDS):
            train, test = data.split(r, k)
            for i in test:
                d = SC.Weighted(data.y[train][data.site[train] == data.site[i]])
                rows.append(_row(data, i, r, k, "baseline",
                                 {"mean": d.mean(), "var": weighted_var(d), "q": list(map(float, d.quantiles())),
                                  "crps": d.crps(data.y[i])}))
    return pd.DataFrame(rows)


def assemble(kriging: pd.DataFrame, forests: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """R1's record: every method's rows, in one order."""
    cols = ["site", "location_id", "repeat", "fold", "method", "y", "mean", "var", "crps", "cdf_error",
            "mean_post", "outside"] + K.QCOLS
    out = pd.concat([kriging[cols], forests[cols], base[cols]], ignore_index=True)
    return out.sort_values(["site", "method", "repeat", "location_id"], kind="stable").reset_index(drop=True)


def own_bounds(record: pd.DataFrame) -> dict:
    own = record[(record["fit"] == "own") & (record["mean"] == "constant")]
    return {s: (float(lo), float(hi)) for s, lo, hi in zip(own["site"], own["range_bound_lo"], own["range_bound_hi"])}


def main(argv: list[str]) -> int:
    """python -m src.transform.forests [--budget SECONDS] [--repeats 0,1,...] [--folds 0,1,...]

    exit 0: written; 2: an input is missing; 4: stopped at the budget, run
    again to continue from the cache. With --repeats or --folds, only those
    forests are computed, into the cache, and nothing is written; a run
    without it then writes the record."""
    budget, repeats, folds = None, None, None
    args = list(argv)
    while args:
        if len(args) >= 2 and args[0] == "--budget":
            budget = float(args[1])
        elif len(args) >= 2 and args[0] == "--repeats":
            repeats = [int(r) for r in args[1].split(",")]
        elif len(args) >= 2 and args[0] == "--folds":
            folds = [int(k) for k in args[1].split(",")]
        else:
            print(main.__doc__)
            return 2
        args = args[2:]
    for p in (K.STOCKS, K.TERRAIN, FO.OUT, M.OUT):
        if not p.exists():
            print(f"NOT THERE: {p}")
            return 2
    data = stack(pd.read_parquet(K.STOCKS), pd.read_parquet(K.TERRAIN), pd.read_parquet(FO.OUT))
    t0 = time.monotonic()
    cache = K.Cache(CACHE, None if budget is None else t0 + budget)
    try:
        rows, imp, sets = run_stage(data, own_bounds(pd.read_parquet(M.OUT)), cache=cache, repeats=repeats, folds=folds,
                              log=lambda m: print(f"[{time.monotonic() - t0:5.0f} s] {m}", flush=True))
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget: {cache.computed} steps computed in this run, "
              f"{cache.hits} taken from the cache. Run again to continue.")
        return 4
    if repeats is not None or folds is not None:
        print(f"repeats {repeats}, folds {folds}: done, {cache.computed} steps computed, {cache.hits} from the cache")
        return 0
    base = baseline_rows(data)
    rows.to_parquet(OUT, index=False)
    imp.to_parquet(OUT_IMPORTANCE, index=False)
    sets.to_parquet(OUT_TRAINING, index=False)
    if not K.OUT.exists():
        print(f"forests written; NOT THERE: {K.OUT}, so no {OUT_CV.name}")
        return 2
    record = assemble(pd.read_parquet(K.OUT), rows, base)
    record.to_parquet(OUT_CV, index=False)
    print(f"{len(rows)} forest predictions -> {OUT.relative_to(K.ROOT).as_posix()}; "
          f"{len(record)} rows -> {OUT_CV.relative_to(K.ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
