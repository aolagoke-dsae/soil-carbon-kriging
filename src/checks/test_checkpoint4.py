"""Checkpoint 4's rules on small synthetic tables: each passes what it should and fails what it should."""
import numpy as np
import pandas as pd
import pytest

import checkpoint4 as C4
import folds as FO
import kriging as K
import scores as SC


def toy_stocks(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for site, n in (("F1", 23), ("F2", 31)):
        for i in range(n):
            rows.append({"site": site, "location_id": f"{site}-{i:03d}", "x": rng.uniform(0, 300),
                         "y": rng.uniform(0, 200), "stock": rng.normal(60, 6)})
    return pd.DataFrame(rows)


def toy_cv(stocks, folds, cdf_error=1e-13):
    cv = folds.merge(stocks[["site", "location_id", "stock"]], on=["site", "location_id"])
    cv = cv.rename(columns={"stock": "y"}).assign(method="kriging_a", mean=60.0, var=36.0, crps=3.0,
                                                  cdf_error=cdf_error, mean_post=60.0, outside=np.nan)
    for c, lev in zip(K.QCOLS, SC.LEVELS):
        cv[c] = 60.0 + 6.0 * float(SC.Normal(0, 1).quantiles([lev])[0])
    return cv


def test_c1_passes_the_assignment_and_fails_a_moved_prediction():
    stocks = toy_stocks()
    folds = FO.assign(stocks)
    d = {"stocks": stocks, "folds": folds, "cv": toy_cv(stocks, folds)}
    assert C4.c1_folds(d).passed
    bad = d["cv"].copy()
    bad.loc[0, "fold"] = (bad.loc[0, "fold"] + 1) % 10
    assert not C4.c1_folds(dict(d, cv=bad)).passed
    same = folds.copy()
    same.loc[same["repeat"] == 1, "fold"] = same.loc[same["repeat"] == 0, "fold"].to_numpy()
    r = C4.c1_folds(dict(d, folds=same, cv=toy_cv(stocks, same)))
    assert not r.passed and "share an assignment" in r.detail


def test_k3_fails_a_distribution_function_off_its_levels():
    stocks = toy_stocks()
    folds = FO.assign(stocks)
    assert C4.k3_predictive({"cv": toy_cv(stocks, folds)}).passed
    assert not C4.k3_predictive({"cv": toy_cv(stocks, folds, cdf_error=1e-6)}).passed
    cv = toy_cv(stocks, folds)
    cv.loc[3, "mean_post"] = 60.001
    assert not C4.k3_predictive({"cv": cv}).passed


def test_s1_passes():
    r = C4.s1_scores()
    assert r.passed, r.detail


def test_k5_finds_the_lowest_aic_and_fails_another_choice():
    stocks = toy_stocks()
    folds = FO.assign(stocks)
    rng = np.random.default_rng(4)
    terrain = stocks[["site", "location_id"]].copy()
    for c in K.COVARIATES:
        terrain[c] = rng.normal(size=len(terrain))
    fields = K.load_fields(stocks, terrain, folds)
    grid = K.Grid("exponential", tuple(np.log([30.0, 120.0])), (1e-6, 0.5), (None,))
    rows = []
    for site, fd in fields.items():
        for k in range(FO.FOLDS):
            train = fd.split(0, k)[0]
            t = K.ml_table(grid, fd.xy[train], fd.z[train], fd.cov[train])
            chosen = K.choose(t)
            rows.append(t.assign(site=site, repeat=0, fold=k, chosen=t["candidate"] == chosen))
    drifts = pd.concat(rows, ignore_index=True)
    d = {"drifts": drifts, "fields": fields}
    r = C4.k5_drift(d, K.Cache())
    assert r.passed, r.detail
    bad = drifts.copy()
    first = (bad["site"] == "F1") & (bad["fold"] == 2)
    bad.loc[first, "chosen"] = bad.loc[first, "candidate"] == (int(bad.loc[first & bad["chosen"], "candidate"].iloc[0]) + 1) % 256
    assert not C4.k5_drift(dict(d, drifts=bad), K.Cache()).passed


def test_bordered_system_by_hand():
    C = np.array([[1.0, 0.3], [0.3, 1.0]])
    X = np.ones((2, 1))
    pred, var, lam = C4.bordered(C, X, np.array([0.5, 0.2]), np.array([1.0]), np.array([10.0, 20.0]))
    # weights 0.5 +- (0.5 - 0.2) / (2 (1 - 0.3)): 0.7143 and 0.2857
    assert lam == pytest.approx([0.5 + 0.3 / 1.4, 0.5 - 0.3 / 1.4])
    assert pred == pytest.approx(10 * lam[0] + 20 * lam[1])
    assert X.T @ lam == pytest.approx([1.0])
