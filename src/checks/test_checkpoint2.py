"""Checkpoint 2's rules on hand-made tables, without raw/ or curated/."""
import numpy as np
import pandas as pd
import pytest

import checkpoint2 as C2

MEAS_COLUMNS = ["site", "location_id", "sample_depth_min", "sample_depth_max", "SOCc", "BD"]


def table(sizes=(30, 25), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for site, n in zip(("A", "B"), sizes):
        xy = rng.uniform(0.0, 300.0, size=(n, 2))
        rows += [{"site": site, "location_id": str(i), "x": xy[i, 0], "y": xy[i, 1],
                  "stock": 50.0 + rng.normal(0.0, 5.0)} for i in range(n)]
    return pd.DataFrame(rows)


def test_s1(monkeypatch):
    t = table()
    monkeypatch.setattr(C2, "N_STOCKS", len(t))
    assert C2.s1_stocks(t).passed
    assert not C2.s1_stocks(t.iloc[1:]).passed
    assert not C2.s1_stocks(pd.concat([t.iloc[1:], t.iloc[1:2]])).passed
    zero = t.copy()
    zero.loc[0, "stock"] = 0.0
    assert not C2.s1_stocks(zero).passed
    incomplete = t.copy()
    incomplete.loc[0, ["site", "location_id"]] = ["IL-PT", "1"]
    assert not C2.s1_stocks(incomplete).passed


def test_s2_against_the_loop():
    meas = pd.DataFrame([("A", "1", 0, 15, 2.0, 1.30), ("A", "1", 15, 30, 1.2, 1.45),
                         ("A", "1", 30, 60, 0.5, 1.50), ("A", "2", 0, 30, 1.5, 1.40)],
                        columns=MEAS_COLUMNS)
    t = pd.DataFrame({"site": ["A", "A"], "location_id": ["1", "2"], "stock": [65.1, 63.0]})
    assert C2.s2_arithmetic(t, meas).passed
    t.loc[0, "stock"] += 1e-6
    assert not C2.s2_arithmetic(t, meas).passed


def test_r1_passes_on_a_hand_made_table():
    assert C2.r1_likelihood(table()).passed


def test_r1_catches_a_wrong_likelihood(monkeypatch):
    monkeypatch.setattr(C2.M, "gaussian_loglik", lambda *a, **k: -1.0)
    assert not C2.r1_likelihood(table()).passed


def test_r2_reports_medians_and_convergence():
    r = C2.r2_recovery(fields=3, points=80, seed=1)
    assert r.rule == "R2 recovery"
    assert "medians over 3 fields of 80 points" in r.detail and "3/3 fits converged" in r.detail


def test_reml_direct_agrees_with_the_contrast_density():
    from scipy import linalg, stats
    t = table()
    g = t[t.site == "A"]
    xy, z = g[["x", "y"]].to_numpy(), g["stock"].to_numpy()
    V = C2.M.covariance(xy, "exponential", 25.0, 120.0, 0.3)
    X = C2._design_direct(xy, "quadratic")
    L = linalg.null_space(X.T)
    ref = stats.multivariate_normal(mean=np.zeros(L.shape[1]), cov=L.T @ V @ L).logpdf(L.T @ z)
    assert C2.reml_direct(V, X, z) == pytest.approx(ref, rel=1e-9)


def test_an_incomplete_checkpoint_does_not_pass():
    text = C2.report([C2.Rule("S1 stocks", True, "ok")], None)
    assert "NOT COMPLETE" in text and "R3 fits: not built yet" in text


def test_missing_inputs_exit_2(monkeypatch):
    def missing():
        raise C2.NotThere("the stock table")
    monkeypatch.setattr(C2, "load", missing)
    assert C2.main([]) == 2
