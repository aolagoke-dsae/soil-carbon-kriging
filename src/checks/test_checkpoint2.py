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


def test_the_verdict():
    assert "Checkpoint 2 FAILED" in C2.report([C2.Rule("S1 stocks", False, "no")], None)
    assert "Checkpoint 2 PASSED: 1/1 rules" in C2.report([C2.Rule("S1 stocks", True, "ok")], None)


def test_missing_inputs_exit_2(monkeypatch):
    def missing():
        raise C2.NotThere("the stock table")
    monkeypatch.setattr(C2, "load", missing)
    assert C2.main([]) == 2


def test_r3_and_r4_pass_on_a_stage_run(small_stage):
    t, (record, families, pooling, _) = small_stage
    assert C2.r3_fits(families, record).passed
    assert C2.r4_record(record, pooling, t, large=("A", "B"), small=("C",)).passed


def test_r3_catches_a_fit_that_did_not_converge_and_an_interval_without_its_estimate(small_stage):
    t, (record, families, pooling, _) = small_stage
    f = families.copy()
    f.loc[0, "converged"] = False
    assert not C2.r3_fits(f, record).passed
    r = record.copy()
    r.loc[0, "eff_range_lo"] = r.loc[0, "eff_range"] * 1.01
    assert not C2.r3_fits(families, r).passed
    r = record.copy()
    r.loc[0, "profile_above_fit"] = 0.01
    assert not C2.r3_fits(families, r).passed


def test_r4_catches_an_empty_cell_a_missing_row_and_a_wrong_decision(small_stage):
    t, (record, families, pooling, _) = small_stage
    args = dict(large=("A", "B"), small=("C",))
    r = record.copy()
    r.loc[0, "aic"] = np.nan
    assert not C2.r4_record(r, pooling, t, **args).passed
    r = record.copy()
    r.loc[0, "at_bound"] = ""
    assert not C2.r4_record(r, pooling, t, **args).passed
    assert not C2.r4_record(record.iloc[1:], pooling, t, **args).passed
    r = record.copy()
    small = (r["fit"] == "own") & (r["site"] == "C")
    r.loc[small, "takes"] = r.loc[small, "takes"].map({"own": "shared", "shared": "own"})
    assert not C2.r4_record(r, pooling, t, **args).passed


@pytest.mark.parametrize("tau, label", [(0.24, "strong"), (0.25, "moderate"), (0.75, "moderate"), (0.76, "weak")])
def test_cambardella_classes(tau, label):
    assert C2.cambardella(tau) == label


def test_the_reported_tables(small_stage):
    t, (record, families, pooling, boxcox) = small_stage
    tables = C2.model_diagnostics({"record": record, "families": families, "pooling": pooling,
                                   "boxcox": boxcox})
    assert len(next(iter(tables.values()))) == 6 and any(k.startswith("D3") for k in tables)


@pytest.fixture(scope="module")
def baseline(small_stage):
    import anisotropy as AN
    import variograms as VG
    t, (record, *_) = small_stage
    vario, fits = VG.run_stage(t)
    return t, vario, fits, AN.run_stage(t, record, large=("A", "B"))


def test_v1_to_v3_pass_on_hand_made_fields(baseline):
    t, vario, fits, _ = baseline
    assert C2.v1_bins(vario, t).passed
    assert C2.v2_independent(vario, t).passed
    assert C2.v3_fits(fits, vario).passed


def test_v1_v2_v3_catch_a_moved_edge_a_wrong_semivariance_and_a_worse_fit(baseline):
    t, vario, fits, _ = baseline
    moved = vario.copy()
    moved.loc[moved.index[1], "hi"] *= 1.01
    assert not C2.v1_bins(moved, t).passed
    wrong = vario.copy()
    k = wrong.index[(wrong["direction"] == "omni") & (wrong["pairs"] > 0)][0]
    wrong.loc[k, "gamma"] *= 1.001
    assert not C2.v2_independent(wrong, t).passed
    worse = fits.copy()
    worse.loc[0, "wss"] *= 1.5
    assert not C2.v3_fits(worse, vario).passed


def test_r3_covers_the_anisotropic_refits(small_stage, baseline):
    _, (record, families, *_) = small_stage
    an = baseline[3]
    assert C2.r3_fits(families, record, an).passed
    broken = an.copy()
    broken.loc[0, "loglik"] = broken.loc[0, "loglik_iso"] - 1.0
    assert not C2.r3_fits(families, record, broken).passed
