"""The addendum's rules on synthetic fields: GSTools against this
repository's kriging (K6), the 40-digit reference where the two disagree,
K6 on a toy run, the hull's cells two ways (K7) and D12's ratios by hand."""
import math

import mpmath as mp
import numpy as np
import pandas as pd
import pytest

import independent as I
import kriging as K
import maps as MP
import reml as M
import scores as SC


@pytest.mark.parametrize("family, nu", [("exponential", None), ("spherical", None), ("matern", 0.2),
                                        ("matern", 0.7), ("matern", 3.0)])
def test_gstools_correlation_is_this_repositorys(family, nu):
    a = 340.0
    h = np.r_[0.0, np.geomspace(0.5, 3 * a, 60), a]
    gm = I.gstools_model(family, 1.0, 0.0, a, nu, unit=1.0)
    assert np.max(np.abs(gm.correlation(h) - M.correlation(family, h, a, nu))) < 1e-12
    assert M.correlation(family, np.array([a]), a, nu)[0] == pytest.approx(0.0 if family == "spherical" else 0.05,
                                                                           abs=1e-12)


def ours(mean, family, xy, z, xy0, cov, cov0, a, tau, nu, angle, ratio, drift):
    X, X0 = K.design(mean, xy, xy0, cov, cov0, drift)
    post = K.evaluate(K.Grid(family, (math.log(a),), (tau,), (nu,)), xy, z, X, xy0, X0, angle, ratio)
    sill = float(post.s2[0] / post.df)
    return post.pred[0], sill * post.var1[0], sill


def test_gstools_agrees_on_simulated_fields_in_projected_coordinates():
    rng = np.random.default_rng(17)
    worst_p = worst_v = 0.0
    for trial in range(12):
        n, m = int(rng.integers(40, 120)), 8
        origin = np.array([3.5e5, 4.4e6]) + rng.uniform(0, 1e5, 2)
        side = rng.uniform(250, 1200)
        xy = origin + rng.uniform(0, 1, (n, 2)) * [side, 0.8 * side]
        xy0 = origin + rng.uniform(0, 1, (m, 2)) * [side, 0.8 * side]
        scale = np.array([3.0, 0.4, 2e-3, 2.0])          # elevation, slope, curvature, TWI, roughly
        cov, cov0 = (np.r_[220, 1, 0, 8] + rng.normal(0, 1, (k, 4)) * scale for k in (n, m))
        z = 55 + rng.normal(0, 6, n)
        family = ("exponential", "spherical", "matern")[trial % 3]
        nu = (0.2, 0.7, 1.5, 3.0)[trial % 4] if family == "matern" else None
        a = float(np.exp(rng.uniform(np.log(30), np.log(3000))))
        tau = (1e-6, 0.05, 0.3, 0.7, 1.0)[trial % 5]
        angle, ratio = (float(rng.uniform(0, 180)), float(rng.uniform(0.3, 0.9))) if trial % 4 == 1 else (None, None)
        for mean, drift in (("constant", ()), ("quadratic", ()), ("drift", (0, 2, 3))):
            if mean == "drift" and angle is not None:
                continue                                   # (c) is isotropic (D4)
            p, v, sill = ours(mean, family, xy, z, xy0, cov, cov0, a, tau, nu, angle, ratio, drift)
            gm = I.gstools_model(family, sill, tau, a, nu, angle, ratio)
            gp, gv = I.gstools_krige(mean, gm, xy, z, xy0, cov[:, list(drift)], cov0[:, list(drift)])
            worst_p = max(worst_p, float(np.max(np.abs(gp - p) / np.abs(p))))
            worst_v = max(worst_v, float(np.max(np.abs(gv - v)) / sill))
    assert worst_p < I.TOL_K6_PRED and worst_v < I.TOL_K6_VAR


def test_where_the_system_is_ill_conditioned_this_repository_keeps_40_digit_accuracy():
    """A smooth Matern (nu = 3) with almost no nugget, at a range longer than
    the field: the kriging variance is a small difference of numbers the size
    of the sill. This repository's stays accurate against 40-digit arithmetic;
    GSTools', formed by subtraction from an explicit inverse, stays within K6's
    tolerance of the sill but not of itself."""
    rng = np.random.default_rng(11)
    n, m, nu, a, tau = 40, 3, 3.0, 1446.0, 1e-6
    xy, xy0 = rng.uniform(0, 1000, (n, 2)), rng.uniform(0, 1000, (m, 2))
    z = 55 + rng.normal(0, 6, n)
    X, X0 = K.design("constant", xy, xy0)
    post = K.evaluate(K.Grid("matern", (math.log(a),), (tau,), (nu,)), xy, z, X, xy0, X0)
    lam = np.linalg.eigvalsh((1 - tau) * M.correlation("matern", K.distances(xy, xy), a, nu) + tau * np.eye(n))
    assert lam.max() / lam.min() > 1e6
    gm = I.gstools_model("matern", 1.0, tau, a, nu)
    _, gv = I.gstools_krige("constant", gm, xy, z, xy0)
    mp.mp.dps = 40
    cor = lambda h, phi: mp.mpf(1) if h == 0 else 2 ** (1 - nu) / mp.gamma(nu) * (h / phi) ** nu * mp.besselk(nu, h / phi)
    phi = mp.findroot(lambda f: cor(mp.mpf(a), f) - mp.mpf("0.05"), mp.mpf(a) / 6)
    P = [tuple(map(mp.mpf, map(float, p))) for p in xy]
    dist = lambda p, q: mp.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
    C = mp.matrix(n + 1, n + 1)
    for i in range(n):
        for j in range(i, n):
            C[i, j] = C[j, i] = (1 - mp.mpf(tau)) * cor(dist(P[i], P[j]), phi) + (mp.mpf(tau) if i == j else 0)
        C[i, n] = C[n, i] = 1
    for t in range(m):
        q = tuple(map(mp.mpf, map(float, xy0[t])))
        b = mp.matrix([(1 - mp.mpf(tau)) * cor(dist(q, P[i]), phi) for i in range(n)] + [1])
        w = mp.lu_solve(C, b)
        exact = float(1 - sum(w[i] * b[i] for i in range(n)) - w[n])
        assert abs(post.var1[0, t] - exact) < 1e-8 * exact
        assert abs(gv[t] - exact) < I.TOL_K6_VAR


def toy_run(tmp_path):
    """Two small fields, (a), (b) and (c) on narrow grids, repeat 0's folds,
    and the record's plug-in rows from the run's own fold results."""
    rng = np.random.default_rng(23)
    fields, models, rows, drifts = {}, {}, [], []
    cache = K.Cache(tmp_path)
    for site, n in (("F1", 46), ("F2", 52)):
        xy = np.c_[5.1e5 + rng.uniform(0, 280, n), 4.43e6 + rng.uniform(0, 200, n)]
        cov = rng.normal(0, 1, (n, len(K.COVARIATES))) * np.linspace(0.01, 50, len(K.COVARIATES))
        z = 60 + 0.01 * (xy[:, 0] - 5.1e5) + 0.05 * cov[:, 9] + rng.normal(0, 4, n)
        fd = K.FieldData(site, np.array([f"{site}-{i:03d}" for i in range(n)]), xy, z, cov)
        fd.folds[0] = rng.permutation(np.arange(n) % 10)
        fields[site] = fd
        models[site] = [K.Model("a", "constant", "exponential", 60.0, 150.0),
                        K.Model("b", "quadratic", "spherical", 80.0, 200.0),
                        K.Model("c", "drift", "exponential", 60.0, 120.0)]
    d = {"fields": fields, "models": models}
    for site in fields:
        for model in models[site]:
            res = I.run_result(d, site, model.name, cache)
            rows += K._rows(site, model, 0, 0, res)
            if "drift" in res:
                drifts.append({"site": site, "repeat": 0, "fold": 0, "candidate": res["drift"]["chosen"],
                               "chosen": True})
    d.update(cv=pd.DataFrame(rows), drifts=pd.DataFrame(drifts))
    return d, cache


def test_k6_passes_the_run_and_fails_a_moved_plug_in(tmp_path):
    d, cache = toy_run(tmp_path)
    rule, table = I.k6_gstools(d, cache)
    assert rule.passed, rule.detail
    assert len(table) == 6 and table["pred"].max() < 1e-12
    cv = d["cv"].copy()
    i = cv.index[cv["method"] == "plugin_b"][0]
    cv.loc[i, "mean"] *= 1 + 1e-7
    assert not I.k6_gstools(dict(d, cv=cv), cache)[0].passed
    dr = d["drifts"].copy()
    dr.loc[0, "candidate"] = (dr.loc[0, "candidate"] + 1) % 256
    assert not I.k6_gstools(dict(d, drifts=dr), cache)[0].passed


def test_k7s_hull_is_the_maps_hull():
    rng = np.random.default_rng(4)
    ny, nx, cell = 50, 60, 10.0
    g = {"elevation": np.zeros((ny, nx)), "cell": cell, "x0": 6.1e5, "y1": 4.41e6, "epsg": 32616}
    for i, c in enumerate(K.COVARIATES):
        g[c] = rng.normal(0, 1, (ny, nx))
    g[K.COVARIATES[5]][rng.integers(0, ny, 40), rng.integers(0, nx, 40)] = np.nan
    xy = np.c_[6.1e5 + rng.uniform(40, 560, 35), 4.41e6 - rng.uniform(40, 460, 35)]
    rows, cols, _, _ = MP.hull_cells(xy, g)
    assert I.map_cells(g, xy) == set(zip(rows.tolist(), cols.tolist()))


def test_deviation_ratios_by_hand():
    q = SC.Normal(0.0, 1.0).quantiles()
    base = {"site": "F1", "method": "kriging_a", **{c: v for c, v in zip(K.QCOLS, 10.0 + 2.0 * q)}}
    ys = [(0, 10.0 + 2.0, 4.0), (0, 10.0 - 4.0, 4.0), (0, 10.0, 4.0), (1, 10.0 + 6.0, 4.0), (1, 10.0, 4.0)]
    cv = pd.DataFrame([dict(base, repeat=r, y=y, mean=10.0, var=v) for r, y, v in ys])
    t = I.deviation_ratios(cv)
    # repeat 0: ratios 1, 4, 0 (mean 5/3, median 1); repeat 1: 9, 0 (mean 4.5, median 4.5)
    assert t["MSDR"].iloc[0] == pytest.approx((5 / 3 + 4.5) / 2)
    assert t["MSDR_median"].iloc[0] == pytest.approx((1.0 + 4.5) / 2)
    # the 80 % interval is 10 -+ 2.563: inside 10 + 2, 10, 10; the 95 % one, 10 -+ 3.920: and 10 - 4 is outside
    assert t["PICP80"].iloc[0] == pytest.approx(3 / 5) and t["PICP95"].iloc[0] == pytest.approx(3 / 5)
