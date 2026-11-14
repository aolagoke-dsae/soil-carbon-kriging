"""Stage 5's block kriging, its pairs' histogram and its planning grid, on synthetic fields."""
import math

import numpy as np
import pytest

import field_means as FM
import kriging as K
import reml as M


def toy(n=45, m=400, seed=9):
    rng = np.random.default_rng(seed)
    xy = np.c_[5.0e5 + rng.uniform(0, 300, n), 4.4e6 + rng.uniform(0, 240, n)]
    cells = np.c_[5.0e5 + rng.uniform(20, 280, m), 4.4e6 + rng.uniform(20, 220, m)]
    cov = rng.normal(0, 1, (n, len(K.COVARIATES)))
    cov_c = rng.normal(0, 1, (m, len(K.COVARIATES)))
    z = 60 + 0.02 * (xy[:, 0] - 5.0e5) + 2.0 * cov[:, 2] + rng.normal(0, 4, n)
    return K.FieldData("T", np.array([str(i) for i in range(n)]), xy, z, cov), cells, cov_c


@pytest.mark.parametrize("angle, ratio", [(None, None), (35.0, 0.5)])
def test_transformed_distances_are_stage_2s_and_the_histogram_is_exact_enough(angle, ratio):
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 300, (250, 2))
    t = FM.transformed(xy, angle, ratio)
    d = np.hypot(*(t[:, None, :] - t[None, :, :]).transpose(2, 0, 1))
    assert np.max(np.abs(d - K.distances(xy, xy, angle, ratio))) < 1e-9
    hist = FM.pair_histogram(xy, angle, ratio)
    for family, nu in (("exponential", None), ("spherical", None), ("matern", 0.2), ("matern", 3.0)):
        exact = FM.exact_mean_correlation(xy, family, 120.0, nu, angle, ratio)
        assert FM.mean_correlation(hist, len(xy), family, 120.0, nu) == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("mean, family, angle, ratio, drift", [
    ("constant", "exponential", None, None, ()), ("quadratic", "spherical", None, None, ()),
    ("drift", "exponential", None, None, (2, 5)), ("constant", "matern", 40.0, 0.6, ())])
def test_block_kriging_equals_the_bordered_system(mean, family, angle, ratio, drift):
    fd, cells, cov_c = toy()
    model = K.Model("x", mean, family, 60.0, 160.0, angle, ratio)
    grid = model.grid()
    a_arr, t_arr, nu_arr = grid.arrays()
    lp = np.zeros(grid.size)
    keep = np.zeros(grid.size, dtype=bool)
    g = int(np.argmin(np.abs(a_arr - 100.0) + np.abs(t_arr - 0.3)))
    keep[g] = True
    hist = FM.pair_histogram(cells, angle, ratio)
    res = FM.block_grid(fd, model, cells, cov_c, drift, lp, hist, keep)
    a, tau = a_arr[g], t_arr[g]
    nu = None if np.isnan(nu_arr[g]) else nu_arr[g]
    X, X0 = K.design(mean, fd.xy, cells, fd.cov, cov_c, drift)
    n, p = X.shape
    C = (1 - tau) * M.correlation(family, K.distances(fd.xy, fd.xy, angle, ratio), a, nu) + tau * np.eye(n)
    kb = (1 - tau) * M.correlation(family, K.distances(cells, fd.xy, angle, ratio), a, nu).mean(axis=0)
    A = np.block([[C, X], [X.T, np.zeros((p, p))]])
    sol = np.linalg.solve(A, np.r_[kb, X0.mean(axis=0)])
    lam = sol[:n]
    cbb = (1 - tau) * FM.exact_mean_correlation(cells, family, a, nu, angle, ratio)
    assert res["pred"][0] == pytest.approx(lam @ fd.z, rel=1e-9)
    assert res["v"][0] == pytest.approx(cbb - lam @ kb - sol[n:] @ X0.mean(axis=0), rel=1e-6)


def test_summary_by_hand_and_the_planning_grid():
    fd, cells, cov_c = toy()
    model = K.Model("x", "constant", "exponential", 60.0, 160.0)
    lp = K.evaluate(model.grid(), fd.xy, fd.z, K.design("constant", fd.xy, fd.xy)[0]).logpost
    res = FM.block_grid(fd, model, cells, cov_c, (), lp, FM.pair_histogram(cells))
    s = FM.summary(res, lp, fd)
    assert s["sample_mean"] == pytest.approx(fd.z.mean())
    assert s["sample_se"] == pytest.approx(fd.z.std(ddof=1) / math.sqrt(len(fd.z)))
    assert s["q025"] < s["q50"] < s["q975"] and s["sd"] > 0 and s["d2"] > 0
    pts = FM.planning_grid(cells, fd.xy, 36, 10.0)
    assert 15 < len(pts) < 60
    thetas, we = FM.heavy_points(model, res, lp)
    assert we.sum() > 0 and len(thetas) == len(we)
    v_bin = FM.effort_variance(model, pts, cov_c[:len(pts)], cells, cov_c, (), thetas[:3], FM.pair_histogram(cells))
    v_ex = FM.effort_variance(model, pts, cov_c[:len(pts)], cells, cov_c, (), thetas[:3], None, exact=True)
    assert np.allclose(v_bin, v_ex, rtol=1e-5)
