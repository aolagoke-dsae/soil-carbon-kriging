"""D8's bins, estimators and fits on hand-made data, without raw/."""
import math

import numpy as np
import pytest
from scipy.spatial.distance import pdist

import reml as M
import variograms as V


def grid_field(seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(10) * 30.0, np.arange(8) * 30.0)
    xy = np.c_[x.ravel(), y.ravel()] + rng.normal(0, 0.5, (80, 2))
    return xy, 50.0 + rng.normal(0, 5.0, 80)


def test_edges_are_centred_on_multiples_of_the_spacing_and_stop_at_half_the_largest_separation():
    xy, _ = grid_field()
    e = V.edges(xy)
    w = e[1] * 2
    assert e[0] == 0.0 and np.allclose(np.diff(e[1:]), w)
    assert e[-1] <= pdist(xy).max() / 2 < e[-1] + w


def test_matheron_and_cressie_hawkins_by_brute_force():
    xy, z = grid_field()
    e = V.edges(xy)
    d, azimuth, diff = V.pairs(xy, z)
    rows = V.semivariances(d, diff, e)
    k = 2
    sel = (d >= e[k]) & (d < e[k + 1])
    assert rows[k]["pairs"] == int(sel.sum())
    assert rows[k]["gamma"] == pytest.approx(np.mean(diff[sel] ** 2) / 2, rel=1e-12)
    skg = pytest.importorskip("skgstat")
    # scikit-gstat's estimators take the absolute differences
    assert rows[k]["gamma_ch"] == pytest.approx(skg.estimators.cressie(np.abs(diff[sel])), rel=1e-12)


def test_the_four_sectors_partition_the_pairs():
    xy, z = grid_field()
    _, azimuth, _ = V.pairs(xy, z)
    counts = sum(V.in_sector(azimuth, a).astype(int) for a in V.DIRECTIONS)
    assert (counts == 1).all()
    assert V.in_sector(np.array([0.0, 22.4, 22.5, 179.0]), 0.0).tolist() == [True, True, False, True]


def test_residuals_of_mean_model_b_are_orthogonal_to_the_trend():
    xy, z = grid_field()
    r = V.values(xy, z, "quadratic")
    assert np.abs(M.design(xy, "quadratic").T @ r).max() < 1e-8


@pytest.mark.parametrize("family", V.FAMILIES)
def test_a_fit_recovers_an_exact_model(family):
    h = np.linspace(20.0, 400.0, 15)
    truth = (5.0, 20.0, 150.0)
    gamma = truth[0] + truth[1] * V.shape(family, h, truth[2])
    f = V.fit(family, h, gamma, np.full(15, 100))
    assert f["converged"] and f["wss"] < 1e-12
    assert (f["nugget"], f["partial_sill"], f["a"]) == pytest.approx(truth, rel=1e-5)
    assert f["eff_range"] == pytest.approx(150.0 if family == "spherical" else 150.0 * math.log(20), rel=1e-5)


def test_a_rising_variogram_puts_the_range_at_its_bound():
    h = np.linspace(20.0, 400.0, 15)
    f = V.fit("spherical", h, 2.0 + 0.05 * h, np.full(15, 100))
    assert f["a"] == pytest.approx(f["a_max"]) and "a" in f["at_bound"]


def test_a_field_gives_omnidirectional_and_directional_rows_and_two_fits():
    xy, z = grid_field()
    rows, fits = V.field_variograms("F", "constant", xy, z)
    directions = {r["direction"] for r in rows}
    assert directions == {"omni", "0", "45", "90", "135"}
    omni = [r for r in rows if r["direction"] == "omni"]
    for k, r in enumerate(omni):
        assert sum(x["pairs"] for x in rows if x["direction"] != "omni" and x["bin"] == k) == r["pairs"]
    assert all(r["fitted"] == (r["pairs"] >= V.MIN_PAIRS) for r in omni)
    assert sorted(f["family"] for f in fits) == sorted(V.FAMILIES) and sum(f["chosen"] for f in fits) == 1
