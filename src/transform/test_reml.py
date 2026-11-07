"""The REML code against closed forms, scipy and scikit-learn, without raw/."""
import math

import numpy as np
import pytest
from scipy import linalg, stats

import reml as M


def field(n=60, mean="constant", seed=1):
    rng = np.random.default_rng(seed)
    xy, z = M.simulate(rng, n, (300.0, 200.0), "exponential", 50.0, 80.0, 0.3, mean=60.0)
    return M.Field.of(xy, z, mean)


@pytest.mark.parametrize("family, nu", [("exponential", None), ("spherical", None),
                                        ("matern", 0.3), ("matern", 1.5), ("matern", 2.8)])
def test_one_at_zero_and_005_at_the_effective_range(family, nu):
    r = M.correlation(family, np.array([0.0, 120.0]), 120.0, nu)
    assert r[0] == 1.0
    assert r[1] == pytest.approx(0.0 if family == "spherical" else 0.05, abs=1e-10)


def test_spherical_values():
    assert M.correlation("spherical", np.array([50.0, 100.0, 150.0]), 100.0) == pytest.approx([0.3125, 0.0, 0.0])


def test_matern_one_half_is_the_exponential_and_three_halves_its_closed_form():
    h, a = np.linspace(0.0, 400.0, 41), 150.0
    assert M.correlation("matern", h, a, 0.5) == pytest.approx(M.correlation("exponential", h, a), rel=1e-12, abs=1e-15)
    u = h * M.scale("matern", 1.5) / a
    assert M.correlation("matern", h, a, 1.5) == pytest.approx((1 + u) * np.exp(-u), rel=1e-10, abs=1e-15)


def test_the_profile_is_the_maximum_over_the_sill():
    f = field()
    ll, s2, _ = M.reml_profile(f, "exponential", 90.0, 0.25)
    assert M.reml_loglik(f, "exponential", s2, 90.0, 0.25) == pytest.approx(ll, rel=1e-12)
    assert all(M.reml_loglik(f, "exponential", s2 * k, 90.0, 0.25) < ll for k in (0.9, 1.1))


@pytest.mark.parametrize("mean", M.MEANS)
def test_reml_is_the_density_of_orthonormal_error_contrasts(mean):
    f = field(mean=mean)
    V = M.covariance(f.xy, "matern", 40.0, 100.0, 0.2, 1.2)
    L = linalg.null_space(f.X.T)                      # L'X = 0, L'L = I
    ref = stats.multivariate_normal(mean=np.zeros(L.shape[1]), cov=L.T @ V @ L).logpdf(L.T @ f.z)
    assert M.reml_loglik(f, "matern", 40.0, 100.0, 0.2, 1.2) == pytest.approx(ref, rel=1e-9)


def test_reml_does_not_depend_on_how_the_trend_is_scaled():
    f = field(mean="quadratic")
    X = f.X * np.array([1.0, 7.0, 0.1, 3.0, 5.0, 0.2])
    g = M.Field(f.xy, f.z, f.mean, X, f.d, float(np.linalg.slogdet(X.T @ X)[1]))
    assert M.reml_loglik(g, "spherical", 40.0, 120.0, 0.3) == pytest.approx(
        M.reml_loglik(f, "spherical", 40.0, 120.0, 0.3), rel=1e-11)


def test_gaussian_loglik_matches_scikit_learn():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    f = field()
    y = f.z - f.z.mean()
    sill, a, tau, nu = 45.0, 110.0, 0.3, 1.3
    phi = a / M.scale("matern", nu)
    kernel = (ConstantKernel(sill * (1 - tau), "fixed") * Matern(math.sqrt(2 * nu) * phi, "fixed", nu)
              + WhiteKernel(sill * tau, "fixed"))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer=None).fit(f.xy, y)
    assert M.gaussian_loglik(f.xy, y, "matern", sill, a, tau, nu) == pytest.approx(
        gp.log_marginal_likelihood_value_, rel=1e-10)


@pytest.mark.parametrize("family", M.FAMILIES)
def test_a_fit_stays_in_its_bounds_and_beats_the_truth_where_it_contains_it(family):
    rng = np.random.default_rng(3)
    xy, z = M.simulate(rng, 120, (400.0, 300.0), "exponential", 100.0, 150.0, 0.3)
    f = M.Field.of(xy, z, "constant")
    fit = M.fit(f, family)
    lo, hi = f.range_bounds()
    assert fit.converged
    assert lo <= fit.eff_range <= hi and M.TAU_BOUNDS[0] <= fit.tau <= M.TAU_BOUNDS[1]
    assert fit.aic == pytest.approx(-2 * fit.loglik + 2 * fit.k)
    if family != "spherical":           # the exponential, or a Matern family containing it
        assert fit.loglik >= M.reml_profile(f, "exponential", 150.0, 0.3)[0] - 1e-6


def test_fit_all_puts_the_lowest_aic_first():
    fits = M.fit_all(field(n=50))
    assert [f.aic for f in fits] == sorted(f.aic for f in fits)
    assert {f.family for f in fits} == set(M.FAMILIES)


def test_a_design_that_cannot_carry_the_trend_is_refused():
    xy = np.c_[np.arange(10.0), np.zeros(10)]          # points on a line: y, xy, y2 vanish
    with pytest.raises(ValueError):
        M.Field.of(xy, np.ones(10), "quadratic")


# --- joint fits, D6 intervals, D3, the stage run --------------------------------------------

import dataclasses
import time

import pandas as pd


def test_a_joint_fit_of_one_field_is_its_own_fit():
    f = field(n=50)
    own, joint = M.fit(f, "exponential"), M.fit([f], "exponential")
    assert joint.loglik == pytest.approx(own.loglik, abs=1e-9) and joint.k == own.k == 3


def test_a_joint_fit_shares_the_correlation_and_keeps_a_sill_per_field():
    f1, f2 = field(n=50, seed=1), field(n=40, seed=2)
    j = M.fit([f1, f2], "exponential")
    assert len(j.sills) == 2 and j.k == 4 and j.n == 90
    total = sum(M.reml_profile(f, "exponential", j.eff_range, j.tau)[0] for f in (f1, f2))
    assert j.loglik == pytest.approx(total, rel=1e-12)
    assert j.loglik <= M.fit(f1, "exponential").loglik + M.fit(f2, "exponential").loglik + 1e-6


@pytest.mark.parametrize("parameter", ["eff_range", "tau"])
def test_a_profile_interval_holds_the_estimate_and_its_ends_lie_on_the_threshold(parameter):
    rng = np.random.default_rng(5)
    xy, z = M.simulate(rng, 100, (400.0, 300.0), "exponential", 100.0, 120.0, 0.3)
    f = M.Field.of(xy, z, "constant")
    fit = M.fit(f, "exponential")
    iv = M.profile_interval(f, fit, parameter)
    assert iv.lo.value <= iv.estimate <= iv.hi.value and iv.above_fit < 1e-4
    for end in (iv.lo, iv.hi):
        if not end.open:
            drop = fit.loglik - M.profile_loglik(f, fit, parameter, end.value)
            assert drop == pytest.approx(M.CHI2_95_1 / 2, abs=0.01)


def test_boxcox_at_one_is_the_original_scale():
    f = field(n=50)
    fit = M.fit(f, "exponential")
    g = dataclasses.replace(f, z=M.boxcox(f.z, 1.0))
    assert M.reml_profile(g, "exponential", fit.eff_range, fit.tau)[0] == pytest.approx(fit.loglik, abs=1e-9)


def test_the_boxcox_profile_finds_a_lognormal_field():
    rng = np.random.default_rng(11)
    xy, g = M.simulate(rng, 80, (300.0, 200.0), "exponential", 0.5, 80.0, 0.3)
    f = M.Field.of(xy, np.exp(3.0 + g), "constant")
    bc = M.boxcox_profile(f, M.fit(f, "exponential"))
    assert bc.lo <= bc.lam <= bc.hi and bc.excludes_1 and bc.loglik >= bc.loglik_at_1


def test_the_stage_run_marks_one_model_per_field_and_mean_model_for_stage_4(small_stage):
    t, (record, families, pooling, boxcox) = small_stage
    assert list(record.columns) == M.RECORD_COLUMNS
    s4 = record[record["for_stage4"]]
    assert sorted(zip(s4["site"], s4["mean"])) == sorted((s, m) for s in "ABC" for m in M.MEANS)
    assert sorted(pooling["mean"]) == sorted(M.MEANS) and len(boxcox) == 3
    own = record[record["fit"] == "own"]
    for m in M.MEANS:
        p = pooling[pooling["mean"] == m].iloc[0]
        large = own[(own["mean"] == m) & own["site"].isin(["A", "B"])]
        assert p["delta_aic"] == pytest.approx(p["aic_joint"] - large["aic"].sum())
        small = own[(own["mean"] == m) & (own["site"] == "C")]
        assert (small["takes"] == ("shared" if p["pooled"] else "own")).all()


def test_a_cached_run_resumes_and_gives_the_same_record(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for site in ("A", "B"):
        xy, z = M.simulate(rng, 30, (200.0, 200.0), "exponential", 25.0, 80.0, 0.3, mean=50.0)
        rows += [{"site": site, "location_id": str(i), "x": xy[i, 0], "y": xy[i, 1], "stock": z[i]}
                 for i in range(30)]
    t = pd.DataFrame(rows)
    args = dict(families=("exponential",), large=("A", "B"), small=())
    with pytest.raises(M.OutOfTime):
        M.run_stage(t, cache=M.Cache(tmp_path / "late", deadline=time.monotonic() - 1.0), **args)
    first, again = M.Cache(tmp_path / "c"), M.Cache(tmp_path / "c")
    r1 = M.run_stage(t, cache=first, **args)[0]
    r2 = M.run_stage(t, cache=again, **args)[0]
    assert first.computed > 0 and again.computed == 0 and again.hits == first.computed
    pd.testing.assert_frame_equal(r1, r2)
