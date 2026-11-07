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
