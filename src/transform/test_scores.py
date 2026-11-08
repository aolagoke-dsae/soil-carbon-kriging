"""Stage 4's scores against closed forms, quadrature and hand-worked cases."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

import scores as S


def quad_crps(cdf, y, lo, hi):
    left = integrate.quad(lambda x: cdf(x) ** 2, lo, y, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
    right = integrate.quad(lambda x: (1.0 - cdf(x)) ** 2, y, hi, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
    return left + right


@pytest.mark.parametrize("y", [-3.0, 0.2, 4.5])
def test_normal_crps_against_quadrature(y):
    d = stats.norm(1.0, 2.0)
    assert S.crps_normal(y, 1.0, 2.0) == pytest.approx(quad_crps(d.cdf, y, -60, 60), rel=1e-9)


@pytest.mark.parametrize("df, y", [(5.0, -2.0), (12.0, 0.7), (60.0, 9.0)])
def test_student_t_crps_against_quadrature(df, y):
    d = stats.t(df, loc=2.0, scale=3.0)
    ref = quad_crps(d.cdf, y, -1e4, 1e4)
    assert S.crps_t(y, 2.0, 3.0, df) == pytest.approx(ref, rel=1e-8)


def test_student_t_crps_tends_to_the_normal():
    assert S.crps_t(1.3, 0.0, 1.0, 1e7) == pytest.approx(S.crps_normal(1.3, 0.0, 1.0), rel=1e-6)


def test_one_component_mixture_matches_the_closed_form():
    m = S.MixtureT([1.0], [50.0], [6.0], 40.0)
    for y in (31.0, 50.0, 70.5):
        assert m.crps(y) == pytest.approx(S.crps_t(y, 50.0, 6.0, 40.0), rel=1e-10)
    assert m.spread() == pytest.approx(6.0 * S.t_half_spread(40.0), rel=1e-12)


def test_mixture_against_quadrature_and_its_quantiles_hit_their_levels():
    w, loc, sc = [0.2, 0.5, 0.3], [40.0, 55.0, 61.0], [3.0, 5.0, 9.0]
    m = S.MixtureT(w, loc, sc, 30.0)

    def cdf(x):
        return sum(wi * stats.t.cdf(x, 30.0, li, si) for wi, li, si in zip(w, loc, sc))
    for y in (35.0, 52.0, 80.0):
        assert m.crps(y) == pytest.approx(quad_crps(cdf, y, -2e3, 2e3), rel=1e-8)
    q = m.quantiles()
    assert np.max(np.abs(m.cdf(q) - S.LEVELS)) < 1e-11
    assert np.all(np.diff(q) > 0)
    assert m.mean() == pytest.approx(np.dot(w, loc))


def test_weighted_crps_two_ways_and_its_quantiles():
    x, w = np.array([3.0, 1.0, 2.0, 2.0, 7.0]), np.array([0.1, 0.4, 0.2, 0.2, 0.1])
    e = S.Weighted(x, w)
    for y in (0.0, 1.5, 2.0, 6.0, 9.0):
        assert e.crps(y) == pytest.approx(e.crps_integral(y), abs=1e-12)
    # sorted: 1 (0.4), 2 (0.2), 2 (0.2), 3 (0.1), 7 (0.1); cumulative 0.4, 0.6, 0.8, 0.9, 1.0
    assert list(e.quantiles([0.4, 0.41, 0.6, 0.85, 0.95])) == [1.0, 2.0, 2.0, 3.0, 7.0]
    assert e.mean() == pytest.approx(0.4 + 0.8 + 0.3 + 0.7)
    # one atom: the CRPS is the absolute error
    assert S.Weighted([5.0]).crps(8.0) == pytest.approx(3.0)


def test_transformed_identity_is_the_base_and_log_is_lognormal():
    base = S.MixtureT([0.6, 0.4], [3.9, 4.1], [0.1, 0.15], 80.0)
    ident = S.Transformed(base, lambda z: z, lambda y: y, lambda z: np.ones_like(z))
    assert ident.mean() == pytest.approx(base.mean(), rel=1e-12)
    assert ident.crps(4.2) == pytest.approx(base.crps(4.2), rel=1e-9)
    lg = S.Transformed(base, *S.boxcox_maps(0.0))
    m_ref = sum(wi * integrate.quad(lambda z: math.exp(z) * stats.t.pdf(z, 80.0, li, si), li - 3, li + 3,
                                    epsabs=1e-12)[0] for wi, li, si in zip([0.6, 0.4], [3.9, 4.1], [0.1, 0.15]))
    assert lg.mean() == pytest.approx(m_ref, rel=1e-6)
    ref = quad_crps(lambda y: base.cdf(np.log(y))[0], 55.0, 1e-9, 400.0)
    assert lg.crps(55.0) == pytest.approx(ref, rel=1e-7)
    assert np.allclose(lg.quantiles(), np.exp(base.quantiles()))


def test_boxcox_maps_invert_each_other():
    g, gi, gp, dom = S.boxcox_maps(2.76)
    assert dom == (-1.0 / 2.76, math.inf)
    y = np.array([20.0, 60.0, 117.0])
    assert np.allclose(g(gi(y)), y, rtol=1e-13)
    z = gi(y)
    h = 1e-6
    assert np.allclose(gp(z), (g(z + h) - g(z - h)) / (2 * h), rtol=1e-6)


def test_coverage_and_point_scores_by_hand():
    q = np.tile(S.LEVELS * 10.0, (4, 1))           # quantile at level L is 10 L
    y = np.array([0.1, 2.0, 5.0, 9.9])
    qcp, picp = S.coverage(q, y)
    assert qcp[S.level_index(0.5)] == pytest.approx(0.75)     # 0.1, 2.0 and 5.0 are <= the median, 5.0
    lo, hi = S.central_bounds(0.9)
    assert (lo, hi) == (S.level_index(0.05), S.level_index(0.95))
    assert picp[-1] == pytest.approx(0.5)                     # [0.25, 9.75] holds 2.0 and 5.0
    p = S.point_scores([2.0, 4.0], [1.0, 5.0])
    assert p["me"] == 0.0 and p["rmse"] == pytest.approx(1.0) and p["mec"] == pytest.approx(1.0 - 2.0 / 8.0)


@pytest.mark.parametrize("df", [5.0, 41.0, 201.0])
def test_interpolated_t_distribution_function_matches_stdtr(df):
    z = np.concatenate([np.linspace(-45.0, 45.0, 400_001), [-1e3, 0.0, 39.9999, 40.0, 40.0001, 1e3]])
    assert np.max(np.abs(S.t_table(df).cdf(z) - stats.t.cdf(z, df))) < 2e-12


def test_mixture_variance_and_fast_cdf():
    w, loc, sc, df = [0.3, 0.7], [10.0, 14.0], [2.0, 3.5], 25.0
    m = S.MixtureT(w, loc, sc, df)
    mean = sum(wi * li for wi, li in zip(w, loc))
    ref = sum(wi * integrate.quad(lambda x: (x - mean) ** 2 * stats.t.pdf(x, df, li, si), -400, 400,
                                  limit=400, epsabs=1e-12)[0] for wi, li, si in zip(w, loc, sc))
    assert m.var() == pytest.approx(ref, rel=1e-9)
    x = np.linspace(-20.0, 50.0, 1001)
    assert np.max(np.abs(m.fast_cdf(x) - m.cdf(x))) < 2e-12


@pytest.mark.parametrize("lam", [3.63, 1.48, 0.62])
def test_boxcox_back_transform_against_quadrature(lam):
    """D6: the predictive conditioned on positive stocks, where Z's support
    runs past the transform's domain; the mean and the CRPS on the stocks' scale."""
    g, gi, gp, dom = S.boxcox_maps(lam)
    yloc, ysd = 60.0, 9.0
    locs = gi(np.array([yloc - 4.0, yloc + 3.0]))
    scales = ysd * gp(locs) ** -1.0
    base = S.MixtureT([0.45, 0.55], locs, scales, 45.0)
    tr = S.Transformed(base, g, gi, gp, dom)
    p0 = base.cdf(-1.0 / lam)[0]
    assert tr.p_outside == pytest.approx(p0, rel=1e-14)
    if lam == 3.63:
        assert p0 > 0.025            # the case that needs the truncation

    def F(t):
        return (base.cdf(gi(np.array([t])))[0] - p0) / (1.0 - p0)
    mean_ref = integrate.quad(lambda t: 1.0 - F(t), 0.0, 400.0, limit=400, epsabs=1e-11)[0]
    assert tr.mean() == pytest.approx(mean_ref, rel=1e-9)
    for y in (35.0, 61.0, 95.0):
        assert tr.crps(y) == pytest.approx(quad_crps(F, y, 0.0, 400.0), rel=1e-8)
    q = tr.quantiles()
    assert np.all(q > 0) and np.all(np.diff(q) > 0)
    assert np.max(np.abs(tr.cdf(q) - S.LEVELS)) < 1e-11
    assert tr.cdf(0.0)[0] == 0.0 and tr.cdf(-5.0)[0] == 0.0


def test_transformed_without_a_bound_cuts_nothing():
    base = S.MixtureT([1.0], [4.0], [0.2], 60.0)
    tr = S.Transformed(base, *S.boxcox_maps(0.0))
    assert tr.p_outside == 0.0
    assert np.allclose(tr.quantiles(), np.exp(base.quantiles()), rtol=1e-14)
