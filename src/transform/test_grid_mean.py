"""G1 and G3 on small cases; the run itself is RUNBOOK, Stage 5, Addendum on M4."""
import numpy as np
import pytest

import field_means as FM
import grid_mean as G
import kriging as K
import reml as M

RNG = np.random.default_rng(9)
XY = RNG.uniform(0, 100, (15, 2))
Z = RNG.normal(40, 4, 15)
CELLS = np.array([(x, y) for x in np.arange(5, 100, 10.0) for y in np.arange(5, 100, 10.0)])
PTS = np.array([(x, y) for x in (20.0, 50.0, 80.0) for y in (20.0, 50.0, 80.0)])
MODEL = K.Model("a", "constant", "exponential", 10.0, 200.0)


def _exact_rbb(a, nu):
    return FM.exact_mean_correlation(CELLS, "exponential", a, nu)


def test_pure_nugget_equals_random_sampling():
    """G3: with tau = 1, the grid's sample mean has variance sigma^2 / n, D4's D^2 / n."""
    (v, bias), = G.sample_mean_mse(MODEL, XY, Z, None, PTS, None, CELLS, None, (), [(50.0, None, 1.0)], _exact_rbb)
    assert v == pytest.approx(1.0 / len(PTS), abs=1e-12) and bias == 0.0


def test_the_variance_is_the_quadratic_form():
    """G1's variance term equals a' Sigma a for the difference between the
    points' mean (with the nugget) and the cells' mean (without), per unit sill."""
    a, tau = 40.0, 0.3
    (v, _), = G.sample_mean_mse(MODEL, XY, Z, None, PTS, None, CELLS, None, (), [(a, None, tau)], _exact_rbb)
    allxy = np.vstack([PTS, CELLS])
    S = (1 - tau) * M.correlation("exponential", K.distances(allxy, allxy), a, None)
    S[np.arange(len(PTS)), np.arange(len(PTS))] += tau
    w = np.r_[np.full(len(PTS), 1 / len(PTS)), np.full(len(CELLS), -1 / len(CELLS))]
    assert v == pytest.approx(w @ S @ w, rel=1e-3)          # the cross term is binned at 5 cm


def test_a_trend_gives_a_bias_where_the_grid_misses_it():
    """Under the quadratic mean, a grid in one corner of the cells has a bias."""
    m = K.Model("b", "quadratic", "exponential", 10.0, 200.0)
    z = 0.5 * XY[:, 0] + RNG.normal(0, 0.1, 15)
    (_, bias), = G.sample_mean_mse(m, XY, z, None, PTS * 0.3, None, CELLS, None, (), [(40.0, None, 0.3)], _exact_rbb)
    assert bias < -10.0          # the corner's x is about 15, the cells' about 50: 0.5 x (15 - 50)
