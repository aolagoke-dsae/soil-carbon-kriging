"""Geometric anisotropy on simulated fields, without raw/."""
import math

import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import pdist

import anisotropy as A
import reml as M


def test_with_ratio_one_the_rotated_distance_is_the_distance():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (20, 2))
    i, j = np.triu_indices(20, 1)
    d = A.rotated_distances(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1], 37.0, 1.0)
    assert d == pytest.approx(pdist(xy), rel=1e-12)


def anisotropic_field(angle, ratio, n=150, seed=3):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 500, (n, 2))
    i, j = np.triu_indices(n, 1)
    d = A.rotated_distances(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1], angle, ratio)
    V = 100.0 * M._c_matrix(d, "exponential", 200.0, 0.1, None)
    return xy, 50.0 + linalg.cholesky(V, lower=True) @ rng.standard_normal(n)


def test_an_anisotropic_field_is_found():
    xy, z = anisotropic_field(60.0, 0.4)
    f = M.Field.of(xy, z, "constant")
    iso = M.fit(f, "exponential")
    an = A.fit_anisotropic(f, "exponential", iso.eff_range, iso.tau)
    assert an.converged and an.aic < iso.aic - 2
    # 150 points see a 200 m range only a few times over: the direction to within 25 degrees
    assert abs((an.angle - 60.0 + 90.0) % 180.0 - 90.0) < 25.0 and 0.15 < an.ratio < 0.75


def test_on_an_isotropic_field_the_refit_cannot_fall_below_the_isotropic_fit():
    rng = np.random.default_rng(8)
    xy, z = M.simulate(rng, 100, (400.0, 300.0), "spherical", 50.0, 150.0, 0.2, mean=30.0)
    f = M.Field.of(xy, z, "constant")
    iso = M.fit(f, "spherical")
    an = A.fit_anisotropic(f, "spherical", iso.eff_range, iso.tau)
    assert an.loglik >= iso.loglik - 1e-6 and an.k == iso.k + 2
    assert 0.0 <= an.angle < 180.0 and A.RATIO_BOUNDS[0] <= an.ratio <= A.RATIO_BOUNDS[1]
