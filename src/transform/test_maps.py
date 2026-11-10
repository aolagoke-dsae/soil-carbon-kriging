"""The maps' kriging against the cross-validation's own predictive, their
mixtures' summaries against scores.MixtureT, and the hull's cells, on
synthetic fields."""
import math

import numpy as np
import pytest

import kriging as K
import maps as MP
import scores as SC


def toy_field(n=40, seed=5):
    rng = np.random.default_rng(seed)
    xy = np.c_[4.2e5 + rng.uniform(0, 300, n), 4.4e6 + rng.uniform(0, 220, n)]
    cov = rng.normal(0, 1, (n, len(K.COVARIATES))) * np.linspace(0.01, 50, len(K.COVARIATES))
    z = 50 + 0.02 * (xy[:, 0] - 4.2e5) + 2.0 * cov[:, 4] / 20 + rng.normal(0, 3, n)
    return K.FieldData("T1", np.array([f"T1-{i:03d}" for i in range(n)]), xy, z, cov)


def cells_of(fd, m=15, seed=6):
    rng = np.random.default_rng(seed)
    lo, hi = fd.xy.min(axis=0), fd.xy.max(axis=0)
    xy0 = lo + rng.uniform(0.2, 0.8, (m, 2)) * (hi - lo)
    cov0 = rng.normal(0, 1, (m, len(K.COVARIATES))) * np.linspace(0.01, 50, len(K.COVARIATES))
    return xy0, cov0


def reference(fd, model, xy0, cov0, drift):
    """The cross-validation's path: kriging.evaluate on the whole grid, then
    kriging.predictive at each cell."""
    X, X0 = K.design(model.mean, fd.xy, xy0, fd.cov, cov0, drift)
    post = K.evaluate(model.grid(), fd.xy, fd.z, X, xy0, X0, model.angle, model.ratio)
    out = []
    for j in range(len(xy0)):
        mix = K.predictive(post, j)
        q = mix.quantiles(list(MP.LEVELS))
        out.append([mix.mean(), math.sqrt(mix.var()), q[0], q[1]])
    return np.array(out)


@pytest.mark.parametrize("mean, family, angle, ratio, drift", [
    ("constant", "exponential", None, None, ()),
    ("quadratic", "spherical", None, None, ()),
    ("drift", "exponential", None, None, (0, 4)),
    ("constant", "matern", 35.0, 0.6, ()),
])
def test_the_maps_kriging_is_the_cross_validations_predictive(mean, family, angle, ratio, drift):
    fd = toy_field()
    model = K.Model("x", mean, family, 60.0, 180.0 if family != "matern" else 90.0, angle, ratio)
    xy0, cov0 = cells_of(fd)
    got = MP.kriging_cells(fd, model, xy0, cov0, drift)
    ref = reference(fd, model, xy0, cov0, drift)
    width = ref[:, 3] - ref[:, 2]
    assert np.max(np.abs(got[:, :2] - ref[:, :2]) / ref[:, :2]) < 1e-9
    assert np.max(np.abs(got[:, 2:] - ref[:, 2:]) / width[:, None]) < 1e-9


def test_mixture_summary_matches_the_mixture_of_t():
    rng = np.random.default_rng(8)
    k, cells, df = 30, 12, 37.0
    w = rng.dirichlet(np.ones(k))
    locs = 60 + rng.normal(0, 4, (cells, k))
    scales = rng.uniform(2, 6, (cells, k))
    got = MP.mixture_summary(w, locs, scales, df, chunk=5)
    for i in range(cells):
        mix = SC.MixtureT(w, locs[i], scales[i], df)
        q = mix.quantiles(list(MP.LEVELS))
        assert got[i, 0] == pytest.approx(mix.mean(), rel=1e-12)
        assert got[i, 1] == pytest.approx(math.sqrt(mix.var()), rel=1e-12)
        assert got[i, 2:] == pytest.approx(q, abs=1e-9)


def test_hull_cells_are_the_cells_inside_with_their_covariates():
    ny, nx, cell = 30, 40, 10.0
    x0, y1 = 5.0e5, 4.4e6
    g = {"elevation": np.zeros((ny, nx)), "cell": cell, "x0": x0, "y1": y1, "epsg": 32616}
    for i, c in enumerate(K.COVARIATES):
        g[c] = np.full((ny, nx), float(i))
    g[K.COVARIATES[3]][12, 15] = np.nan
    tri = np.array([[x0 + 22.0, y1 - 31.0], [x0 + 371.0, y1 - 48.0], [x0 + 140.0, y1 - 277.0]])
    rows, cols, xy0, cov0 = MP.hull_cells(tri, g)
    # brute force: barycentric coordinates of every cell centre
    r, c = np.mgrid[0:ny, 0:nx]
    px, py = x0 + (c.ravel() + 0.5) * cell, y1 - (r.ravel() + 0.5) * cell
    (ax, ay), (bx, by), (cx, cy) = tri
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
    l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
    inside = (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
    inside[12 * nx + 15] = False
    assert set(zip(rows.tolist(), cols.tolist())) == set(zip(*np.divmod(np.flatnonzero(inside), nx)))
    assert np.allclose(xy0, np.c_[x0 + (cols + 0.5) * cell, y1 - (rows + 0.5) * cell])
    assert np.all(cov0 == np.arange(len(K.COVARIATES), dtype=float))
