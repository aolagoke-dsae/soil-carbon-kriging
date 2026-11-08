"""Stage 3's terrain analysis against closed forms: a plane, a paraboloid, a
pit and a tilted plane for the flow, plus the resampling, from one tile and
from two."""
import math

import numpy as np
import pytest
from skimage.morphology import reconstruction

import cog
import patches as P
import terrain as T

CELL = T.CELL


def _surface(fn, n=81):
    """fn(u, v) on an n x n grid of cell centres, u east and v north of the
    centre cell, rows running south."""
    m = n // 2
    dr, dc = np.mgrid[0:n, 0:n] - m
    u, v = dc * CELL, -dr * CELL
    return fn(u, v), u, v


def test_on_a_plane_every_window_gives_its_slope_and_no_curvature_or_tpi():
    z, u, v = _surface(lambda u, v: 100.0 + 0.02 * u - 0.01 * v)
    inner = (slice(40, 41), slice(40, 41))
    for n in T.WINDOWS:
        fit = T.local_fit(z, n)
        assert fit["p"][inner] == pytest.approx(0.02, abs=1e-12)
        assert fit["q"][inner] == pytest.approx(-0.01, abs=1e-12)
        assert T.slope_degrees(fit)[inner] == pytest.approx(math.degrees(math.atan(math.hypot(0.02, 0.01))), abs=1e-9)
        assert abs(T.mean_curvature(fit)[inner]).max() < 1e-12
    for r in (50.0, 150.0):
        assert abs(T.tpi(z, r)[inner]).max() < 1e-9


def test_on_a_paraboloid_every_window_gives_its_slope_and_mean_curvature():
    """The quadratic fit is exact on a quadratic, at any window size."""
    A, B, C, D, E = -2e-4, -1e-4, 5e-5, 0.01, -0.02
    z, u, v = _surface(lambda u, v: 50.0 + A * u * u + B * v * v + C * u * v + D * u + E * v)
    p, q = 2 * A * u + C * v + D, 2 * B * v + C * u + E
    r, s, t = 2 * A, C, 2 * B
    H = -((1 + q * q) * r - 2 * p * q * s + (1 + p * p) * t) / (2 * (1 + p * p + q * q) ** 1.5)
    inner = (slice(20, 61), slice(20, 61))
    for n in T.WINDOWS:
        fit = T.local_fit(z, n)
        assert np.allclose(fit["p"][inner], p[inner], rtol=1e-9, atol=1e-12)
        assert np.allclose(T.mean_curvature(fit)[inner], H[inner], rtol=1e-8, atol=1e-12)
        assert np.allclose(T.slope_degrees(fit)[inner],
                           np.degrees(np.arctan(np.hypot(p, q)))[inner], rtol=1e-9, atol=1e-12)


def test_a_dome_is_convex_and_a_bowl_concave():
    dome, _, _ = _surface(lambda u, v: -(u * u + v * v) / 2000.0)
    fit = T.local_fit(dome, 9)
    assert T.mean_curvature(fit)[40, 40] == pytest.approx(1 / 1000.0, rel=1e-9)
    assert T.mean_curvature(T.local_fit(-dome, 9))[40, 40] == pytest.approx(-1 / 1000.0, rel=1e-9)


def test_tpi_on_a_paraboloid_is_the_constant_its_offsets_give():
    R = 800.0
    z, _, _ = _surface(lambda u, v: -(u * u + v * v) / (2 * R), n=121)
    for radius in T.RADII[:2]:
        dr, dc = T.disc_offsets(radius)
        expect = np.mean((dr * CELL) ** 2 + (dc * CELL) ** 2) / (2 * R)
        got = T.tpi(z, radius)[60, 60]
        assert got == pytest.approx(expect, rel=1e-9)


def test_the_disc_excludes_its_centre_and_keeps_its_radius():
    dr, dc = T.disc_offsets(50.0)
    assert not np.any((dr == 0) & (dc == 0))
    assert np.all((dr * CELL) ** 2 + (dc * CELL) ** 2 <= 2500.0)
    assert len(dr) == 80                              # 81 centres within 50 m, less the centre


def test_a_pit_is_filled_so_that_every_cell_drains():
    z, u, v = _surface(lambda u, v: 100.0 + 0.001 * u, n=41)
    z[18:23, 18:23] -= 0.5                            # a pit 0.5 m deep
    filled = T.fill(z)
    assert np.all(filled >= z)
    edge = T.edge_mask(z.shape)
    assert np.array_equal(filled[edge], z[edge])
    seed = z.copy()
    seed[1:-1, 1:-1] = z.max()
    flat = reconstruction(seed, z, method="erosion", footprint=np.ones((3, 3)))
    assert np.abs(filled - flat).max() < 1e-6
    pad = np.pad(filled, 1, constant_values=np.inf)
    lower = np.zeros(z.shape, bool)
    for di, dj in T.NEIGHBOURS:
        lower |= pad[1 + di:1 + di + 41, 1 + dj:1 + dj + 41] < filled
    assert lower[~edge].all()


def test_on_a_tilted_plane_each_column_carries_the_area_upslope_of_it():
    """Flow along a grid axis: what reaches column k is all of columns 0..k."""
    ny, nx = 30, 40
    z = 100.0 - 0.01 * CELL * np.arange(nx)[None, :] * np.ones((ny, 1))
    filled = T.fill(z)
    fr = T.fd8_fractions(filled)
    area = T.accumulate(filled, fr, np.full(z.shape, CELL * CELL))
    for k in range(nx - 1):
        assert area[:, k].sum() == pytest.approx((k + 1) * ny * CELL * CELL, rel=1e-12)
    outlets = fr.sum(axis=0) == 0
    assert area[outlets].sum() == pytest.approx(ny * nx * CELL * CELL, rel=1e-12)


def test_fd8_shares_flow_in_proportion_to_the_slope_towards_each_neighbour():
    z = np.array([[3.0, 3.0, 3.0], [3.0, 2.0, 1.0], [3.0, 3.0, 3.0]])
    fr = T.fd8_fractions(z)
    centre = fr[:, 1, 1]
    assert centre.sum() == pytest.approx(1.0)
    assert centre[T.NEIGHBOURS.index((0, 1))] == pytest.approx(1.0)     # the only lower neighbour


def test_twi_takes_the_floor_where_the_filled_dem_is_level():
    z = np.full((21, 21), 10.0)
    w = T.wetness(z)
    inner = w["twi"][T.FLOOR][5:16, 5:16]
    assert np.all(np.isfinite(inner))
    assert np.allclose(w["tanb"][5:16, 5:16], 0.0, atol=1e-9)
    area = w["area"][5:16, 5:16]
    assert np.allclose(inner, np.log((area / CELL) / T.FLOOR))


def _plane(lon, lat):
    return 1000.0 * (lon + 88.17) + 2000.0 * (lat - 39.92)


def _part(geo, window, version):
    r0, r1, c0, c1 = window
    rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    tv = T.TileVersion("u" + version, "b" + version, version, 1, "e")
    return T.Part(tv, geo, window, _plane(geo.centre_lon(cols), geo.centre_lat(rows)))


def test_resampling_reproduces_a_plane_in_longitude_and_latitude(synthetic_tile):
    """Bilinear interpolation is exact on a plane: any shift between the
    patch and the tile's grid, or between a window and its tile, would show."""
    _, transform, _, loc = synthetic_tile
    d = transform.a
    geo = cog.Geo("EPSG:4269", transform.c, transform.f, d, -transform.e, 600, 600, 64, 64, None)
    grid = P.patch_grids(loc)["T1"]
    window = cog.window_for(grid, geo)
    assert window[0] > 0 and window[2] > 0                 # a window inside the tile
    got = T.resample(_part(geo, window, "20240104"), grid)
    lon, lat = T.patch_lonlat(grid, geo.crs)
    assert np.isfinite(got).all()
    assert np.abs(got - _plane(lon, lat)).max() < 1e-6
    x, y = loc["x"].to_numpy(), loc["y"].to_numpy()
    lon_l, lat_l = P.transformer(grid.epsg, geo.crs).transform(x, y)
    # a plane in lon/lat is not exactly a plane in UTM, so interpolating the
    # patch at the locations is close, not exact
    assert np.abs(T.at_locations(grid, got, x, y) - _plane(np.asarray(lon_l), np.asarray(lat_l))).max() < 1e-3


def test_two_tiles_off_one_lattice_are_each_resampled_on_their_own_grid(synthetic_tile):
    """IL-RS's case: the newer tile's cells lie 1e-5 of a cell off the older
    one's lattice. A plane comes through exactly, the newer version is used
    wherever its cells surround a centre, and the older one where the newer
    has no value."""
    _, transform, _, loc = synthetic_tile
    d = transform.a
    grid = P.patch_grids(loc)["T1"]
    old = cog.Geo("EPSG:4269", transform.c, transform.f, d, d, 600, 600, 64, 64, None)
    new = cog.Geo("EPSG:4269", transform.c, transform.f + 1e-5 * d, d, d, 600, 600, 64, 64, None)
    lon, lat = T.patch_lonlat(grid, "EPSG:4269")
    split = int(np.median(old.frac_row(lat)))              # the patch straddles it
    a = _part(old, (split, 599, 0, 599), "20240104")
    b = _part(new, (0, split + 6, 0, 599), "20260902")     # six rows shared, as at a tile's edge
    dem, used = T.compose([a, b], grid)
    assert np.abs(dem - _plane(lon, lat)).max() < 1e-9
    r = new.frac_row(lat)
    assert np.array_equal(used == 1, (r >= 0) & (r <= split + 6))
    assert set(np.unique(used)) == {0, 1}
    col = int(np.median(new.frac_col(lon)))
    b.values[split + 3, col] = np.nan                      # a nodata cell in the shared rows
    dem, used2 = T.compose([a, b], grid)
    assert np.isfinite(dem).all() and np.abs(dem - _plane(lon, lat)).max() < 1e-9
    assert ((used == 1) & (used2 == 0)).sum() > 0           # there the older version's value
