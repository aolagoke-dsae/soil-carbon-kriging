"""D3's two readings on synthetic grids, where the answers are known, and
the rasters' building blocks (Checkpoint 6, C3's synthetic half)."""
import gzip
import math

import numpy as np
import pandas as pd
import pytest

import bze_covariates as BC
import checkpoint6 as C6


def _grid(f, n=101, d=200.0, x0=0.0, y0=20200.0):
    r, c = np.mgrid[0:n, 0:n]
    x = x0 + (c + 0.5) * d
    y = y0 - (r + 0.5) * d
    return BC.Raster(f(x, y).astype(float), x0, y0, d, "EPSG:25832")


def test_a_plane_averages_to_its_value_at_a_cell_centre():
    """Over a disc centred on a cell centre the cells are symmetric, so a
    plane's disc mean is its value there."""
    r = _grid(lambda x, y: 3.0 + 0.01 * x - 0.02 * y)
    x, y = r.x0 + 50.5 * r.d, r.y0 - 50.5 * r.d
    got = BC.read(r, np.array([x]), np.array([y]))
    assert got["point"][0] == pytest.approx(3.0 + 0.01 * x - 0.02 * y)
    assert got["disc"][0] == pytest.approx(got["point"][0], rel=1e-12) and got["share"][0] == 1.0


def test_the_disc_holds_the_cells_within_4_km_and_skips_missing_ones():
    r = _grid(lambda x, y: np.ones_like(x))
    r.values[:, :50] = np.nan                                    # the western half has no data
    x, y = r.x0 + 50.5 * r.d, r.y0 - 50.5 * r.d
    got = BC.read(r, np.array([x]), np.array([y]))
    n_disc = len(BC.disc_offsets(BC.RADIUS, r.d)[0])
    assert n_disc == sum(1 for i in range(-20, 21) for j in range(-20, 21) if (i * 200) ** 2 + (j * 200) ** 2 <= 4000 ** 2)
    assert got["disc"][0] == 1.0 and 0.45 < got["share"][0] < 0.55
    assert C6.brute_disc_mean(r, x, y) == 1.0


def test_brute_force_agrees_off_the_cell_centres():
    rng = np.random.default_rng(3)
    r = _grid(lambda x, y: rng.normal(size=x.shape))
    for x, y in rng.uniform(4000, 16000, (5, 2)):
        assert BC.read(r, np.array([x]), np.array([y]))["disc"][0] == pytest.approx(C6.brute_disc_mean(r, x, y), rel=1e-12)


def test_neighbourhood_mean_leaves_the_cell_out():
    z = np.zeros((41, 41))
    z[20, 20] = 100.0
    m = BC.neighbourhood_mean(z, 1000.0, 200.0)
    n = len(BC.disc_offsets(1000.0, 200.0, exclude_centre=True)[0])
    assert m[20, 20] == pytest.approx(0.0, abs=1e-9) and m[20, 21] == pytest.approx(100.0 / n)


def test_slope_of_a_plane():
    r = _grid(lambda x, y: 0.1 * x)
    assert np.allclose(BC.slope_degrees(r.values, r.d), math.degrees(math.atan(0.1)))


def test_ascii_grid_reading():
    raw = gzip.compress(b"NCOLS 3\nNROWS 2\nXLLCORNER 10\nYLLCORNER 20\nCELLSIZE 1000\nNODATA_VALUE -999\n"
                        b"1 2 -999\n4 5 6\n")
    r = BC.read_ascii_grid(raw, "EPSG:31467")
    assert (r.x0, r.y0, r.d) == (10.0, 2020.0, 1000.0) and np.isnan(r.values[0, 2]) and r.values[1, 0] == 4.0


def test_c2_as_amended():
    cov = pd.DataFrame({"pointid": range(C6.N_SITES), "a_point": 1.0, "a_disc": 1.0, "a_share": 1.0})
    cov["kept"] = BC.kept(cov)
    assert C6.c2_coverage(cov).passed
    cov.loc[0, "a_point"] = np.nan
    cov.loc[1, "a_share"] = 0.4
    cov["kept"] = BC.kept(cov)
    assert not cov.loc[0, "kept"] and not cov.loc[1, "kept"] and C6.c2_coverage(cov).passed
    cov["kept"] = True                                          # a site without its covariate kept: fails
    assert not C6.c2_coverage(cov).passed
    cov.loc[2:60, "a_point"] = np.nan                           # more than 1 % left out: fails
    cov["kept"] = BC.kept(cov)
    assert not C6.c2_coverage(cov).passed
