"""Make the package layout importable from a plain `pytest` at the repo root,
and share one small stand-in for the Stage 2 REML run between the tests."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
for _p in ("src/ingest", "src/checks", "src/transform"):
    sys.path.insert(0, str(ROOT / _p))


@pytest.fixture(scope="session")
def small_stage():
    """Three hand-made fields (two large, one small) run through Stage 2's
    REML steps with the exponential family only."""
    import numpy as np
    import pandas as pd
    import reml as M
    rng = np.random.default_rng(21)
    rows = []
    for site, n in (("A", 45), ("B", 40), ("C", 30)):
        xy, z = M.simulate(rng, n, (300.0, 250.0), "exponential", 25.0, 90.0, 0.3, mean=50.0)
        rows += [{"site": site, "location_id": str(i), "x": xy[i, 0], "y": xy[i, 1], "stock": z[i]}
                 for i in range(n)]
    t = pd.DataFrame(rows)
    return t, M.run_stage(t, families=("exponential",), large=("A", "B"), small=("C",))


@pytest.fixture(scope="session")
def synthetic_tile(tmp_path_factory):
    """A small Cloud Optimized GeoTIFF laid out as USGS lays out its 1/3
    arc-second tiles (NAD83 longitude and latitude, cells of 1/10800 degree,
    a corner tiepoint, LZW with the floating-point predictor, nodata
    -999999), 600 x 600 cells in blocks of 128, around one made-up field near
    88.17 W, 39.92 N. Returns (bytes, rasterio transform, values, locations)."""
    import numpy as np
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    d = 1.0 / 10800.0
    west, north = -88.2 - 6 * d, 39.95 + 6 * d
    n = 600
    rows, cols = np.mgrid[0:n, 0:n]
    lon = west + (cols + 0.5) * d
    lat = north - (rows + 0.5) * d
    z = (210.0 + 900.0 * (lon + 88.17) - 400.0 * (lat - 39.92)
         + 0.8 * np.sin(cols / 23.0) * np.cos(rows / 31.0)).astype("float32")
    path = tmp_path_factory.mktemp("tile") / "USGS_13_n40w089_20240104.tif"
    with rasterio.open(path, "w", driver="COG", width=n, height=n, count=1, dtype="float32",
                       crs="EPSG:4269", transform=rasterio.Affine(d, 0.0, west, 0.0, -d, north),
                       nodata=-999999.0, compress="LZW", predictor=3, blocksize=128,
                       overview_resampling="average") as dst:
        dst.write(z, 1)
    with rasterio.open(path) as src:
        transform = src.transform
    t = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
    rng = np.random.default_rng(3)
    plon = -88.17 + rng.uniform(-0.002, 0.002, 12)
    plat = 39.92 + rng.uniform(-0.0015, 0.0015, 12)
    x, y = t.transform(plon, plat)
    locations = pd.DataFrame({"site": "T1", "location_id": [str(i) for i in range(12)],
                              "lon": plon, "lat": plat, "epsg": 32616, "x": x, "y": y})
    return path.read_bytes(), transform, z, locations
