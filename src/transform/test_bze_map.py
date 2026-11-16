"""D8's building blocks: the sparse weights equal Stage 4's, the summary
equals Stage 4's weighted quantiles, and the convolved disc means equal D3's
reader at cell centres."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

import bze_covariates as BC
import bze_map as MP
import forests as F
import scores as SC


def test_sparse_weights_and_summary_equal_stage_4s():
    rng = np.random.default_rng(4)
    X, y, Xc = rng.normal(size=(400, 4)), rng.gamma(5, 10, 400), rng.normal(size=(60, 4))
    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=11, max_features=2, random_state=2).fit(X, y)
    W = MP.weights_dense(rf, X, Xc)
    assert np.abs(W - F.qrf_weights(rf, X, Xc)).max() < 1e-12
    s = MP.summarise(W, y)
    for j in (0, 17, 59):
        d = SC.Weighted(y, W[j])
        assert np.allclose([s[f"q{lev:.2f}"][j] for lev in MP.LEVELS], d.quantiles(list(MP.LEVELS)))
        assert abs(s["mean"][j] - d.mean()) < 1e-9


def test_convolved_disc_means_equal_the_reader_at_cell_centres():
    rng = np.random.default_rng(5)
    v = rng.normal(size=(80, 90))
    v[:5, :] = np.nan
    r = BC.Raster(v, 0.0, 16000.0, 200.0, "EPSG:25832")
    m, share = MP.disc_mean_raster(v, 200.0)
    for i, j in ((40, 45), (8, 10), (79, 0)):
        x, y = (j + 0.5) * 200.0, 16000.0 - (i + 0.5) * 200.0
        got = BC.read(r, np.array([x]), np.array([y]))
        assert abs(got["disc"][0] - m[i, j]) < 1e-9 and abs(got["share"][0] - share[i, j]) < 1e-12
