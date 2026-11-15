"""Stage 6's models on made-up sites: the folds (C5), the feature sets, the
forest's weights (C4) and the second displacement (C7)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import bze_covariates as BC
import bze_models as BM


def _sites(n=120, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"pointid": np.arange(n), "xcoord": rng.uniform(3e5, 9e5, n), "ycoord": rng.uniform(5.3e6, 6.1e6, n),
                      "landnutzung": rng.choice(["A", "G", "SO"], n), "kv_0_30": rng.gamma(9, 7, n)})
    for name in BC.NAMES:
        t[f"{name}_point"] = rng.normal(size=n)
        t[f"{name}_disc"] = t[f"{name}_point"] + rng.normal(scale=0.1, size=n)
    for c in BM.DESCRIPTION:
        t[c] = rng.choice(["x", "y", "z"], n)
    return t


def test_folds_partition_with_sizes_differing_by_at_most_one():
    for r in range(BM.REPEATS):
        f = BM.folds(2937, r)
        sizes = np.bincount(f, minlength=BM.FOLDS)
        assert sizes.sum() == 2937 and sizes.max() - sizes.min() <= 1
    assert not (BM.folds(500, 0) == BM.folds(500, 1)).all()


def test_feature_sets():
    t = _sites()
    Xp, cp = BM.features(t, "point")
    Xd, cd = BM.features(t, "disc")
    Xs, cs = BM.features(t, "desc")
    assert Xp.shape == Xd.shape == (120, 12) and cp[0] == "temperature_point" and cd[0] == "temperature_disc"
    assert Xs.shape == (120, 11) and "xcoord" not in cs                    # the descriptions carry no location
    assert (Xp[:, -3:].sum(axis=1) == 1).all()


def test_the_forest_weights_and_predictive():
    t = _sites()
    X, _ = BM.features(t, "point")
    y = t["kv_0_30"].to_numpy()
    train, test = np.arange(100), np.arange(100, 120)
    res = BM.forest_fold(X, y, train, test, 3)
    assert len(res["qrf"]) == 20 and all(abs(b["weight_sum"] - 1) < 1e-12 and b["weight_min"] >= 0 for b in res["qrf"])
    assert all(b["q"] == sorted(b["q"]) and b["crps"] >= 0 for b in res["qrf"])


def test_second_displacement_is_uniform_over_the_disc():
    """C7: within 4 km, the distance / 4 km distributed as sqrt(U)."""
    xy = np.zeros((20000, 2))
    d = np.hypot(*BM.displace(xy, 0).T)
    assert d.max() <= BC.RADIUS
    assert stats.kstest((d / BC.RADIUS) ** 2, "uniform").pvalue > 0.01
    assert not np.allclose(BM.displace(xy, 0), BM.displace(xy, 1))
