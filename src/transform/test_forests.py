"""Stage 4's forests: Meinshausen's weights, the out-of-bag residuals, the stacked table."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import folds as FO
import forests as F
import kriging as K
import scores as SC


def small_forest(n=300, trees=60, seed=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 3))
    y = 3.0 * X[:, 0] - X[:, 1] + rng.normal(0, 0.3, n)
    rf = RandomForestRegressor(n_estimators=trees, max_features=2, min_samples_leaf=11, bootstrap=True,
                               oob_score=True, random_state=seed).fit(X[:200], y[:200])
    return rf, X, y


def test_qrf_weights_are_meinshausens():
    rf, X, y = small_forest()
    W = F.qrf_weights(rf, X[:200], X[200:])
    assert np.all(W >= 0) and np.allclose(W.sum(axis=1), 1.0, atol=1e-12)
    j = 7
    ref = np.zeros(200)
    for tree in rf.estimators_:
        leaves_train = tree.apply(X[:200].astype(np.float32))
        leaf = tree.apply(X[200 + j:201 + j].astype(np.float32))[0]
        members = np.flatnonzero(leaves_train == leaf)
        ref[members] += 1.0 / len(members)
    assert np.allclose(W[j], ref / len(rf.estimators_), atol=1e-12)
    # in-sample, the leaf holds the location itself
    W_in = F.qrf_weights(rf, X[:200], X[:5])
    assert np.all(W_in[np.arange(5), np.arange(5)] > 0)


def test_out_of_bag_predictions_from_the_trees_own_samples():
    rf, X, y = small_forest()
    assert np.allclose(F.oob_from_trees(rf, X[:200]), rf.oob_prediction_, atol=1e-12)


def test_min_leaf_holds_more_than_ten_locations():
    rf, X, y = small_forest()
    for tree, sample in zip(rf.estimators_, rf.estimators_samples_):
        leaves = tree.apply(X[:200][np.unique(sample)].astype(np.float32))
        assert np.bincount(leaves)[np.unique(leaves)].min() >= 11


def test_qrf_is_calibrated_on_a_simple_regression():
    """M1's simulated regression, smaller: 1,500 + 1,500 points, 300 trees."""
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 1, (3000, 1))
    y = 2.0 * x[:, 0] + rng.standard_normal(3000)
    rf = RandomForestRegressor(n_estimators=300, max_features=1, min_samples_leaf=11, random_state=3,
                               n_jobs=2).fit(x[:1500], y[:1500])
    W = F.qrf_weights(rf, x[:1500], x[1500:])
    q = np.array([SC.Weighted(y[:1500], w).quantiles() for w in W])
    qcp = (y[1500:, None] <= q).mean(axis=0)
    inside = (SC.LEVELS >= 0.05) & (SC.LEVELS <= 0.95)
    assert np.max(np.abs(qcp[inside] - SC.LEVELS[inside])) < 0.05


def test_stack_keeps_the_kriging_order_and_the_folds():
    rng = np.random.default_rng(0)
    rows, t_rows = [], []
    for site, n in (("B", 12), ("A", 15)):
        for i in range(n):
            rows.append({"site": site, "location_id": f"{site}{i:02d}", "lon": -88 + rng.normal(), "lat": 40.0,
                         "x": rng.uniform(0, 100), "y": rng.uniform(0, 100), "stock": rng.normal(60, 5)})
            t_rows.append({"site": site, "location_id": f"{site}{i:02d}",
                           **{c: rng.normal() for c in K.COVARIATES}})
    stocks, terrain = pd.DataFrame(rows), pd.DataFrame(t_rows)
    folds = FO.assign(stocks)
    data = F.stack(stocks, terrain, folds)
    assert list(data.site[:15]) == ["A"] * 15 and list(data.ids[:3]) == ["A00", "A01", "A02"]
    fields = K.load_fields(stocks.rename(columns={}), terrain, folds)
    assert np.array_equal(data.y[:15], fields["A"].z) and np.array_equal(data.folds[1][15:], fields["B"].folds[1])
    train, test = data.split(2, 3)
    assert set(np.flatnonzero(data.folds[2] == 3)) == set(test) and not set(train) & set(test)


def test_residual_kriging_is_the_kriging_shifted():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 200, (30, 2))
    e = rng.normal(0, 2, 30)
    shift = np.array([50.0, 61.0])
    out = F.residual_kriging(xy[:28], e[:28], xy[28:], shift, np.array([52.0, 60.0]), 5.0, 600.0)
    grid = K.make_grid("exponential", 5.0, 600.0)
    X, X0 = K.design("constant", xy[:28], xy[28:])
    post = K.evaluate(grid, xy[:28], e[:28], X, xy[28:], X0)
    for j in range(2):
        q = K.predictive(post, j).quantiles()
        assert np.allclose(out[j]["q"], q + shift[j], rtol=0, atol=1e-9)
        assert out[j]["cdf_error"] < 1e-9
        assert out[j]["mean_post"] == pytest.approx(out[j]["mean"], rel=1e-12)
