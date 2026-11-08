"""Stage 4's Bayesian kriging against closed forms, direct algebra and Stage 2's REML."""
import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

import anisotropy as A
import kriging as K
import reml as M
import scores as SC


def data(n=40, seed=3, tau=0.3, eff_range=90.0, family="exponential"):
    rng = np.random.default_rng(seed)
    xy, z = M.simulate(rng, n, (300.0, 200.0), family, 25.0, eff_range, tau, mean=60.0)
    cov = rng.normal(size=(n, len(K.COVARIATES))) + xy[:, :1] / 100.0
    return xy, z, cov


def one_point(family, a, tau, nu=None):
    return K.Grid(family, (math.log(a),), (tau,), (nu,))


def bordered(C, X, c0, x0, z):
    """The kriging system with its Lagrange multipliers, solved directly."""
    n, p = X.shape
    Abig = np.block([[C, X], [X.T, np.zeros((p, p))]])
    sol = np.linalg.solve(Abig, np.concatenate([c0, x0]))
    lam, mu = sol[:n], sol[n:]
    return lam @ z, 1.0 - lam @ c0 - mu @ x0, lam


@pytest.mark.parametrize("mean", ["constant", "quadratic", "drift"])
@pytest.mark.parametrize("angle", [None, 70.0])
def test_universal_kriging_equals_the_bordered_system(mean, angle):
    xy, z, cov = data()
    train, test = np.arange(32), np.arange(32, 40)
    drift = (0, 4, 10)
    X, X0 = K.design(mean, xy[train], xy[test], cov[train], cov[test], drift)
    a, tau, ratio = 110.0, 0.25, 0.4
    post = K.evaluate(one_point("exponential", a, tau), xy[train], z[train], X, xy[test], X0,
                      angle, None if angle is None else ratio)
    d = K.distances(xy[train], xy[train], angle, None if angle is None else ratio)
    C = (1 - tau) * M.correlation("exponential", d, a)
    np.fill_diagonal(C, 1.0)
    d0 = K.distances(xy[test], xy[train], angle, None if angle is None else ratio)
    c0 = (1 - tau) * M.correlation("exponential", d0, a)
    for j in range(len(test)):
        pred, var, lam = bordered(C, X, c0[j], X0[j], z[train])
        assert post.pred[0, j] == pytest.approx(pred, rel=1e-10)
        assert post.var1[0, j] == pytest.approx(var, rel=1e-10)
        assert np.allclose(X.T @ lam, X0[j], atol=1e-10)


def test_exact_interpolation_without_a_nugget():
    xy, z, _ = data(n=25)
    X, X0 = K.design("constant", xy, xy[[3, 7]])
    post = K.evaluate(one_point("spherical", 150.0, 0.0), xy, z, X, xy[[3, 7]], X0)
    assert post.pred[0] == pytest.approx(z[[3, 7]], rel=1e-9)
    assert np.all(post.var1[0] <= 1e-9)


def test_the_variance_with_a_single_neighbour():
    xy = np.array([[0.0, 0.0]])
    x0 = np.array([[30.0, 40.0]])                    # 50 m away
    a, tau = 120.0, 0.2
    X, X0 = K.design("constant", xy, x0)
    post = K.evaluate(one_point("exponential", a, tau), xy, np.array([7.0]), X, x0, X0)
    r = float(M.correlation("exponential", np.array([50.0]), a)[0])
    assert post.pred[0, 0] == pytest.approx(7.0)
    assert post.var1[0, 0] == pytest.approx(2.0 * (1.0 - (1.0 - tau) * r), rel=1e-12)


def test_one_grid_point_gives_the_student_t():
    xy, z, _ = data()
    X, X0 = K.design("quadratic", xy[:35], xy[35:])
    post = K.evaluate(one_point("exponential", 80.0, 0.4), xy[:35], z[:35], X, xy[35:], X0)
    mix = K.predictive(post, 2)
    loc = post.pred[0, 2]
    scale = math.sqrt(post.s2[0] / (35 - 6) * post.var1[0, 2])
    assert np.allclose(mix.quantiles(), stats.t.ppf(SC.LEVELS, 29, loc, scale), rtol=1e-10, atol=0)
    assert mix.crps(z[37]) == pytest.approx(SC.crps_t(z[37], loc, scale, 29.0), rel=1e-9)
    nd = K.plug_in(post, 2)
    assert nd.mean() == loc and nd.scale == pytest.approx(scale)


@pytest.mark.parametrize("mean, family, nu, angle", [("constant", "spherical", None, None),
                                                     ("quadratic", "exponential", None, None),
                                                     ("constant", "matern", 0.6, 120.0)])
def test_log_posterior_is_reml_plus_one_constant(mean, family, nu, angle):
    xy, z, _ = data(n=45)
    field = M.Field.of(xy, z, mean)
    ratio = 0.5 if angle is not None else None
    if angle is not None:
        i, j = np.triu_indices(len(z), 1)
        field = dataclasses.replace(field, d=A.rotated_distances(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1],
                                                                 angle, ratio))
    grid = K.Grid(family, tuple(np.log([30.0, 90.0, 400.0])), (1e-6, 0.3, 0.8), (nu,))
    post = K.evaluate(grid, xy, z, M.design(xy, mean), angle=angle, ratio=ratio)
    a, t, _ = grid.arrays()
    reml = np.array([M.reml_profile(field, family, ai, ti, nu)[0] for ai, ti in zip(a, t)])
    diff = post.logpost - reml
    assert np.ptp(diff) < 1e-8


def test_log_posterior_against_explicit_inverses():
    xy, z, cov = data(n=30)
    X = K.design("drift", xy, xy[:1], cov, cov[:1], (1, 9))[0]
    grid = K.Grid("exponential", tuple(np.log([20.0, 150.0])), (1e-6, 0.5, 1.0), (None,))
    post = K.evaluate(grid, xy, z, X)
    n, p = X.shape
    for g, (a, t) in enumerate(zip(*grid.arrays()[:2])):
        C = (1 - t) * M.correlation("exponential", K.distances(xy, xy), a)
        np.fill_diagonal(C, 1.0)
        Ci = np.linalg.inv(C)
        XCX = X.T @ Ci @ X
        b = np.linalg.solve(XCX, X.T @ Ci @ z)
        S2 = (z - X @ b) @ Ci @ (z - X @ b)
        ref = -0.5 * np.linalg.slogdet(C)[1] - 0.5 * np.linalg.slogdet(XCX)[1] - 0.5 * (n - p) * math.log(S2)
        assert post.logpost[g] == pytest.approx(ref, rel=1e-10)
        assert post.s2[g] == pytest.approx(S2, rel=1e-10)
    assert post.weights().sum() == pytest.approx(1.0, abs=1e-14)


def test_grid_bounds_steps_and_refinement():
    g = K.make_grid("matern", 10.0, 2400.0)
    steps = np.diff(g.log_ranges)
    assert g.log_ranges[0] == pytest.approx(math.log(10.0)) and g.log_ranges[-1] == pytest.approx(math.log(2400.0))
    assert np.all(steps <= 0.1 + 1e-12) and np.ptp(steps) < 1e-12
    assert g.taus[0] == 1e-6 and g.taus[1] == 0.05 and g.taus[-1] == 1.0 and len(g.taus) == 21
    assert len(g.nus) == 7 and g.nus[0] == pytest.approx(0.2) and g.nus[-1] == pytest.approx(3.0)
    r = K.make_grid("matern", 10.0, 2400.0, refine=2)
    assert len(r.log_ranges) == 2 * len(g.log_ranges) - 1 and len(r.taus) == 41 and len(r.nus) == 13
    assert set(np.round(g.log_ranges, 12)) <= set(np.round(r.log_ranges, 12))
    assert K.make_grid("exponential", 10.0, 2400.0).nus == (None,)


def test_drift_candidates():
    c = K.drift_candidates()
    assert len(c) == 256 and c[0] == () and len(set(c)) == 256
    groups = [set(range(0, 1)), set(range(1, 4)), set(range(4, 7)), set(range(7, 10)), set(range(10, 11))]
    for cand in c:
        assert all(len(set(cand) & gr) <= 1 for gr in groups)
    assert K.COVARIATES[1:4] == ("slope_30", "slope_90", "slope_270")


def test_ml_table_against_a_direct_maximum_likelihood():
    xy, z, cov = data(n=35)
    z = z + 3.0 * (cov[:, 4] - cov[:, 4].mean())
    grid = K.Grid("exponential", tuple(np.log([25.0, 70.0, 200.0])), (1e-6, 0.3, 0.7), (None,))
    cands = [(), (4,), (1, 4), (0, 6, 10)]
    table = K.ml_table(grid, xy, z, cov, cands)
    covs = K.standardise(cov)[0]
    n = len(z)
    for k, cand in enumerate(cands):
        X = np.column_stack([np.ones(n)] + [covs[:, i] for i in cand])
        best = -np.inf
        for a, t in zip(*grid.arrays()[:2]):
            C = (1 - t) * M.correlation("exponential", K.distances(xy, xy), a)
            np.fill_diagonal(C, 1.0)
            Ci = np.linalg.inv(C)
            b = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ z)
            e = z - X @ b
            s2 = e @ Ci @ e / n
            ll = -0.5 * (n * math.log(2 * math.pi * s2) + np.linalg.slogdet(C)[1] + n)
            best = max(best, ll)
        assert table.loc[k, "loglik"] == pytest.approx(best, rel=1e-10)
    assert K.choose(table) == 1
    assert table.loc[1, "aic"] == pytest.approx(-2 * table.loc[1, "loglik"] + 2 * (2 + 3))


def test_choose_breaks_ties_to_fewer_covariates():
    t = K.pd.DataFrame({"candidate": [0, 1, 2], "p": [3, 2, 2], "aic": [10.0, 10.0, 10.0]})
    assert K.choose(t) == 1


def toy_fields(seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    for s, n in (("F1", 34), ("F2", 41), ("F3", 30)):
        xy, z = M.simulate(rng, n, (250.0, 180.0), "exponential", 30.0, 70.0, 0.3, mean=50.0)
        fd = K.FieldData(s, np.array([f"{s}-{i:03d}" for i in range(n)]), xy, z,
                         rng.normal(size=(n, len(K.COVARIATES))))
        for r in range(2):
            perm = np.random.default_rng(100 * r + n).permutation(n)
            f = np.empty(n, dtype=int)
            f[perm] = np.arange(n) % 10
            fd.folds[r] = f
        out[s] = fd
    return out


def test_held_out_stocks_do_not_move_any_prediction():
    """C2 in small: a shared model and a drift model, the held-out fold raised by 1,000 in every field."""
    fields = toy_fields()
    shared = K.Model("b", "quadratic", "exponential", 10.0, 600.0, shared=True)
    drift = K.Model("c", "drift", "exponential", 10.0, 600.0)
    cache = K.Cache()
    before = {}
    for m in (shared, drift):
        others = K.shared_others(fields, "F2", m, 1, 4, cache) if m.shared else None
        train, test = fields["F2"].split(1, 4)
        before[m.name] = K.fold_result(fields["F2"], m, train, test, others)
    raised = {s: dataclasses.replace(fd, z=fd.z + 1000.0 * (fd.folds[1] == 4)) for s, fd in fields.items()}
    for m in (shared, drift):
        others = K.shared_others(raised, "F2", m, 1, 4, K.Cache()) if m.shared else None
        train, test = raised["F2"].split(1, 4)
        after = K.fold_result(raised["F2"], m, train, test, others)
        assert [b["q"] for b in after["bayes"]] == [b["q"] for b in before[m.name]["bayes"]]
        assert [p["mean"] for p in after["plugin"]] == [p["mean"] for p in before[m.name]["plugin"]]


def test_shared_posterior_adds_the_other_fields_training_likelihoods():
    fields = toy_fields()
    m = K.Model("b", "quadratic", "exponential", 10.0, 600.0, shared=True)
    total = K.shared_others(fields, "F1", m, 0, 2, K.Cache())
    ref = 0.0
    for s in ("F2", "F3"):
        tr = fields[s].split(0, 2)[0]
        xy, z = fields[s].xy[tr], fields[s].z[tr]
        ref = ref + K.evaluate(m.grid(), xy, z, M.design(xy, "quadratic", xy.mean(axis=0))).logpost
    assert np.array_equal(total, ref)


def test_boxcox_at_one_reproduces_the_original_scale():
    fields = toy_fields()
    fd = fields["F1"]
    train, test = fd.split(0, 3)
    a = K.Model("a", "constant", "exponential", 10.0, 600.0)
    bc = dataclasses.replace(a, name="boxcox_a", lam=1.0)
    ra, rb = K.fold_result(fd, a, train, test), K.fold_result(fd, bc, train, test)
    for x, y in zip(ra["bayes"], rb["bayes"]):
        assert np.allclose(x["q"], y["q"], rtol=1e-11)
        assert y["mean"] == pytest.approx(x["mean"], rel=1e-9)
        assert y["crps"] == pytest.approx(x["crps"], rel=1e-7)
        assert y["cdf_error"] < 1e-9 and x["cdf_error"] < 1e-9
        assert x["mean_post"] == pytest.approx(x["mean"], rel=1e-12)


def test_weighted_quantiles_and_parameter_summary():
    v, w = np.array([3.0, 1.0, 2.0]), np.array([0.2, 0.5, 0.3])
    assert list(K.weighted_quantiles(v, w, [0.05, 0.5, 0.6, 0.95])) == [1.0, 1.0, 2.0, 3.0]


def test_code_hash_ignores_docstrings_and_comments(tmp_path):
    a, b, c = tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"
    a.write_text('def f(x):\n    """One."""\n    return x + 1  # a comment\n')
    b.write_text('def f(x):\n    """Two, longer."""\n    # another\n    return x + 1\n')
    c.write_text('def f(x):\n    """One."""\n    return x + 2\n')
    assert K.code_hash(a) == K.code_hash(b) != K.code_hash(c)


def test_one_factorisation_for_several_value_vectors():
    xy, z, _ = data(n=30)
    X, X0 = K.design("constant", xy[:24], xy[24:])
    grid = K.Grid("spherical", tuple(np.log([40.0, 160.0])), (1e-6, 0.4), (None,))
    zs = [z[:24], np.log(z[:24]), 2.0 * z[:24] + 1.0]
    many = K.evaluate_many(grid, xy[:24], zs, X, xy[24:], X0)
    for zi, post in zip(zs, many):
        one = K.evaluate(grid, xy[:24], zi, X, xy[24:], X0)
        assert np.allclose(post.logpost, one.logpost, rtol=1e-13, atol=0)
        assert np.allclose(post.pred, one.pred, rtol=1e-13, atol=0)
        assert np.allclose(post.var1, one.var1, rtol=1e-13, atol=0)


def test_models_are_grouped_with_their_transforms_only():
    a = K.Model("a", "constant", "matern", 10.0, 2400.0, 150.0, 0.5)
    groups = K.group_models([a, K.Model("b", "quadratic", "spherical", 10.0, 2400.0),
                             K.Model("c", "drift", "exponential", 10.0, 2400.0),
                             dataclasses.replace(a, name="log_a", lam=0.0),
                             dataclasses.replace(a, name="boxcox_a", lam=0.62)])
    assert [[m.name for m in g] for g in groups] == [["a", "log_a", "boxcox_a"], ["b"], ["c"]]
