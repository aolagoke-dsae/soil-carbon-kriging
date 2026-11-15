"""H2-H5 on small cases; the run itself is RUNBOOK, Stage 5, Addendum on M3."""
import numpy as np
import pandas as pd
import pytest

import halves as H
import kriging as K
import reml as M


def test_splits_are_halves_and_repeatable():
    s = H.splits(11, count=5, seed=3)
    assert all(len(h) == 5 and len(set(h)) == 5 and h.max() < 11 for h in s)
    assert all((a == b).all() for a, b in zip(s, H.splits(11, count=5, seed=3)))
    assert not all((a == b).all() for a, b in zip(s, H.splits(11, count=5, seed=4)))


def test_point_set_mean_equals_the_bordered_system():
    """At one grid point, with a constant mean: the predictor of B's mean and
    its variance (nugget included) from the bordered kriging system, solved
    directly."""
    rng = np.random.default_rng(5)
    xyA, xyB = rng.uniform(0, 100, (12, 2)), rng.uniform(0, 100, (7, 2))
    zA = rng.normal(50, 3, 12)
    model = K.Model("a", "constant", "exponential", 10.0, 200.0)
    a, t, _ = model.grid().arrays()
    g = int(np.flatnonzero(t > 0.2)[0])
    lp = np.full(model.grid().size, -np.inf)
    lp[g] = 0.0
    res = H.point_set_mean(model, xyA, zA, None, xyB, None, (), lp)
    assert res["g"] == [g]
    a, tau = a[g], t[g]
    R = M.correlation("exponential", K.distances(xyA, xyA), a, None)
    C = (1 - tau) * R + tau * np.eye(12)
    k = (1 - tau) * M.correlation("exponential", K.distances(xyB, xyA), a, None).mean(axis=0)
    big = np.block([[C, np.ones((12, 1))], [np.ones((1, 12)), np.zeros((1, 1))]])
    sol = np.linalg.solve(big, np.append(k, 1.0))
    lam, mu = sol[:12], sol[12]
    rbb = M.correlation("exponential", K.distances(xyB, xyB), a, None).mean()
    v = (1 - tau) * rbb + tau / 7 - lam @ k - mu
    assert res["pred"][0] == pytest.approx(lam @ zA, rel=1e-10)
    assert res["v"][0] == pytest.approx(v, rel=1e-8)


def test_the_sample_means_predictive():
    z = np.array([1.0, 2.0, 3.0, 4.0])
    loc, scale, df = H.sample_mean_predictive(z, 6)
    assert (loc, df) == (2.5, 3.0) and scale == pytest.approx(z.std(ddof=1) * np.sqrt(1 / 4 + 1 / 6))


def test_verdict_needs_coverage_and_crps():
    rows = pd.DataFrame({"site": ["A"] * 4, "k_in95": [1, 1, 1, 0], "s_in95": [1] * 4, "k_in80": [1] * 4,
                         "s_in80": [1] * 4, "k_crps": [1.0] * 4, "s_crps": [1.1] * 4, "k_width95": [2.0] * 4,
                         "s_width95": [3.0] * 4})
    per = H.summarise(rows)
    assert per.loc["pooled", "k_in95"] == 0.75 and not H.verdict(per)[0]
    rows["k_in95"] = 1
    assert H.verdict(H.summarise(rows))[0]
