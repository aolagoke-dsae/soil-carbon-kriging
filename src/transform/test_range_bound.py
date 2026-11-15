"""W1, W3 and W4 on small cases; the run itself is RUNBOOK, Stage 5, Addendum on M2."""
import numpy as np
import pandas as pd
import pytest

import kriging as K
import range_bound as RB

MODEL = K.Model("a", "constant", "exponential", 10.0, 1000.0)


def test_widened_changes_only_the_upper_bound():
    m = RB.widened(MODEL, 2)
    assert (m.lo, m.hi, m.family, m.mean) == (10.0, 2000.0, "exponential", "constant")
    assert RB.widened(MODEL, 1) is MODEL
    g1, g2 = MODEL.grid(), m.grid()
    assert np.exp(g2.log_ranges[-1]) == pytest.approx(2000.0) and len(g2.log_ranges) > len(g1.log_ranges)
    assert max(np.diff(g2.log_ranges)) <= K.LOG_RANGE_STEP + 1e-12


def test_mass_above_the_original_bound():
    m = RB.widened(MODEL, 3)
    a, _, _ = m.grid().arrays()
    lp = np.where(a > 1000.0 * (1 + 1e-12), 0.0, -np.inf)      # all weight above
    assert RB.mass_above(m, lp, 1000.0) == pytest.approx(1.0)
    assert RB.mass_above(m, np.zeros_like(a), 1000.0) == pytest.approx(np.mean(a > 1000.0 * (1 + 1e-12)))


def test_verdict_needs_both_the_mean_and_the_width_to_stay():
    rows = pd.DataFrame({"site": ["F"] * 3, "factor": [1, 2, 3], "mean": [50.0, 50.1, 50.2], "sd": [1.0, 1.0, 1.0],
                         "q025": [48.0, 48.0, 48.0], "q975": [52.0, 52.1, 52.2]})
    assert bool(RB.verdict(rows)["robust"].iloc[0])
    rows.loc[2, "mean"] = 50.6                                  # more than half a standard deviation
    assert not bool(RB.verdict(rows)["robust"].iloc[0])
    rows.loc[2, "mean"] = 50.2
    rows.loc[2, "q975"] = 53.0                                  # width 4 -> 5: a quarter
    assert not bool(RB.verdict(rows)["robust"].iloc[0])
