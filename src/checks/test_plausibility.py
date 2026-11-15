"""Q1-Q5 on made-up segments: the flag, the per-field table and the verdict."""
import pandas as pd
import pytest

import plausibility as P


def _meas(rows):
    return pd.DataFrame(rows, columns=["site", "location_id", "sample_depth_min", "sample_depth_max", "SOCc", "BD"])


def test_flag_needs_both_loose_and_low_carbon():
    seg = _meas([("A", "1", 0, 10, 1.0, 0.6),     # flagged
                 ("A", "2", 0, 10, 3.0, 0.6),     # loose but rich in carbon
                 ("A", "3", 0, 10, 1.0, 1.2),     # low carbon, ordinary density
                 ("A", "4", 0, 10, 2.0, 0.9)])    # both at the threshold: not flagged
    assert P.flag(P.above(seg)).tolist() == [True, False, False, False]


def test_segments_below_30_cm_are_not_counted():
    seg = _meas([("A", "1", 30, 60, 0.5, 0.5), ("A", "1", 0, 30, 1.0, 1.2)])
    assert len(P.above(seg)) == 1


def test_per_field_and_verdict():
    meas = _meas([("A", "1", 0, 30, 1.0, 0.6), ("A", "2", 0, 30, 1.0, 1.2), ("A", "3", 0, 30, 1.0, 1.2),
                  ("B", "1", 0, 30, 1.0, 1.2)])
    table = pd.DataFrame({"site": ["A", "A", "A", "B"], "location_id": ["1", "2", "3", "1"],
                          "stock": [18.0, 36.0, 36.0, 40.0]})
    out = P.per_field(meas, table).set_index("site")
    assert out.loc["A", "flagged"] == 1 and out.loc["A", "stocks_flagged"] == 1
    assert out.loc["A", "mean_all"] == pytest.approx(30.0)
    assert out.loc["A", "shift"] == pytest.approx(6.0)
    assert out.loc["B", "flagged"] == 0 and out.loc["B", "shift"] == pytest.approx(0.0)
    assert "provisional" in P.verdict(out.reset_index().iloc[0])
    assert "carbon content's" in P.verdict(out.reset_index().iloc[1])
