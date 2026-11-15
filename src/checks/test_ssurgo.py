"""U2-U4 on made-up map units and locations."""
import numpy as np
import pandas as pd
import pytest

import ssurgo as U


def _hz(rows):
    return pd.DataFrame(rows, columns=["mukey", "cokey", "compname", "comppct_r", "majcompflag", "hzdept_r",
                                       "hzdepb_r", "om_l", "om_r", "om_h", "dbthirdbar_r"])


def test_component_is_thickness_weighted_above_30_cm():
    """0-20 cm at 3 % and 20-50 cm at 1 % OM: (3*20 + 1*10) / 30 = 7/3 % OM above 30 cm."""
    h = _hz([["1", "a", "X", 100, "Yes", 0, 20, 2, 3, 4, 1.2],
             ["1", "a", "X", 100, "Yes", 20, 50, 0.5, 1, 1.5, 1.4]])
    c = U.component_values(h).iloc[0]
    assert c["oc_r"] == pytest.approx(7 / 3 * U.OM_TO_OC)
    assert c["oc_l"] < c["oc_r"] < c["oc_h"]
    assert c["stock"] == pytest.approx(U.OM_TO_OC * (3 * 1.2 * 20 + 1 * 1.4 * 10))


def test_mapunit_weights_by_comppct_over_components_with_data():
    comp = pd.DataFrame({"mukey": ["1", "1", "1"], "cokey": ["a", "b", "c"], "comppct_r": [60.0, 30.0, 10.0],
                         "oc_l": [1.0, 2.0, np.nan], "oc_r": [1.0, 2.0, np.nan], "oc_h": [1.0, 2.0, np.nan],
                         "stock": [10.0, np.nan, np.nan]})
    m = U.mapunit_values(comp).iloc[0]
    assert m["oc_r"] == pytest.approx((60 * 1 + 30 * 2) / 90)
    assert m["stock"] == pytest.approx(10.0)


def test_locate_takes_the_containing_polygon():
    polys = pd.DataFrame({"mukey": ["1", "2"], "wkt": ["POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                                                        "POLYGON((1 0, 2 0, 2 1, 1 1, 1 0))"]})
    pts = pd.DataFrame({"lon": [0.5, 1.5, 3.0], "lat": [0.5, 0.5, 0.5]})
    got = U.locate(pts, polys).tolist()
    assert got[:2] == ["1", "2"] and pd.isna(got[2])


def test_observed_soc_is_thickness_weighted():
    meas = pd.DataFrame({"site": ["A", "A", "A"], "location_id": ["1", "1", "1"],
                         "sample_depth_min": [0, 10, 30], "sample_depth_max": [10, 30, 60],
                         "SOCc": [3.0, 1.5, 0.1]})
    assert U.observed_soc(meas).loc[("A", "1")] == pytest.approx((3 * 10 + 1.5 * 20) / 30)


def test_verdict_against_the_other_fields_range():
    out = pd.DataFrame({"site": ["IL-BR", "B", "C"], "ratio": [1.1, 0.9, 1.3]})
    assert U.verdict(out)[0]
    out.loc[0, "ratio"] = 0.5
    ok, text = U.verdict(out)
    assert not ok and "INCONSISTENT" in text
