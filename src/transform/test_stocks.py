"""D1 and D2 on hand-made tables, without raw/."""
import math

import numpy as np
import pandas as pd
import pytest

import stocks as S


def tables(rows, locations):
    meas = pd.DataFrame(rows, columns=["site", "location_id", "sample_depth_min",
                                       "sample_depth_max", "SOCc", "BD"])
    loc = pd.DataFrame(locations, columns=["site", "location_id", "X", "Y"])
    return loc, meas


LOCATIONS = [("F", "a", -88.20, 40.10), ("F", "b", -88.2001, 40.1001),
             ("F", "gap", -88.2002, 40.1002), ("F", "short", -88.2003, 40.1003),
             ("F", "none", -88.2004, 40.1004)]
ROWS = [
    # the hand-worked example: 2.0 x 1.30 x 15 + 1.2 x 1.45 x 15 = 39.0 + 26.1 = 65.1
    ("F", "a", 0, 15, 2.0, 1.30), ("F", "a", 15, 30, 1.2, 1.45), ("F", "a", 30, 60, 0.6, 1.50),
    # one 0-30 segment: 1.5 x 1.4 x 30 = 63.0
    ("F", "b", 0, 30, 1.5, 1.4),
    # a gap between 15 and 20 cm, and a location that stops at 27 cm: both left out
    ("F", "gap", 0, 15, 2.0, 1.3), ("F", "gap", 20, 30, 1.0, 1.4),
    ("F", "short", 0, 27, 1.0, 1.4),
]


def test_hand_worked_example():
    loc, meas = tables(ROWS, LOCATIONS)
    t = S.stock_table(loc, meas).set_index("location_id")
    assert t.loc["a", "stock"] == pytest.approx(65.1, abs=1e-12)
    assert t.loc["b", "stock"] == pytest.approx(63.0, abs=1e-12)
    assert t.loc["a", "segments"] == "0-15 15-30"


def test_incomplete_locations_are_left_out_not_filled():
    loc, meas = tables(ROWS, LOCATIONS)
    t = S.stock_table(loc, meas)
    assert list(t["location_id"]) == ["a", "b"]
    assert list(t.columns) == S.COLUMNS


def test_a_segment_across_30_cm_stops_the_run():
    loc, meas = tables([("F", "a", 0, 15, 2.0, 1.3), ("F", "a", 15, 40, 1.0, 1.4)], LOCATIONS[:1])
    with pytest.raises(ValueError, match="cross"):
        S.stock_table(loc, meas)


@pytest.mark.parametrize("lon, epsg", [(-90.19, 32615), (-88.2, 32616), (-96.45, 32614), (-90.0, 32616)])
def test_utm_zone(lon, epsg):
    assert S.utm_epsg(lon) == epsg


def test_coordinates_are_metres_in_the_fields_zone():
    loc, meas = tables(ROWS, LOCATIONS)
    t = S.stock_table(loc, meas).set_index("location_id")
    assert set(t["epsg"]) == {32616}
    d = math.hypot(t.loc["a", "x"] - t.loc["b", "x"], t.loc["a", "y"] - t.loc["b", "y"])
    # 0.0001 degree in each direction at 40.1 N: about 8.5 m east and 11.1 m north
    expected = math.hypot(0.0001 * 111_320 * math.cos(math.radians(40.1)), 0.0001 * 110_574)
    assert d == pytest.approx(expected, rel=0.01)
