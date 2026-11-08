"""The patches of D2: in the field's UTM zone, on multiples of 10 m, at
least 1,000 m beyond every location, and the blocks they touch."""
import numpy as np
import pandas as pd

import patches as P


def test_a_patch_is_aligned_and_reaches_a_kilometre_beyond_every_location():
    x = np.array([400123.4, 400456.7, 400300.0])
    y = np.array([4420011.1, 4420377.7, 4420200.0])
    g = P.patch_grid("A", 32616, x, y)
    assert g.x0 % 10 == 0 and g.y1 % 10 == 0
    assert (x - g.x0).min() >= 1000 and (g.x1 - x).min() >= 1000
    assert (y - g.y0).min() >= 1000 and (g.y1 - y).min() >= 1000
    assert (x - g.x0).min() < 1010 and (g.y1 - y).min() < 1010


def test_cells_and_fractions_agree():
    g = P.Grid("A", 32616, 1000.0, 5000.0, 30, 20)
    r, c = g.cell_of([1005.0, 1299.9], [4995.0, 4800.1])
    assert list(r) == [0, 19] and list(c) == [0, 29]
    fr, fc = g.fractional([1005.0], [4995.0])
    assert fr[0] == 0 and fc[0] == 0                 # a cell centre is a whole number


def test_blocks_are_named_after_their_north_west_corner():
    assert P.block_names((-88.31, 39.99, -88.27, 40.02)) == ["n40w089", "n41w089"]
    assert P.block_names((-90.20, 39.04, -90.17, 39.07)) == ["n40w091"]
    assert P.block_names((-96.47, 41.14, -96.44, 41.16)) == ["n42w097"]


def test_one_patch_per_field_from_the_stock_table_columns():
    loc = pd.DataFrame({"site": ["A", "A", "B"], "location_id": ["1", "2", "1"],
                        "lon": [0, 0, 0], "lat": [0, 0, 0], "epsg": [32616, 32616, 32615],
                        "x": [400000.0, 400100.0, 700000.0], "y": [4.42e6, 4.4201e6, 4.3e6]})
    grids = P.patch_grids(loc)
    assert sorted(grids) == ["A", "B"] and grids["B"].epsg == 32615
