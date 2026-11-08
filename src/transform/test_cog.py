"""Reading stored parts of a Cloud Optimized GeoTIFF: reads outside what is
stored are refused, the grid comes out as GDAL reads it, and a window
decoded block by block equals the tile's own values."""
import io
import math

import numpy as np
import pytest
import tifffile

import cog
import patches as P


def test_reads_are_served_only_from_stored_ranges():
    """Prevents: a gap in the stored bytes being read as zeros."""
    f = cog.RangeFile({0: b"abcd", 4: b"efgh", 100: b"xyz"}, 200)
    f.seek(2)
    assert f.read(4) == b"cdef"                      # across two adjoining ranges
    f.seek(100)
    assert f.read(3) == b"xyz"
    f.seek(6)
    with pytest.raises(cog.OutsideStoredRanges):
        f.read(4)                                    # 8 and 9 are not stored
    f.seek(150)
    with pytest.raises(cog.OutsideStoredRanges):
        f.read(1)
    assert f.seek(0, io.SEEK_END) == 200


def test_a_short_range_inside_a_long_one_does_not_hide_it():
    f = cog.RangeFile({0: bytes(range(100)), 10: b"\x0a\x0b"}, 100)
    f.seek(50)
    assert f.read(2) == bytes([50, 51])


def test_the_grid_is_read_as_gdal_reads_it(synthetic_tile):
    """Prevents: a half-cell shift from a corner or centre tiepoint."""
    data, transform, z, _ = synthetic_tile
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        geo = cog.geometry(tif.pages[0])
    assert geo.crs == "EPSG:4269"
    assert geo.west == pytest.approx(transform.c, abs=1e-12)
    assert geo.north == pytest.approx(transform.f, abs=1e-12)
    assert geo.dx == pytest.approx(transform.a, rel=1e-12)
    assert geo.dy == pytest.approx(-transform.e, rel=1e-12)
    assert (geo.width, geo.height, geo.nodata) == (600, 600, -999999.0)


def test_a_window_decodes_from_the_header_and_its_blocks_alone(synthetic_tile):
    data, _, z, loc = synthetic_tile
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        first = cog.first_data_offset(tif)
        page0 = tif.pages[0]
        offsets, counts = list(page0.dataoffsets), list(page0.databytecounts)
    grid = P.patch_grids(loc)["T1"]
    ranges = {0: data[:first]}
    with tifffile.TiffFile(cog.RangeFile(ranges, len(data))) as tif:
        geo = cog.geometry(tif.pages[0])
        window = cog.window_for(grid, geo)
        for a, b in cog.block_ranges(tif.pages[0], cog.blocks_for(geo, window)):
            ranges[a] = data[a:b + 1]
    f = cog.RangeFile(ranges, len(data))
    with tifffile.TiffFile(f) as tif:
        got = cog.decode_window(tif.pages[0], geo, f, window)
    r0, r1, c0, c1 = window
    assert np.array_equal(got, z[r0:r1 + 1, c0:c1 + 1].astype(np.float64))
    stored = sum(len(v) for v in ranges.values())
    assert stored < 0.6 * len(data)                  # the blocks, not the tile


def test_the_window_holds_every_patch_cell_centre_with_cells_to_spare(synthetic_tile):
    data, _, _, loc = synthetic_tile
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        geo = cog.geometry(tif.pages[0])
    grid = P.patch_grids(loc)["T1"]
    r0, r1, c0, c1 = cog.window_for(grid, geo)
    x, y = grid.centres()
    X, Y = np.meshgrid(x, y)
    lon, lat = P.transformer(grid.epsg, geo.crs).transform(X.ravel(), Y.ravel())
    c, r = geo.frac_col(lon), geo.frac_row(lat)
    assert math.floor(c.min()) - c0 >= cog.MARGIN and c1 - (math.floor(c.max()) + 1) >= cog.MARGIN
    assert math.floor(r.min()) - r0 >= cog.MARGIN and r1 - (math.floor(r.max()) + 1) >= cog.MARGIN


def test_adjoining_blocks_are_merged_into_one_request():
    class Page:
        dataoffsets = (100, 200, 300, 900)
        databytecounts = (100, 100, 50, 10)
    assert cog.block_ranges(Page, [0, 1, 2, 3]) == [(100, 349), (900, 909)]
