"""Stage 3: reading the stored parts of a Cloud Optimized GeoTIFF.

The client stores a tile's header and some of its internal blocks, each as
the range response it came in. RangeFile presents them to tifffile as one
file of the tile's full size and refuses any read that falls outside what is
stored, so a decode can never fill a gap with zeros (rule E1). geometry()
reads the tile's grid from its GeoTIFF tags, window_for() finds the native
cells a patch needs, and decode_window() decodes just the blocks that hold
them. Makes no request.
"""
from __future__ import annotations

import bisect
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:                        # run as a module from the repository root
    from src.transform import patches as P
except ImportError:         # imported with src/transform on sys.path (pytest)
    import patches as P

MARGIN = 2                  # native cells to spare around a patch (D2)
TILE_URL = re.compile(r"/StagedProducts/Elevation/13/TIFF/historical/"
                      r"(n\d{2}w\d{3})/USGS_13_(n\d{2}w\d{3})_(\d{8})\.tif$")


def tile_version(url: str) -> tuple[str, str]:
    """(block, version date) of a tile's dated URL; ValueError otherwise."""
    m = TILE_URL.search(url)
    if not m or m.group(1) != m.group(2):
        raise ValueError(f"not a dated 1/3 arc-second tile URL: {url}")
    return m.group(1), m.group(3)


def parse_revision(revision: str) -> dict:
    """A range row's revision, bytes=a-b;etag=...;size=N, as a dict with
    first, last, etag and size."""
    parts = dict(p.split("=", 1) for p in revision.split(";") if "=" in p)
    a, b = parts["bytes"].split("-")
    return {"first": int(a), "last": int(b), "etag": parts.get("etag", ""),
            "size": int(parts["size"]) if "size" in parts else None}


class OutsideStoredRanges(OSError):
    """A read asked for bytes the client did not store."""
    def __init__(self, pos: int, n: int):
        super().__init__(f"{n} bytes at offset {pos} are not in the stored ranges")
        self.pos, self.n = pos, n


class RangeFile(io.RawIOBase):
    """A file of `size` bytes of which only `ranges` (offset -> bytes) are
    known. Reads inside them, also across adjoining ranges, are served; any
    other read raises OutsideStoredRanges. Every read is logged."""

    def __init__(self, ranges: dict[int, bytes], size: int):
        super().__init__()
        self._offsets = sorted(ranges)
        self._data = ranges
        self.size = int(size)
        self._pos = 0
        self.reads: list[tuple[int, int]] = []

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = int(offset)
        elif whence == io.SEEK_CUR:
            self._pos += int(offset)
        elif whence == io.SEEK_END:
            self._pos = self.size + int(offset)
        else:
            raise ValueError(f"whence {whence}")
        return self._pos

    def _covering(self, pos: int) -> Optional[int]:
        """The start of a stored range that holds byte pos, the one reaching
        furthest if several do."""
        best = None
        for a in self._offsets[:bisect.bisect_right(self._offsets, pos)]:
            end = a + len(self._data[a])
            if end > pos and (best is None or end > best + len(self._data[best])):
                best = a
        return best

    def _get(self, pos: int, n: int) -> bytes:
        out = bytearray()
        start, want = pos, n
        while n > 0:
            a = self._covering(pos)
            if a is None:
                raise OutsideStoredRanges(start, want)
            b = self._data[a]
            take = min(n, a + len(b) - pos)
            out += b[pos - a:pos - a + take]
            pos += take
            n -= take
        return bytes(out)

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = max(0, min(int(n), self.size - self._pos))
        data = self._get(self._pos, n) if n else b""
        self.reads.append((self._pos, n))
        self._pos += n
        return data

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)

    def readall(self) -> bytes:
        return self.read(-1)


@dataclass(frozen=True)
class Geo:
    """A tile's grid: its geographic CRS, its west and north edges and cell
    size in degrees, its size in cells and blocks, and its nodata value."""
    crs: str
    west: float
    north: float
    dx: float
    dy: float
    width: int
    height: int
    tilewidth: int
    tilelength: int
    nodata: Optional[float]

    @property
    def east(self) -> float:
        return self.west + self.width * self.dx

    @property
    def south(self) -> float:
        return self.north - self.height * self.dy

    def frac_col(self, lon) -> np.ndarray:
        """Column in units of cells, 0 at the first column's centre."""
        return (np.asarray(lon, float) - self.west) / self.dx - 0.5

    def frac_row(self, lat) -> np.ndarray:
        return (self.north - np.asarray(lat, float)) / self.dy - 0.5

    def centre_lon(self, col) -> np.ndarray:
        return self.west + (np.asarray(col, float) + 0.5) * self.dx

    def centre_lat(self, row) -> np.ndarray:
        return self.north - (np.asarray(row, float) + 0.5) * self.dy


def geometry(page) -> Geo:
    """The grid of a tifffile page, from its GeoTIFF tags: tiepoint and pixel
    scale, the raster type (a tiepoint on a cell's corner or its centre) and
    the geographic CRS. Refuses what Stage 3 does not expect: untiled or
    multi-band images, rotated grids, projected coordinates."""
    if not page.is_tiled:
        raise ValueError("the image is not tiled")
    if page.samplesperpixel != 1:
        raise ValueError(f"{page.samplesperpixel} samples per pixel, expected 1")
    tags = page.geotiff_tags
    if not tags:
        raise ValueError("no GeoTIFF tags")
    if "ModelTransformation" in tags:
        raise ValueError("a rotated or sheared grid (ModelTransformation)")
    if int(tags.get("GTModelTypeGeoKey", 2)) != 2:
        raise ValueError("not geographic coordinates")
    sx, sy = (float(v) for v in tags["ModelPixelScale"][:2])
    i, j, _, x, y, _ = (float(v) for v in tags["ModelTiepoint"][:6])
    area = int(tags.get("GTRasterTypeGeoKey", 1)) == 1      # 1 PixelIsArea, 2 PixelIsPoint
    west = x - i * sx - (0.0 if area else 0.5 * sx)
    north = y + j * sy + (0.0 if area else 0.5 * sy)
    code = int(tags.get("GeographicTypeGeoKey", 0))
    if not 4000 <= code < 32767:
        raise ValueError(f"geographic CRS code {code} is not an EPSG code")
    nodata = None
    if "GDAL_NODATA" in page.tags:
        nodata = float(str(page.tags["GDAL_NODATA"].value).strip("\x00 "))
    shape = page.shape
    return Geo(f"EPSG:{code}", west, north, sx, sy, int(shape[-1]), int(shape[-2]),
               int(page.tilewidth), int(page.tilelength), nodata)


def window_for(grid: P.Grid, geo: Geo, margin: int = MARGIN) -> Optional[tuple[int, int, int, int]]:
    """Native rows and columns (first, last, inclusive) that bilinear
    interpolation at every cell centre of the patch needs, with `margin`
    cells to spare, clipped to the tile; None if the patch misses it."""
    lon, lat = P.transformer(grid.epsg, geo.crs).transform(*P.ring(grid))
    c, r = geo.frac_col(lon), geo.frac_row(lat)
    c0, c1 = math.floor(c.min()) - margin, math.floor(c.max()) + 1 + margin
    r0, r1 = math.floor(r.min()) - margin, math.floor(r.max()) + 1 + margin
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, geo.width - 1), min(r1, geo.height - 1)
    if c0 > c1 or r0 > r1:
        return None
    return r0, r1, c0, c1


def blocks_for(geo: Geo, window: tuple[int, int, int, int]) -> list[int]:
    """Indices of the internal blocks that hold a window's cells."""
    r0, r1, c0, c1 = window
    across = math.ceil(geo.width / geo.tilewidth)
    return [tr * across + tc
            for tr in range(r0 // geo.tilelength, r1 // geo.tilelength + 1)
            for tc in range(c0 // geo.tilewidth, c1 // geo.tilewidth + 1)]


def block_ranges(page, indices: list[int]) -> list[tuple[int, int]]:
    """(first byte, last byte) of each block, adjoining blocks merged."""
    spans = sorted((int(page.dataoffsets[i]), int(page.databytecounts[i])) for i in indices)
    merged: list[list[int]] = []
    for off, n in spans:
        if n == 0:
            raise ValueError("an empty block")
        if merged and off == merged[-1][1] + 1:
            merged[-1][1] = off + n - 1
        elif merged and off <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], off + n - 1)
        else:
            merged.append([off, off + n - 1])
    return [(a, b) for a, b in merged]


def first_data_offset(tif) -> int:
    """Where image data begins: in a Cloud Optimized GeoTIFF every
    directory and tag value lies before it."""
    return min(int(min(p.dataoffsets)) for p in tif.pages)


def decode_window(page, geo: Geo, f: RangeFile, window: tuple[int, int, int, int]) -> np.ndarray:
    """The native values of a window, decoded block by block from f."""
    r0, r1, c0, c1 = window
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), np.nan, dtype=np.float64)
    across = math.ceil(geo.width / geo.tilewidth)
    for idx in blocks_for(geo, window):
        f.seek(int(page.dataoffsets[idx]))
        data = f.read(int(page.databytecounts[idx]))
        tile = page.decode(data, idx)[0]
        tile = np.asarray(tile).reshape(tile.shape[-3], tile.shape[-2])
        tr, tc = divmod(idx, across)
        R, C = tr * geo.tilelength, tc * geo.tilewidth
        ra, rb = max(r0, R), min(r1, R + geo.tilelength - 1, geo.height - 1)
        ca, cb = max(c0, C), min(c1, C + geo.tilewidth - 1, geo.width - 1)
        out[ra - r0:rb - r0 + 1, ca - c0:cb - c0 + 1] = tile[ra - R:rb - R + 1, ca - C:cb - C + 1]
    if geo.nodata is not None:
        out[out == geo.nodata] = np.nan
    return out


def sparse_xml(size: int, pieces: list[tuple[int, Path, int]]) -> str:
    """A GDAL /vsisparse/ description of a file of `size` bytes made of
    stored pieces (offset in the file, stored file, length): what lets GDAL
    read the same bytes the transform reads (rule E3)."""
    regions = "".join(
        f"<SubfileRegion><Filename relative=\"0\">{Path(path).resolve().as_posix()}</Filename>"
        f"<DestinationOffset>{off}</DestinationOffset><SourceOffset>0</SourceOffset>"
        f"<RegionLength>{n}</RegionLength></SubfileRegion>"
        for off, path, n in pieces)
    return f"<VSISparseFile><Length>{size}</Length>{regions}</VSISparseFile>"
