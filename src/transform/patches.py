"""Stage 3, D2: the patch around each field, and the 1 x 1 degree blocks of
the USGS 1/3 arc-second DEM it lies on.

A patch is in its field's Stage 2 UTM zone, with 10 m cells whose edges lie
on multiples of 10 m, and covers the field's locations and 1,000 m beyond
them on every side. Rows run from north to south. The client uses the
patches to know what to fetch, the transform to know what to compute.
Makes no request.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import Transformer

CELL = 10.0                     # m
BUFFER = 1000.0                 # m beyond every location
LOCATION_COLUMNS = ["site", "location_id", "lon", "lat", "epsg", "x", "y"]


@dataclass(frozen=True)
class Grid:
    """A patch: its UTM zone, its west and north edges and its size in cells."""
    site: str
    epsg: int
    x0: float
    y1: float
    nx: int
    ny: int

    @property
    def x1(self) -> float:
        return self.x0 + self.nx * CELL

    @property
    def y0(self) -> float:
        return self.y1 - self.ny * CELL

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """The x of each column's and the y of each row's cell centres."""
        return (self.x0 + CELL * (np.arange(self.nx) + 0.5),
                self.y1 - CELL * (np.arange(self.ny) + 0.5))

    def cell_of(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Row and column of the cell that contains each (x, y)."""
        return (np.floor((self.y1 - np.asarray(y, float)) / CELL).astype(int),
                np.floor((np.asarray(x, float) - self.x0) / CELL).astype(int))

    def fractional(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Row and column of each (x, y) in units of cells, 0 at the first
        cell's centre: what bilinear interpolation between centres needs."""
        return ((self.y1 - np.asarray(y, float)) / CELL - 0.5,
                (np.asarray(x, float) - self.x0) / CELL - 0.5)


def patch_grid(site: str, epsg: int, x, y) -> Grid:
    """The patch of D2 around a field's locations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x0 = math.floor((x.min() - BUFFER) / CELL) * CELL
    x1 = math.ceil((x.max() + BUFFER) / CELL) * CELL
    y0 = math.floor((y.min() - BUFFER) / CELL) * CELL
    y1 = math.ceil((y.max() + BUFFER) / CELL) * CELL
    return Grid(site, int(epsg), x0, y1, int(round((x1 - x0) / CELL)), int(round((y1 - y0) / CELL)))


def patch_grids(locations: pd.DataFrame) -> dict[str, Grid]:
    """One patch per field, from the stock table's keys and coordinates."""
    out = {}
    for site, g in locations.groupby("site", sort=True):
        epsg = g["epsg"].unique()
        if len(epsg) != 1:
            raise ValueError(f"{site}: locations in more than one UTM zone")
        out[site] = patch_grid(site, int(epsg[0]), g["x"], g["y"])
    return out


def transformer(epsg: int, crs: str) -> Transformer:
    """UTM to the DEM's geographic coordinates, longitude first."""
    return Transformer.from_crs(f"EPSG:{epsg}", crs, always_xy=True)


def edges(grid: Grid, per_side: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Points along the patch's outer edges, corners included."""
    t = np.linspace(0.0, 1.0, per_side + 1)
    xs = np.concatenate([grid.x0 + t * (grid.x1 - grid.x0), np.full_like(t, grid.x1),
                         grid.x1 - t * (grid.x1 - grid.x0), np.full_like(t, grid.x0)])
    ys = np.concatenate([np.full_like(t, grid.y1), grid.y1 - t * (grid.y1 - grid.y0),
                         np.full_like(t, grid.y0), grid.y0 + t * (grid.y1 - grid.y0)])
    return xs, ys


def ring(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """The cell centres on the patch's outermost ring of cells. A UTM
    rectangle is not a rectangle in longitude and latitude, but its extremes
    lie on its boundary, so the ring bounds every cell centre."""
    x, y = grid.centres()
    xs = np.concatenate([x, np.full(grid.ny, x[-1]), x[::-1], np.full(grid.ny, x[0])])
    ys = np.concatenate([np.full(grid.nx, y[0]), y, np.full(grid.nx, y[-1]), y[::-1]])
    return xs, ys


def lonlat_bbox(grid: Grid, crs: str = "EPSG:4269") -> tuple[float, float, float, float]:
    """West, south, east and north of the patch in longitude and latitude."""
    lon, lat = transformer(grid.epsg, crs).transform(*edges(grid))
    return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))


def block_names(bbox: tuple[float, float, float, float]) -> list[str]:
    """The 1 x 1 degree blocks a box touches, named as USGS names its tiles
    after their north-west corner: n40w089 runs from 39 to 40 N, 89 to 88 W.
    Western and northern hemispheres only."""
    lon0, lat0, lon1, lat1 = bbox
    if not (lon1 < 0 and lat0 > 0):
        raise ValueError("block names here cover the north-western quadrant only")
    return sorted(f"n{lat + 1:02d}w{-lon:03d}"
                  for lat in range(math.floor(lat0), math.floor(lat1) + 1)
                  for lon in range(math.floor(lon0), math.floor(lon1) + 1))
