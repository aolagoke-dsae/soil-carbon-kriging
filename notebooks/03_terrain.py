# %% [markdown]
# # 03 · Terrain covariates (Checkpoint 3)
#
# Exploration only, as the portfolio standard requires: the rules live in
# `src/checks/checkpoint3.py` and run without this notebook
# (`python -m src.checks.checkpoint3`). This draws each field's DEM,
# hillshaded, with its locations, and then, field by field, the grids of the
# eleven covariates (D3–D6), with the cells whose upslope area reaches the
# patch's edge.
#
# Needs `curated/terrain/` and `curated/potash_terrain.parquet` (RUNBOOK,
# Stage 3: `python -m src.transform.terrain`) and `requirements-figures.txt`.
# The figures are written to `notebooks/scratch/`, which git ignores.

# %%
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
if not (ROOT / "src").is_dir():         # VS Code's interactive window starts in notebooks/
    ROOT = ROOT.parent
os.chdir(ROOT)                          # curated/ is relative to the repository root
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap
from scipy.spatial import ConvexHull

from src.transform import terrain as T

FIGS = ROOT / "notebooks" / "scratch"
FIGS.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"     # text, secondary text, gridlines
BLUE = "#2a78d6"                                       # categorical slot 1: the locations
# One-hue sequential ramps, light to dark with the value: the reference blue
# ramp (steps 100-700), and orange and aqua, the next categorical hues, at
# the same OKLab lightness steps. The diverging map sets the blue arm against
# red at matched lightness, around a neutral grey.
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGES = ["#fbd7ca", "#f5c4b2", "#f2b098", "#eb9c7f", "#e68764", "#df7249", "#d95923",
           "#c94908", "#b44005", "#9f3600", "#8a2e00", "#752600", "#611e00"]
AQUAS = ["#cce8d9", "#b4dbc6", "#9ad0b4", "#80c3a2", "#62b890", "#43ac7f", "#009f6d",
         "#008f62", "#007f56", "#006f4b", "#006040", "#005136", "#00432b"]
DIVERGING = ["#0d366b", "#1c5cab", "#2a78d6", "#6da7ec", "#9ec5f4", "#f0efec",
             "#f8aaa3", "#ee7e77", "#d2383a", "#a72528", "#681014"]
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 8, "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7,
})

table = pd.read_parquet(T.OUT)
SITES = sorted(table["site"].unique())


def load(site):
    with np.load(T.OUT_GRIDS / f"{site}.npz") as z:
        return {k: z[k] for k in z.files}


def extent(g):
    """The patch in km from its south-west corner: left, right, bottom, top."""
    ny, nx = g["elevation"].shape
    return 0.0, nx * float(g["cell"]) / 1000, 0.0, ny * float(g["cell"]) / 1000


def locations_km(site, g):
    t = table[table["site"] == site]
    y0 = float(g["y1"]) - g["elevation"].shape[0] * float(g["cell"])
    return (t["x"] - float(g["x0"])) / 1000, (t["y"] - y0) / 1000


def seam(ax, g):
    """Where a field on two tiles changes from one version to the other."""
    if len(g["versions"]) < 2:
        return
    ny, nx = g["version_used"].shape
    cell = float(g["cell"]) / 1000
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell                   # from the south edge up
    ax.contour(xs, ys, g["version_used"][::-1], levels=[0.5], colors=[INK], linewidths=0.9,
               linestyles="--")


# %% [markdown]
# ## Each field's DEM, hillshaded, with its locations
#
# The patch in its Stage 2 UTM zone, reaching 1 km beyond the outermost
# locations (D2). Light from the north-west, 45° above the horizon, and the
# relief exaggerated ten times, since the fields rise only a few metres. In
# IL-RS the dashed line is where the newer tile, n41w089 of September 2026,
# hands over to n40w089 of January 2024, about six native cells south of
# 40° N.

# %%
LIGHT = LightSource(azdeg=315, altdeg=45)
fig, axes = plt.subplots(2, 4, figsize=(11, 6.3))
for ax, site in zip(axes.ravel(), SITES):
    g = load(site)
    z = g["elevation"].astype(float)
    cell = float(g["cell"])
    ax.imshow(LIGHT.hillshade(z, vert_exag=10, dx=cell, dy=cell), cmap="gray", vmin=0, vmax=1,
              extent=extent(g), interpolation="nearest")
    seam(ax, g)
    x, y = locations_km(site, g)
    ax.scatter(x, y, s=7, color=BLUE, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_title(f"{site} · patch {np.nanmin(z):.1f}–{np.nanmax(z):.1f} m · {len(x)} locations")
    ax.set_aspect("equal")
for ax in axes[-1]:
    ax.set_xlabel("km east")
for ax in axes[:, 0]:
    ax.set_ylabel("km north")
fig.suptitle("The patches, hillshaded (relief × 10), with the sampled locations", x=0.01, ha="left",
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIGS / "03_hillshade.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Each field's covariates
#
# One figure per field: the eleven covariates of Stage 3's table on the
# patch, the outline around its locations in ink. The sequential maps run light to dark with
# the value (elevation blue, slope orange, TWI aqua); curvature and TPI
# diverge about zero, red where the ground is convex or stands above its
# surroundings, blue where it is concave or lies below them. The colours
# span the patch's 2nd to 98th percentile (symmetric about zero where they
# diverge), so that a few extreme cells, a ditch or an embankment, do not
# wash out the rest. The last panel marks, dark, the cells whose upslope
# area reaches the patch's edge, where the TWI is a lower bound (D6).

# %%
PANELS = [  # (grid, title, kind, ramp, scale)
    ("elevation", "elevation (m)", "seq", BLUES, 1.0),
    ("slope_30", "slope, 30 m (°)", "seq", ORANGES, 1.0),
    ("slope_90", "slope, 90 m (°)", "seq", ORANGES, 1.0),
    ("slope_270", "slope, 270 m (°)", "seq", ORANGES, 1.0),
    ("twi", "TWI", "seq", AQUAS, 1.0),
    ("curvature_30", "mean curvature, 30 m (1/km)", "div", DIVERGING, 1000.0),
    ("curvature_90", "mean curvature, 90 m (1/km)", "div", DIVERGING, 1000.0),
    ("curvature_270", "mean curvature, 270 m (1/km)", "div", DIVERGING, 1000.0),
    ("from_edge", "upslope area reaches the edge (dark)", "mask", None, 1.0),
    ("tpi_50", "TPI, 50 m (m)", "div", DIVERGING, 1.0),
    ("tpi_150", "TPI, 150 m (m)", "div", DIVERGING, 1.0),
    ("tpi_450", "TPI, 450 m (m)", "div", DIVERGING, 1.0),
]
EDGE = ListedColormap(["#f0efec", MUTED])


def covariate_panel(fig, ax, g, name, title, kind, ramp, scale):
    v = g[name].astype(float) * scale
    if kind == "mask":
        im = ax.imshow((v > 0).astype(float), cmap=EDGE, vmin=0, vmax=1, extent=extent(g),
                       interpolation="nearest")
    else:
        if kind == "seq":
            lo, hi = np.nanpercentile(v, [2, 98]) if name != "elevation" else (np.nanmin(v), np.nanmax(v))
        else:
            hi = np.nanpercentile(np.abs(v), 98)
            lo = -hi
        im = ax.imshow(v, cmap=LinearSegmentedColormap.from_list(name, ramp), vmin=lo, vmax=hi,
                       extent=extent(g), interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors=MUTED)
        cb.outline.set_visible(False)
    ax.set_title(title)
    ax.set_aspect("equal")


for site in SITES:
    g = load(site)
    pts = np.c_[locations_km(site, g)]
    hull = pts[np.r_[ConvexHull(pts).vertices, ConvexHull(pts).vertices[:1]]]
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.4))
    for ax, panel in zip(axes.ravel(), PANELS):
        covariate_panel(fig, ax, g, *panel)
        ax.plot(hull[:, 0], hull[:, 1], color=INK, lw=0.9, zorder=3)   # around the locations
        seam(ax, g)
    for ax in axes[-1]:
        ax.set_xlabel("km east")
    for ax in axes[:, 0]:
        ax.set_ylabel("km north")
    tiles = ", ".join(Path(str(u)).stem.replace("USGS_13_", "") for u in g["versions"])
    fig.suptitle(f"{site}: the covariates on the patch (tiles {tiles})", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / f"03_covariates_{site}.png", dpi=150)
    plt.close(fig)
print(f"figures in {FIGS}:", sorted(p.name for p in FIGS.glob("03_*.png")))

# %% [markdown]
# ## The resampling's trace in the 30 m curvature
#
# Bilinear interpolation smooths least where a patch cell's centre falls on
# a native cell's centre and most halfway between them. Of a white noise's
# variance, the weights keep ((1 - u)² + u²)((1 - v)² + v²), u and v the
# centre's fractional row and column on the native grid: from 1 down to
# 0.25. The maps of the 30 m curvature show faint striping, so this compares
# each curvature's magnitude with that share, cell by cell: their Spearman
# correlation, and the median magnitude where the share is above 0.8 over
# the median where it is below 0.35. Needs `raw/` for the tiles' grids.

# %%
import tifffile
from scipy import stats

from src.transform import cog
from src.transform import patches as P

tiles = T.stored_tiles()
rows = []
for site in SITES:
    g = load(site)
    ny, nx = g["elevation"].shape
    grid = P.Grid(site, int(g["epsg"]), float(g["x0"]), float(g["y1"]), nx, ny)
    keep = np.full((ny, nx), np.nan)
    for i, url in enumerate(g["versions"]):
        with tifffile.TiffFile(tiles[cog.tile_version(str(url))[0]].file()) as tif:
            geo = cog.geometry(tif.pages[0])
        lon, lat = T.patch_lonlat(grid, geo.crs)
        u, v = geo.frac_row(lat) % 1.0, geo.frac_col(lon) % 1.0
        share = ((1 - u) ** 2 + u ** 2) * ((1 - v) ** 2 + v ** 2)
        keep[g["version_used"] == i] = share[g["version_used"] == i]
    row = {"field": site}
    for n in (30, 90, 270):
        c = np.abs(g[f"curvature_{n}"].astype(float))
        ok = np.isfinite(c) & np.isfinite(keep)
        row[f"rho {n} m"] = stats.spearmanr(c[ok], keep[ok]).statistic
    c = np.abs(g["curvature_30"].astype(float))
    row["median ratio, 30 m"] = np.nanmedian(c[keep > 0.8]) / np.nanmedian(c[keep < 0.35])
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))
