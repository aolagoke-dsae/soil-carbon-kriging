# %% [markdown]
# # 01 · Data checks (Checkpoint 1)
#
# Exploration only, as the portfolio standard requires: the rules live in
# `src/checks/checkpoint1.py` and run without this notebook
# (`python -m src.checks.checkpoint1`). This shows what they found, and what
# Stage 2 needs to know before any variogram: how many locations each Potash
# field has and how they are laid out, and how the BZE-LW published points are
# spaced, given that each sits up to 4 km from its planned site.
#
# Needs both pulls (RUNBOOK, Stage 1) and `requirements-figures.txt`. The
# figures are written to `notebooks/scratch/`, which git ignores.

# %%
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
if not (ROOT / "src").is_dir():         # VS Code's interactive window starts in notebooks/
    ROOT = ROOT.parent
os.chdir(ROOT)                          # raw/ is relative to the repository root
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

from src.checks import checkpoint1 as C

FIGS = ROOT / "notebooks" / "scratch"
FIGS.mkdir(parents=True, exist_ok=True)
INK, MUTED, MARK = "#1f2328", "#59636e", "#2f6f9f"   # text, secondary text, the one series
SEQUENTIAL = LinearSegmentedColormap.from_list(      # one hue, light to dark, never white
    "greens", plt.cm.Greens(np.linspace(0.3, 1.0, 256)))
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

# %% The rules: the same call as `python -m src.checks.checkpoint1`
rules, diag = C.run()
print(C.report(rules, diag))
assert all(r.passed for r in rules), "Checkpoint 1 failed: fix the data or the rule first"

# %% [markdown]
# ## Potash et al. (2023): locations per field
#
# Webster and Oliver (1992): a variogram from fewer than 50 points is of little
# value, at least 100 are needed, 150 might often be satisfactory and 225 is
# usually reliable (`config/variogram_sample_sizes.csv`). Whether each field
# can carry its own variogram, or the fields must share one, is decided here.

# %%
per_site = diag["potash"].sort_values("locations")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.barh(per_site["site"], per_site["locations"], color=MARK, height=0.6)
for n in (50, 100, 150, 225):
    ax.axvline(n, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.text(n, 1.01, f"{n}", transform=ax.get_xaxis_transform(),
            color=MUTED, fontsize=8, ha="center", va="bottom")
ax.set_xlabel("georeferenced locations")
ax.set_title("Locations per field, against Webster & Oliver (1992)", pad=14)
fig.tight_layout()
fig.savefig(FIGS / "01_potash_locations_per_field.png", dpi=150)
per_site

# %% [markdown]
# ## Potash et al. (2023): the layout of each field
#
# Each field in its own UTM zone (GeoPandas picks it), in metres from its
# south-west corner. Clusters, transects or a grid change which lags a
# variogram can see.

# %%
loc = diag["locations"]
points = gpd.GeoDataFrame(
    loc, geometry=gpd.points_from_xy(loc["X"].astype(float), loc["Y"].astype(float)),
    crs="EPSG:4326")
sites = sorted(points["site"].unique())
fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
for ax, site in zip(axes.flat, sites):
    g = points[points["site"] == site]
    g = g.to_crs(g.estimate_utm_crs())
    ax.scatter(g.geometry.x - g.geometry.x.min(), g.geometry.y - g.geometry.y.min(),
               s=5, color=MARK, linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(f"{site} · n = {len(g)}", fontsize=9)
    ax.tick_params(labelsize=7)
for ax in list(axes.flat)[len(sites):]:
    ax.set_axis_off()
fig.suptitle("Sampling locations per field, metres from the south-west corner",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIGS / "01_potash_field_layouts.png", dpi=150)

# %% [markdown]
# ## BZE-LW: the published points
#
# The stock in 0–30 cm at each published point. The points are not the sites:
# each was placed at random within 4 km of its planned point for data
# protection (`config/sources.csv`), so this map shows the pattern at national
# scale and nothing below a few kilometres.

# %%
t = diag["bze_sites"]
bze = gpd.GeoDataFrame(
    t.drop(columns=["_geom_type", "_gx", "_gy"]),
    geometry=gpd.points_from_xy(t["_gx"].astype(float), t["_gy"].astype(float)),
    crs="EPSG:25832")
kv = pd.to_numeric(bze["kv_0_30"], errors="coerce")
cap = float(np.nanpercentile(kv, 95))
fig, ax = plt.subplots(figsize=(5.4, 6.6))
bze.assign(kv=kv.clip(upper=cap)).plot(
    column="kv", ax=ax, cmap=SEQUENTIAL, markersize=3, legend=True,
    legend_kwds={"label": f"SOC stock 0-30 cm (Mg/ha), capped at the 95th percentile ({cap:.0f})",
                 "shrink": 0.6},
    missing_kwds={"color": "#c9ced4", "markersize": 3})
ax.set_axis_off()
ax.set_title("BZE-LW published points, each up to 4 km from its site")
fig.tight_layout()
fig.savefig(FIGS / "01_bze_published_points.png", dpi=150)

# %% [markdown]
# ## BZE-LW: how far apart the published points are
#
# On the planned 8 km grid every site would have a neighbour at about 8 km.
# The offset moves each point by up to 4 km, so a pair's distance can be off
# by up to 8 km: the first lag bins of a variogram mix real separation with
# the offset.

# %%
xy = np.c_[t["_gx"].astype(float), t["_gy"].astype(float)]
nn_km = cKDTree(xy).query(xy, k=2)[0][:, 1] / 1000
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.hist(nn_km, bins=np.arange(0, np.ceil(nn_km.max()) + 0.5, 0.5),
        color=MARK, edgecolor="white", linewidth=0.5)
ax.axvline(8, color=INK, lw=1)
ax.text(8, 1.01, "8 km grid", transform=ax.get_xaxis_transform(),
        color=INK, fontsize=8, ha="center", va="bottom")
ax.set_xlabel("distance to the nearest published point (km)")
ax.set_ylabel("sites")
ax.set_title("Nearest published neighbour, all sites", pad=14)
fig.tight_layout()
fig.savefig(FIGS / "01_bze_nearest_neighbour.png", dpi=150)
pd.Series(nn_km).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(2)
