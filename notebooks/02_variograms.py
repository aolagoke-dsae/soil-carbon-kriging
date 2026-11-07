# %% [markdown]
# # 02 · Variograms (Checkpoint 2)
#
# Exploration only, as the portfolio standard requires: the rules live in
# `src/checks/checkpoint2.py` and run without this notebook
# (`python -m src.checks.checkpoint2`). This draws, for every field, the
# moment variogram (Matheron's estimator, with Cressie and Hawkins' robust
# one beside it), the baseline fit of D8 and the REML model Stage 4 krigs
# with; and, for the four large fields, the directional moment variograms
# that go with the REML anisotropy check.
#
# Needs the curated tables (RUNBOOK, Stage 2: stocks, reml, anisotropy,
# variograms) and `requirements-figures.txt`. The figures are written to
# `notebooks/scratch/`, which git ignores.

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
from matplotlib.lines import Line2D

from src.transform import anisotropy as AN
from src.transform import reml as M
from src.transform import variograms as VG

FIGS = ROOT / "notebooks" / "scratch"
FIGS.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"     # text, secondary text, gridlines
# Categorical slots in their fixed order (the validated reference palette,
# light mode): the first three hold in a scatter, all four in a line chart.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})
MEAN = {"constant": "(a)", "quadratic": "(b)"}
SHORT = {"exponential": "exp.", "spherical": "sph.", "matern": "Matérn"}

vario = pd.read_parquet(VG.OUT)
fits = pd.read_parquet(VG.OUT_FITS)
record = pd.read_parquet(M.OUT)
aniso = pd.read_parquet(AN.OUT)

# %% [markdown]
# ## Each field: the moment estimate, the baseline fit and the REML model
#
# Filled blue points are the bins of 30 pairs or more, the ones the baseline
# fits (D8); hollow ones have fewer pairs. Grey rings: Cressie and Hawkins'
# estimate for the same bins. Orange: the baseline's weighted least-squares
# fit. Aqua: the REML model Stage 4 krigs with, the field's own or the shared
# one (D7), drawn as its semivariogram, sill x [tau + (1 - tau)(1 - R(h))].
# Under (a) the values are the stocks; under (b) the moment variogram uses
# their OLS residuals from the second-order trend, as gstat does. In the
# panel titles: exp. exponential, sph. spherical.


# %%
def reml_semivariogram(x, h):
    nu = x["nu"] if x["family"] == "matern" else None
    return x["sill"] * (x["tau"] + (1.0 - x["tau"]) * (1.0 - M.correlation(x["family"], h, x["eff_range"], nu)))


def baseline_curve(f, h):
    return f["nugget"] + f["partial_sill"] * VG.shape(f["family"], h, f["a"])


def draw_fields(mean):
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8))
    for ax, site in zip(axes.ravel(), sorted(vario["site"].unique())):
        b = vario[(vario["site"] == site) & (vario["mean"] == mean) & (vario["direction"] == "omni")
                  & (vario["pairs"] > 0)]
        f = fits[(fits["site"] == site) & (fits["mean"] == mean) & fits["chosen"]].iloc[0]
        x = record[(record["site"] == site) & (record["mean"] == mean) & record["for_stage4"]].iloc[0]
        h = np.linspace(0.5, b["hi"].max(), 300)
        ax.plot(h, reml_semivariogram(x, h), color=AQUA, lw=1.2, zorder=2)
        ax.plot(h, baseline_curve(f, h), color=ORANGE, lw=1.2, zorder=2)
        ax.scatter(b["lag"], b["gamma_ch"], s=30, facecolors="none", edgecolors=MUTED, linewidths=0.8, zorder=3)
        many, few = b[b["fitted"]], b[~b["fitted"]]
        ax.scatter(many["lag"], many["gamma"], s=22, color=BLUE, linewidths=0, zorder=4)
        ax.scatter(few["lag"], few["gamma"], s=22, facecolors="none", edgecolors=BLUE, linewidths=1.0, zorder=4)
        ax.set_xlim(0, b["hi"].max())
        ax.set_ylim(0, 1.08 * max(b["gamma"].max(), b["gamma_ch"].max(), x["sill"], f["sill"]))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{site}\nREML {SHORT[x['family']]}, {'shared' if x['fit'] == 'joint_all' else 'own'}"
                     f" · baseline {SHORT[f['family']]}", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("lag (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("semivariance ((Mg C/ha)²)")
    handles = [Line2D([], [], marker="o", ls="none", color=BLUE, markersize=5, label="Matheron, 30 pairs or more"),
               Line2D([], [], marker="o", ls="none", markerfacecolor="none", markeredgecolor=BLUE,
                      markersize=5, label="Matheron, fewer pairs"),
               Line2D([], [], marker="o", ls="none", markerfacecolor="none", markeredgecolor=MUTED,
                      markersize=5.5, label="Cressie–Hawkins"),
               Line2D([], [], color=ORANGE, lw=1.2, label="baseline fit (D8)"),
               Line2D([], [], color=AQUA, lw=1.2, label="REML model for Stage 4")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, labelcolor=INK)
    label = "the stocks" if mean == "constant" else "OLS residuals from the second-order trend"
    fig.suptitle(f"Mean model {MEAN[mean]}: {label}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


for mean in M.MEANS:
    fig = draw_fields(mean)
    fig.savefig(FIGS / f"02_variograms_{MEAN[mean][1]}.png", dpi=150)
    plt.close(fig)

# %% [markdown]
# ## The large fields by direction
#
# Matheron's estimate in four directions, 0, 45, 90 and 135 degrees clockwise
# from north with 22.5 degrees either side, bins of 30 pairs or more. The
# title gives the REML refit with geometric anisotropy: the azimuth of its
# major axis, the ratio of the minor range to the major, and the change in AIC against the
# isotropic fit. Stage 4 adopts the anisotropy only where the AIC falls by
# more than 2.

# %%
DIRECTIONS = dict(zip(("0", "45", "90", "135"), (BLUE, ORANGE, AQUA, YELLOW)))


def label_ends(ax, ends):
    """Direct labels at the right of the lines, pushed apart where they would
    collide: a dash in the line's colour, the text in ink."""
    lo, hi = ax.get_ylim()
    gap = 0.07 * (hi - lo)
    ends = sorted(ends, key=lambda e: e[1])
    ys = []
    for _, y, _, _ in ends:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    top = hi - 0.03 * (hi - lo)
    if ys and ys[-1] > top:
        ys = [y - (ys[-1] - top) for y in ys]
    x = max(e[0] for e in ends)
    for (_, _, text, colour), y in zip(ends, ys):
        ax.plot([x], [y], marker="_", markersize=9, mew=2, color=colour, clip_on=False, zorder=5)
        ax.annotate(text, (x, y), xytext=(7, 0), textcoords="offset points", fontsize=7, color=INK,
                    va="center", annotation_clip=False)
fig, axes = plt.subplots(2, 4, figsize=(11, 6.0))
for row, mean in enumerate(M.MEANS):
    for col, site in enumerate(M.LARGE):
        ax = axes[row, col]
        ends = []
        for direction, colour in DIRECTIONS.items():
            b = vario[(vario["site"] == site) & (vario["mean"] == mean) & (vario["direction"] == direction)
                      & (vario["pairs"] >= VG.MIN_PAIRS)]
            ax.plot(b["lag"], b["gamma"], color=colour, lw=1.2, marker="o", markersize=3.5)
            if len(b):
                ends.append((b["lag"].iloc[-1], b["gamma"].iloc[-1], f"{direction}°", colour))
        a = aniso[(aniso["site"] == site) & (aniso["mean"] == mean)].iloc[0]
        ax.set_title(f"{site} {MEAN[mean]}\nREML {a['angle']:.0f}°, ratio {a['ratio']:.2f}, "
                     f"ΔAIC {a['delta_aic']:+.1f}", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
        label_ends(ax, ends)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.12)
for ax in axes[-1]:
    ax.set_xlabel("lag (m)")
for ax in axes[:, 0]:
    ax.set_ylabel("semivariance ((Mg C/ha)²)")
fig.legend(handles=[Line2D([], [], color=c, lw=1.2, marker="o", markersize=3.5, label=f"{d}°")
                    for d, c in DIRECTIONS.items()],
           loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK,
           title="direction, clockwise from north", title_fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIGS / "02_directional.png", dpi=150)
plt.close(fig)
print("figures:", sorted(p.name for p in FIGS.glob("02_*.png")))
