# %% [markdown]
# # 04 · Kriging and its uncertainty, against two forests (Checkpoint 4)
#
# Exploration only, as the portfolio standard requires: the rules live in
# `src/checks/checkpoint4.py` and run without this notebook
# (`python -m src.checks.checkpoint4`). This draws, field by field, the
# cross-validation's coverage (QCP and PICP, D8) for the selected kriging,
# its plug-in twin, the two forests and the baseline, and then the maps of
# D10: the selected kriging's predictive mean and 90 % interval width on
# Stage 3's 10 m cells inside each field's location hull, with QRF's beside.
#
# Needs `curated/potash_cv.parquet` and the other tables of Stage 4 (RUNBOOK,
# Stage 4) and `requirements-figures.txt`. The maps take a few minutes; each
# field's cells are cached in `notebooks/scratch/`, which git ignores, with
# the figures.

# %%
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
if not (ROOT / "src").is_dir():         # VS Code's interactive window starts in notebooks/
    ROOT = ROOT.parent
os.chdir(ROOT)                          # curated/ is relative to the repository root
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, Delaunay

from src.transform import forests as F, kriging as K, reml as M, scores as SC
from src.transform import patches as P

FIGS = ROOT / "notebooks" / "scratch"
FIGS.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"     # text, secondary text, gridlines
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGES = ["#fbd7ca", "#f5c4b2", "#f2b098", "#eb9c7f", "#e68764", "#df7249", "#d95923",
           "#c94908", "#b44005", "#9f3600", "#8a2e00", "#752600", "#611e00"]
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 8, "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})

cv = pd.read_parquet(F.OUT_CV)
selected = pd.read_parquet(K.OUT_SELECTION).set_index("site")["selected"]
fields, record, aniso, boxcox = K.load_inputs()
SITES = sorted(fields)
MODEL_NAME = {"a": "(a) constant", "b": "(b) quadratic", "c": "(c) terrain drift"}

# %% [markdown]
# ## Coverage, field by field
#
# QCP is the share of held-out stocks below each of the 39 predictive
# quantiles, PICP the share inside each central interval from 5 to 95 %; both
# pool the three repeats. On the grey diagonal a method's intervals hold what
# they claim. The kriging is the mean model D5 selected for the field, its
# Bayesian predictive (D3) and its plug-in twin; QRF and RF-RK are D7's
# forests, the baseline D8's.

# %%
SERIES = [("kriging_{m}", BLUE, "-", "Bayesian kriging, the selected model"),
          ("plugin_{m}", ORANGE, "-", "plug-in kriging, the same model"),
          ("qrf", AQUA, "-", "QRF"), ("rfrk", YELLOW, "-", "RF-RK"),
          ("baseline", MUTED, "--", "baseline")]
fig, axes = plt.subplots(4, 4, figsize=(11, 10.4))
for i, site in enumerate(SITES):
    ax_q, ax_p = axes[i // 2, 2 * (i % 2)], axes[i // 2, 2 * (i % 2) + 1]
    for ax in (ax_q, ax_p):
        ax.plot([0, 1], [0, 1], color=GRID, lw=2.0, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    for pattern, colour, style, _ in SERIES:
        g = cv[(cv["site"] == site) & (cv["method"] == pattern.format(m=selected[site]))]
        qcp, picp = SC.coverage(g[K.QCOLS].to_numpy(), g["y"].to_numpy())
        ax_q.plot(SC.LEVELS, qcp, color=colour, ls=style, lw=1.2)
        ax_p.plot(SC.CENTRAL, picp, color=colour, ls=style, lw=1.2, marker="o", ms=2.2)
    ax_q.set_title(f"{site}, {MODEL_NAME[selected[site]]}: QCP")
    ax_p.set_title(f"{site}: PICP")
    if i // 2 == 3:
        ax_q.set_xlabel("quantile level")
        ax_p.set_xlabel("nominal coverage")
    if i % 2 == 0:
        ax_q.set_ylabel("share observed")
handles = [Line2D([], [], color=c, ls=s, lw=1.4, label=lab) for _, c, s, lab in SERIES]
fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.01, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(FIGS / "04_coverage.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## The maps: predicting at the 10 m cells
#
# Kriging at a cell is the selected model's Bayesian predictive from all the
# field's stocks (D3), for a new observation there, nugget included, as in the
# cross-validation. Every grid point's correlation matrix is diagonalised once,
# R = U diag(l) U', so that C = (1 - tau) R + tau I has the eigenvalues
# (1 - tau) l + tau for every nugget-to-sill ratio at the cost of one: the same
# algebra as `kriging.evaluate`, organised for thousands of targets. Grid
# points whose posterior weight is below 1e-12 of the largest are left out of
# the maps' mixtures (they move no quantile by more than about 1e-9). QRF is
# one forest trained on all 1,163 stocks, with D7's settings.

# %%
def load_grid(site):
    with np.load(K.CURATED / "terrain" / f"{site}.npz") as z:
        return {k: z[k] for k in z.files}


def hull_cells(site, g):
    """The 10 m cells whose centres lie inside the field's location hull, with
    every covariate defined: rows, columns, coordinates and covariates."""
    ny, nx = g["elevation"].shape
    cell = float(g["cell"])
    xs = float(g["x0"]) + (np.arange(nx) + 0.5) * cell
    ys = float(g["y1"]) - (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    inside = Delaunay(fields[site].xy).find_simplex(np.c_[X.ravel(), Y.ravel()]) >= 0
    rows, cols = np.divmod(np.flatnonzero(inside), nx)
    cov = np.column_stack([g[c][rows, cols].astype(float) for c in K.COVARIATES])
    ok = np.all(np.isfinite(cov), axis=1)
    return rows[ok], cols[ok], np.c_[X[rows[ok], cols[ok]], Y[rows[ok], cols[ok]]], cov[ok]


def drift_of(site):
    fd = fields[site]
    model = next(m for m in K.models_for(site, record, aniso, boxcox) if m.name == "c")
    return K.drift_candidates()[K.choose(K.ml_table(model.grid(), fd.xy, fd.z, fd.cov))]


def shared_others_all(site, model):
    """Under (b) at a small field: the other fields' integrated likelihoods, all stocks."""
    total = 0.0
    for other in SITES:
        if other != site:
            o = fields[other]
            total = total + K.evaluate(model.grid(), o.xy, o.z, M.design(o.xy, "quadratic", o.xy.mean(axis=0))).logpost
    return total


def kriging_cells(site, xy0, cov0):
    """Mean, 5 % and 95 % quantiles of the selected model's predictive at each cell."""
    fd = fields[site]
    model = next(m for m in K.models_for(site, record, aniso, boxcox) if m.name == selected[site])
    drift = drift_of(site) if model.mean == "drift" else ()
    X, X0 = K.design(model.mean, fd.xy, xy0, fd.cov, cov0, drift)
    n, p = X.shape
    D, D0 = K.distances(fd.xy, fd.xy, model.angle, model.ratio), K.distances(xy0, fd.xy, model.angle, model.ratio)
    grid = model.grid()
    lp = K.evaluate(grid, fd.xy, fd.z, X, angle=model.angle, ratio=model.ratio).logpost
    if model.shared:
        lp = lp + shared_others_all(site, model)
    keep = lp >= lp.max() + np.log(1e-12)
    w = np.exp(lp[keep] - lp.max())
    w /= w.sum()
    locs, scales, g = [], [], 0
    for a, nu, taus in grid.points():
        if not keep[g:g + len(taus)].any():
            g += len(taus)
            continue
        lam, U = np.linalg.eigh(M.correlation(model.family, D, a, nu))
        Uz, UX, P0 = U.T @ fd.z, U.T @ X, U.T @ M.correlation(model.family, D0, a, nu).T
        for tau in taus:
            if keep[g]:
                dinv = 1.0 / ((1.0 - tau) * lam + tau)
                XCX = UX.T @ (dinv[:, None] * UX)
                beta = np.linalg.solve(XCX, UX.T @ (dinv * Uz))
                r = Uz - UX @ beta
                s2 = float(r @ (dinv * r)) / (n - p)
                Kw = (1.0 - tau) * P0
                pred = X0 @ beta + Kw.T @ (dinv * r)
                G = X0.T - UX.T @ (dinv[:, None] * Kw)
                v1 = 1.0 - ((Kw * Kw) * dinv[:, None]).sum(axis=0) + (G * np.linalg.solve(XCX, G)).sum(axis=0)
                locs.append(pred)
                scales.append(np.sqrt(s2 * np.maximum(v1, 1e-12)))
            g += 1
    locs, scales = np.array(locs).T, np.array(scales).T          # cells x components
    return mixture_summary(w, locs, scales, float(n - p))


def mixture_summary(w, locs, scales, df, chunk=400):
    """The mean and the 5 and 95 % quantiles of each cell's mixture of t, by
    Newton's method from the normal with the mixture's mean and variance."""
    table, out = SC.t_table(df), []
    for s in range(0, len(locs), chunk):
        L, S = locs[s:s + chunk], scales[s:s + chunk]
        mean = L @ w
        sd = np.sqrt((S ** 2 * df / (df - 2.0) + L ** 2) @ w - mean ** 2)
        qs = []
        for level in (0.05, 0.95):
            x = mean + sd * SC.t_ppf(level, df)
            for _ in range(60):
                z = (x[:, None] - L) / S
                err = table.cdf(z) @ w - level
                if np.max(np.abs(err)) < 1e-12:
                    break
                x = x - err / ((SC.t_pdf(z, df) / S) @ w)
            qs.append(x)
        out.append(np.c_[mean, qs[0], qs[1]])
    return np.vstack(out)


stacked = F.stack(pd.read_parquet(K.STOCKS), pd.read_parquet(K.TERRAIN), pd.read_parquet(F.FO.OUT))
forest_all = None


def qrf_cells(site, xy0, cov0, g):
    global forest_all
    if forest_all is None:
        forest_all = F.forest(F.SEED)
        forest_all.set_params(oob_score=False, n_jobs=-1)
        forest_all.fit(stacked.X, stacked.y)
    lon, lat = P.transformer(int(g["epsg"]), "EPSG:4326").transform(xy0[:, 0], xy0[:, 1])
    W = F.qrf_weights(forest_all, stacked.X, np.column_stack([lon, lat, cov0]))
    out = []
    for w in W:
        d = SC.Weighted(stacked.y, w)
        q = d.quantiles([0.05, 0.95])
        out.append([d.mean(), q[0], q[1]])
    return np.array(out)


# The cache lets a run stop between fields: SCK_MAP_SITES names the fields to
# compute in this run (all by default), SCK_MAP_BUDGET the seconds after which
# no new field is started.
ONLY = os.environ.get("SCK_MAP_SITES")
BUDGET = float(os.environ.get("SCK_MAP_BUDGET", "inf"))
start = time.monotonic()
maps = {}
for site in SITES:
    path = FIGS / f"04_cells_{site}.parquet"
    if path.exists():
        maps[site] = pd.read_parquet(path)
        continue
    if (ONLY and site not in ONLY.split(",")) or time.monotonic() - start > BUDGET:
        print(f"{site}: not computed in this run")
        continue
    g = load_grid(site)
    rows, cols, xy0, cov0 = hull_cells(site, g)
    k = kriging_cells(site, xy0, cov0)
    q = qrf_cells(site, xy0, cov0, g)
    maps[site] = pd.DataFrame({"row": rows, "col": cols, "kriging_mean": k[:, 0],
                               "kriging_width": k[:, 2] - k[:, 1], "qrf_mean": q[:, 0],
                               "qrf_width": q[:, 2] - q[:, 1]})
    maps[site].to_parquet(path, index=False)
    print(f"{site}: {len(rows)} cells, {time.monotonic() - start:.0f} s")

# %% [markdown]
# ## The maps
#
# Each field on its own scales: one for the two predictive means, one for the
# two 90 % interval widths, so that kriging and QRF are read against each
# other. The dots are the sampled locations, the line their hull.

# %%
def as_image(g, m, column):
    a = np.full(g["elevation"].shape, np.nan)
    a[m["row"].to_numpy(), m["col"].to_numpy()] = m[column].to_numpy()
    return a


def extent(g):
    ny, nx = g["elevation"].shape
    return 0.0, nx * float(g["cell"]) / 1000, 0.0, ny * float(g["cell"]) / 1000


for site, m in maps.items():
    g = load_grid(site)
    fd = fields[site]
    y0 = float(g["y1"]) - g["elevation"].shape[0] * float(g["cell"])
    pts = np.c_[(fd.xy[:, 0] - float(g["x0"])) / 1000, (fd.xy[:, 1] - y0) / 1000]
    hull = pts[np.r_[ConvexHull(pts).vertices, ConvexHull(pts).vertices[:1]]]
    rr, cc = m["row"].to_numpy(), m["col"].to_numpy()
    ny, nx = g["elevation"].shape
    box = (cc.min() / nx * extent(g)[1] - 0.03, (cc.max() + 1) / nx * extent(g)[1] + 0.03,
           (ny - rr.max() - 1) / ny * extent(g)[3] - 0.03, (ny - rr.min()) / ny * extent(g)[3] + 0.03)
    aspect = (box[3] - box[2]) / (box[1] - box[0])          # each panel as tall as the field is, for its width
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 2 * 3.6 * min(max(aspect, 0.4), 1.7) + 1.4))
    scales = {"mean": (np.nanpercentile(m[["kriging_mean", "qrf_mean"]], [1, 99]), BLUES, "predictive mean"),
              "width": (np.nanpercentile(m[["kriging_width", "qrf_width"]], [1, 99]), ORANGES,
                        "90 % interval width")}
    for row, kind in enumerate(("mean", "width")):
        (lo, hi), ramp, label = scales[kind]
        for col, method in enumerate(("kriging", "qrf")):
            ax = axes[row, col]
            im = ax.imshow(as_image(g, m, f"{method}_{kind}"), extent=extent(g), vmin=lo, vmax=hi,
                           cmap=LinearSegmentedColormap.from_list(kind, ramp), interpolation="nearest")
            ax.plot(hull[:, 0], hull[:, 1], color=INK, lw=0.8)
            ax.scatter(pts[:, 0], pts[:, 1], s=2.5, color=INK, lw=0)
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
            ax.set_aspect("equal")
            who = f"kriging {MODEL_NAME[selected[site]]}" if method == "kriging" else "QRF"
            ax.set_title(f"{who}: {label} (Mg C/ha)")
        cb = fig.colorbar(im, ax=axes[row].tolist(), shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        cb.outline.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("km east")
    for ax in axes[:, 0]:
        ax.set_ylabel("km north")
    fig.suptitle(f"{site}: 0-30 cm carbon stocks predicted at {len(m):,} cells of 10 m", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(FIGS / f"04_maps_{site}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
print(f"figures in {FIGS}:", sorted(p.name for p in FIGS.glob("04_*.png")))

# %% [markdown]
# ## What the maps add up to
#
# The cells' predictive means and widths, summarised beside the
# cross-validation's widths for the same methods.

# %%
rows = []
for site, m in maps.items():
    k = cv[(cv["site"] == site) & (cv["method"] == f"kriging_{selected[site]}")]
    q = cv[(cv["site"] == site) & (cv["method"] == "qrf")]
    lo, hi = SC.central_bounds(0.9)
    rows.append({"field": site, "model": selected[site], "cells": len(m),
                 "kriging mean": m["kriging_mean"].mean(), "QRF mean": m["qrf_mean"].mean(),
                 "kriging width, cells": m["kriging_width"].median(), "kriging width, CV":
                 (k[K.QCOLS[hi]] - k[K.QCOLS[lo]]).median(),
                 "QRF width, cells": m["qrf_width"].median(), "QRF width, CV": (q[K.QCOLS[hi]] - q[K.QCOLS[lo]]).median()})
print(pd.DataFrame(rows).round(2).to_string(index=False))
