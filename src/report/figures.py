"""Stage 4's two figures (RUNBOOK, Stage 4, Addendum, D11), drawn from the
curated tables alone.

    python -m src.report.figures
        docs/figures/map_<field>.png    the field whose selected kriging has the
                                        highest MEC in the cross-validation: its
                                        predictive mean beside its standard deviation
        docs/figures/calibration.png    QCP against the quantile level, with the
                                        1:1 line: Bayesian and plug-in kriging under
                                        each field's selected mean model, and QRF
        docs/figures/calibration.csv    the values the calibration figure draws

Needs matplotlib (requirements.txt). The colours are the portfolio's
data-visualisation palette: three categorical hues, checked for colour-vision
deficiency, and one-hue sequential ramps for the maps.
"""
from __future__ import annotations

import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

try:                        # run as a module from the repository root
    from src.transform import forests as F, kriging as K, maps as MP, reml as M, scores as SC
except ImportError:         # imported with src/* on sys.path
    import forests as F
    import kriging as K
    import maps as MP
    import reml as M
    import scores as SC

FIGURES = K.ROOT / "docs" / "figures"
SURFACE, INK, SECONDARY, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGES = ["#fbd7ca", "#f5c4b2", "#f2b098", "#eb9c7f", "#e68764", "#df7249", "#d95923",
           "#c94908", "#b44005", "#9f3600", "#8a2e00", "#752600", "#611e00"]
SERIES = (("kriging", BLUE, "Bayesian kriging"), ("plugin", ORANGE, "plug-in kriging"),
          ("qrf", AQUA, "quantile regression forest"))
MEAN_NAME = {"a": "constant mean", "b": "quadratic trend", "c": "terrain as external drift"}
DPI = 200

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.titlelocation": "left", "axes.titlecolor": INK,
    "axes.labelcolor": SECONDARY, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.solid_capstyle": "round",
})


def method_of(kind: str, model: str) -> str:
    return "qrf" if kind == "qrf" else f"{kind}_{model}"


def mec_by_field(cv: pd.DataFrame, selected: pd.Series) -> pd.Series:
    """Each field's MEC under its selected kriging, per repeat and averaged (D8)."""
    out = {}
    for site, model in selected.items():
        g = cv[(cv["site"] == site) & (cv["method"] == f"kriging_{model}")]
        out[site] = np.mean([SC.point_scores(r["mean"], r["y"])["mec"] for _, r in g.groupby("repeat")])
    return pd.Series(out)


def calibration_table(cv: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    """QCP at the 39 levels, the repeats pooled: for the eight fields together
    and for each, per series."""
    rows = []
    for kind, _, _ in SERIES:
        parts = [cv[(cv["site"] == s) & (cv["method"] == method_of(kind, m))] for s, m in selected.items()]
        for scope, g in [("all fields", pd.concat(parts))] + [(p["site"].iloc[0], p) for p in parts]:
            qcp, _ = SC.coverage(g[K.QCOLS].to_numpy(), g["y"].to_numpy())
            rows += [{"scope": scope, "series": kind, "predictions": len(g), "level": lev, "qcp": q,
                      "departure": q - lev} for lev, q in zip(SC.LEVELS, qcp)]
    return pd.DataFrame(rows)


def draw_map(site: str, maps: pd.DataFrame, fd: K.FieldData, model: K.Model, mec: float, out) -> None:
    g = MP.load_grid(site)
    m = maps[maps["site"] == site]
    ny, nx = g["elevation"].shape
    cell = float(g["cell"])
    hull = ConvexHull(fd.xy)
    corner = fd.xy.min(axis=0)
    left, top = (float(g["x0"]) - corner[0]) / 1000, (float(g["y1"]) - corner[1]) / 1000
    extent = (left, left + nx * cell / 1000, top - ny * cell / 1000, top)
    pts = (fd.xy - corner) / 1000
    ring = pts[np.r_[hull.vertices, hull.vertices[:1]]]
    span = pts.max(axis=0)
    aspect = min(max((span[1] + 0.04) / (span[0] + 0.04), 0.6), 1.7)
    width, head, foot = 8.6, 1.15, 0.55                   # inches: the figure, the header, the axis labels
    panel = (0.82 * width) / 2.5                           # two panels and half a panel between them
    height = head + panel * aspect + foot
    fig = plt.figure(figsize=(width, height))
    axes = fig.subplots(1, 2, gridspec_kw={"left": 0.08, "right": 0.9, "wspace": 0.5,
                                            "top": 1 - head / height, "bottom": foot / height})
    family = {"matern": "Matérn", "exponential": "exponential", "spherical": "spherical"}[model.family]
    fig.text(0.008, 1 - 0.1 / height, f"{site}: soil organic carbon stock in 0–30 cm, predicted at {len(m):,} "
             f"cells of 10 m", ha="left", va="top", fontsize=11, color=INK)
    fig.text(0.008, 1 - 0.45 / height, textwrap.fill(
        f"Bayesian kriging from the field's {len(fd.z)} sampled stocks (dots), with a {MEAN_NAME[model.name]} and "
        f"a {family} covariance{' with anisotropy' if model.angle is not None else ''}, the uncertainty of its "
        f"parameters carried through. Cross-validated MEC {mec:.2f}.", 125), ha="left", va="top", fontsize=8,
        color=SECONDARY, linespacing=1.4)
    panels = (("kriging_mean", BLUES, "Predicted stock: the predictive mean"),
              ("kriging_sd", ORANGES[:9], "Uncertainty: the predictive standard deviation"))
    for i, (ax, (column, ramp, title)) in enumerate(zip(axes, panels)):
        image = np.full((ny, nx), np.nan)
        image[m["row"].to_numpy(), m["col"].to_numpy()] = m[column].to_numpy()
        im = ax.imshow(image, extent=extent, cmap=LinearSegmentedColormap.from_list(column, ramp),
                       vmin=float(m[column].min()), vmax=float(m[column].max()), interpolation="nearest")
        ax.plot(ring[:, 0], ring[:, 1], color=INK, lw=0.7)
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color=INK, edgecolors=SURFACE, linewidths=0.5, zorder=3)
        ax.set_xlim(-0.02, span[0] + 0.02)
        ax.set_ylim(-0.02, span[1] + 0.02)
        ax.set_aspect("equal")
        ax.set_title(title, pad=6)
        ax.set_xlabel("km east")
        if i == 0:
            ax.set_ylabel("km north")
        cb = fig.colorbar(im, cax=ax.inset_axes([1.05, 0.15, 0.045, 0.7]))
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5, colors=MUTED, labelcolor=SECONDARY)
        cb.set_label("Mg C/ha", color=SECONDARY, fontsize=8)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def draw_calibration(table: pd.DataFrame, selected: pd.Series, out) -> None:
    fig = plt.figure(figsize=(11.2, 5.2))
    fig.text(0.008, 0.985, "Calibration: the share of held-out stocks below each predictive quantile (QCP)",
             ha="left", va="top", fontsize=11, color=INK)
    fig.text(0.008, 0.925, textwrap.fill(
        "Three repeats of ten-fold cross-validation at 1,163 locations, each field under the mean model D5 "
        "selected. A calibrated predictive follows the 1:1 line. Left, the eight fields pooled; right, each "
        "field's departure from the 1:1 line.", 175), ha="left", va="top", fontsize=8, color=SECONDARY,
        linespacing=1.4)
    grid = fig.add_gridspec(2, 6, width_ratios=[2.3, 0.18, 1, 1, 1, 1], wspace=0.35, hspace=0.55,
                            left=0.06, right=0.99, top=0.78, bottom=0.1)
    main = fig.add_subplot(grid[:, 0])
    main.plot([0, 1], [0, 1], color=BASELINE, lw=1.0, zorder=1)
    for kind, colour, _ in SERIES:
        t = table[(table["scope"] == "all fields") & (table["series"] == kind)]
        main.plot(t["level"], t["qcp"], color=colour, lw=1.6, zorder=2)
    main.set_xlim(0, 1)
    main.set_ylim(0, 1)
    main.set_aspect("equal")
    main.set_xticks([0, 0.25, 0.5, 0.75, 1])
    main.set_yticks([0, 0.25, 0.5, 0.75, 1])
    main.grid(color=GRID, lw=0.5)
    main.set_axisbelow(True)
    n = int(table.loc[table["scope"] == "all fields", "predictions"].iloc[0])
    main.set_title(f"The eight fields pooled ({n:,} predictions each)", pad=6)
    main.set_xlabel("quantile level")
    main.set_ylabel("share of held-out stocks below the quantile")
    keys = [Line2D([], [], color=c, lw=1.8, label=lab) for _, c, lab in SERIES]
    keys.append(Line2D([], [], color=BASELINE, lw=1.0, label="1:1, calibrated"))
    main.legend(handles=keys, loc="upper left", frameon=False, fontsize=7.5, handlelength=1.6, labelcolor=INK)
    dep = table.loc[table["scope"] != "all fields", "departure"].abs().max()
    lim = float(np.ceil(dep / 0.05) * 0.05)
    for i, site in enumerate(selected.index):
        ax = fig.add_subplot(grid[i // 4, 2 + i % 4])
        ax.axhline(0, color=BASELINE, lw=1.0, zorder=1)
        for kind, colour, _ in SERIES:
            t = table[(table["scope"] == site) & (table["series"] == kind)]
            ax.plot(t["level"], t["departure"], color=colour, lw=1.3, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([-lim, 0, lim])
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{site}, ({selected[site]})", pad=4, fontsize=8.5)
        if i % 4 == 0:
            ax.set_ylabel("QCP minus level")
        if i // 4 == 1:
            ax.set_xlabel("quantile level")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


PALETTE8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _title(fig, title: str, subtitle: str, width: int = 120) -> None:
    fig.text(0.008, 0.985, title, ha="left", va="top", fontsize=11, color=INK)
    fig.text(0.008, 0.92, textwrap.fill(subtitle, width), ha="left", va="top", fontsize=8, color=SECONDARY,
             linespacing=1.4)


def variogram_table(site: str, fields: dict, record: pd.DataFrame, aniso, boxcox) -> tuple[pd.DataFrame, dict]:
    """The moment estimates, the baseline fit, the REML model and the 90 % band
    of the semivariogram under Stage 4's posterior from all the stocks, (a)."""
    vg = pd.read_parquet(K.CURATED / "potash_variograms.parquet")
    vg = vg[(vg["site"] == site) & (vg["mean"] == "constant") & (vg["direction"] == "omni") & (vg["pairs"] > 0)]
    fits = pd.read_parquet(K.CURATED / "potash_variogram_fits.parquet")
    base = fits[(fits["site"] == site) & (fits["mean"] == "constant") & fits["chosen"].astype(bool)].iloc[0]
    reml = record[(record["site"] == site) & (record["mean"] == "constant") & record["for_stage4"].astype(bool)].iloc[0]
    model = next(m for m in K.models_for(site, record, aniso, boxcox) if m.name == "a")
    fd = fields[site]
    post = K.evaluate(model.grid(), fd.xy, fd.z, M.design(fd.xy, "constant", fd.xy.mean(axis=0)))
    w = post.weights()
    a, t, nu = model.grid().arrays()
    h = np.linspace(0.0, float(vg["hi"].max()), 121)[1:]
    e_sigma = post.s2 / (post.df - 2.0)
    gam = np.array([e_sigma[i] * (1 - (1 - t[i]) * M.correlation(model.family, h, a[i], None if np.isnan(nu[i]) else nu[i]))
                    for i in range(len(w))])
    lo = [K.weighted_quantiles(gam[:, j], w, [0.05])[0] for j in range(len(h))]
    hi = [K.weighted_quantiles(gam[:, j], w, [0.95])[0] for j in range(len(h))]
    rnu = None if pd.isna(reml["nu"]) else float(reml["nu"])
    curves = pd.DataFrame({"h": h, "posterior_q05": lo, "posterior_q95": hi,
                           "reml": reml["nugget"] + reml["partial_sill"] * (1 - M.correlation(reml["family"], h, reml["eff_range"], rnu)),
                           "baseline": base["nugget"] + base["partial_sill"] * (1 - M.correlation(base["family"], h, base["eff_range"], None))})
    meta = {"reml_family": reml["family"], "base_family": base["family"], "n": len(fd.z)}
    return curves, vg[["lag", "gamma", "pairs", "fitted"]].reset_index(drop=True), meta


def draw_variograms(site: str, curves: pd.DataFrame, points: pd.DataFrame, meta: dict, out) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.subplots_adjust(top=0.78, left=0.1, right=0.97, bottom=0.12)
    ax.fill_between(curves["h"], curves["posterior_q05"], curves["posterior_q95"], color=BLUE, alpha=0.12, lw=0,
                    label="Bayesian posterior, 90 % band (Stage 4)")
    ax.plot(curves["h"], curves["reml"], color=BLUE, lw=1.8, label=f"REML, {meta['reml_family']} (Stage 2)")
    ax.plot(curves["h"], curves["baseline"], color=ORANGE, lw=1.4, label=f"moment fit, {meta['base_family']} (gstat-style baseline)")
    used = points[points["fitted"].astype(bool)]
    ax.scatter(used["lag"], used["gamma"], s=8 + 40 * used["pairs"] / used["pairs"].max(), color=INK,
               edgecolors=SURFACE, linewidths=0.6, zorder=3, label="moment estimates (size: pairs)")
    ax.set_xlim(0, curves["h"].max())
    ax.set_ylim(0, None)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("semivariance ((Mg C/ha)²)")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=INK)
    _title(fig, f"{site}: the semivariogram of the 0–30 cm stocks, three ways",
           f"{meta['n']} stocks, constant mean. The moment estimates and their fit are the baseline the author ran in R "
           f"with gstat; REML is the model Stage 4 krigs with; the band is what its posterior allows.", 110)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def crps_table(cv: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    rows = []
    for site, model in selected.items():
        def crps(method):
            g = cv[(cv["site"] == site) & (cv["method"] == method)]
            return float(g.groupby("repeat")["crps"].mean().mean())
        base = crps("baseline")
        for kind, method in (("kriging", f"kriging_{model}"), ("qrf", "qrf"), ("rfrk", "rfrk")):
            c = crps(method)
            rows.append({"site": site, "series": kind, "crps": c, "crps_baseline": base, "skill": 1 - c / base})
    return pd.DataFrame(rows)


def draw_crps(table: pd.DataFrame, selected: pd.Series, out) -> None:
    names = {"kriging": ("Bayesian kriging, selected model", BLUE), "qrf": ("quantile regression forest", AQUA),
             "rfrk": ("random forest, kriged residuals", ORANGE)}
    sites = list(selected.index)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.subplots_adjust(top=0.74, left=0.14, right=0.97, bottom=0.12)
    ax.axvline(0, color=BASELINE, lw=1.0)
    for k, (kind, (label, colour)) in enumerate(names.items()):
        t = table[table["series"] == kind].set_index("site").loc[sites]
        ax.scatter(t["skill"], np.arange(len(sites)) + (k - 1) * 0.22, s=36, color=colour, edgecolors=SURFACE,
                   linewidths=0.8, zorder=3, label=label)
    ax.set_yticks(np.arange(len(sites)))
    ax.set_yticklabels([f"{s} ({selected[s]})" for s in sites])
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("CRPS skill against the field's own distribution of stocks (1 − CRPS / CRPS of the baseline)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK)
    _title(fig, "How much location and terrain add: CRPS skill by field",
           "Three repeats of ten-fold cross-validation. 0 is the baseline, the training fold's own stocks of the field; "
           "higher is better. The forests learn across all eight fields; kriging field by field.", 110)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def draw_field_means(means: pd.DataFrame, out) -> None:
    m = means.sort_values("site", ascending=False).reset_index(drop=True)
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.subplots_adjust(top=0.7, left=0.14, right=0.97, bottom=0.12)
    ax.hlines(y + 0.15, m["q025"], m["q975"], color=BLUE, lw=2.2)
    ax.scatter(m["mean"], y + 0.15, s=30, color=BLUE, edgecolors=SURFACE, linewidths=0.8, zorder=3,
               label="Bayesian block kriging: mean, 95 % interval")
    ax.hlines(y - 0.15, m["sample_lo"], m["sample_hi"], color=ORANGE, lw=2.2)
    ax.scatter(m["sample_mean"], y - 0.15, s=30, color=ORANGE, edgecolors=SURFACE, linewidths=0.8, zorder=3,
               label="sample mean, 95 % t interval (simple random sampling's formula)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} ({k})" for s, k in zip(m["site"], m["model"])])
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("mean stock in 0–30 cm over the field's cells (Mg C/ha)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, frameon=False, fontsize=7.5, labelcolor=INK)
    _title(fig, "Each field's mean stock, model-based and design-based",
           "The field is the hull of its samples (Potash et al. publish no boundaries). At IL-BR, IL-MC and NE the "
           "samples are not a probability sample of the field, so the design-based interval is a naive one.", 110)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def draw_detectable_change(effort: pd.DataFrame, means: pd.DataFrame, out) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    fig.subplots_adjust(top=0.8, left=0.1, right=0.8, bottom=0.12)
    mm = means.set_index("site")
    for colour, (site, g) in zip(PALETTE8, effort.groupby("site")):
        g = g[g["n"] >= 9].sort_values("n_target").drop_duplicates("n")      # one grid per number of points
        pct = 100 * g["mdc_grid"] / mm.loc[site, "mean"]
        ax.plot(g["n"], pct, color=colour, lw=1.6, label=f"{site} ({mm.loc[site, 'model']})")
        ax.scatter([mm.loc[site, "n"]], [100 * mm.loc[site, "mdc"] / mm.loc[site, "mean"]], s=34, color=colour,
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7.5, labelcolor=INK)
    ax.axhline(10, color=BASELINE, lw=1.0)
    ax.axhline(5, color=BASELINE, lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 25, 50, 100, 200, 400])
    ax.set_xticklabels(["10", "25", "50", "100", "200", "400"])
    ax.set_ylim(1, 40)
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.set_yticklabels(["2 %", "5 %", "10 %", "20 %", "50 %"])
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("samples per campaign, on a square grid")
    ax.set_ylabel("smallest detectable change in the field mean")
    _title(fig, "How many samples to detect a change in a field's mean stock",
           "Two independent campaigns of n samples, block kriging, 80 % power at the 5 % level; the change as a share "
           "of the field's mean stock in 0–30 cm, at fixed depth. Dots: each field's current design. Under the fitted "
           "models, averaged over their posterior, and assuming they carry over to the next campaign.", 110)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def bze_map_table(m: pd.DataFrame) -> pd.DataFrame:
    """Stage 6, D8's maps as the figure draws them: the cell's column and row
    on the 1 km grid (from its centre's x and y, EPSG:25832), the land use,
    and the median and 90 % interval, to 0.1 Mg C/ha."""
    out = pd.DataFrame({"col": ((m["x"] - 280500) // 1000).astype(int), "row": ((6101500 - m["y"]) // 1000).astype(int),
                        "land_use": m["land_use"]})
    for q in ("q0.05", "q0.50", "q0.95"):
        out[q] = m[q].round(1)
    return out


def draw_bze_map(t: pd.DataFrame, out) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap
    nr, nc = int(t["row"].max()) + 1, int(t["col"].max()) + 1
    bounds = [0, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200]
    cmap = ListedColormap(BLUES[1:12])
    norm = BoundaryNorm(bounds, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 6.4))
    fig.subplots_adjust(top=0.78, left=0.02, right=0.98, bottom=0.13, wspace=0.04)
    for ax, (lu, label) in zip(axes, (("A", "if cropland"), ("G", "if grassland"))):
        g = t[t["land_use"] == lu]
        img = np.full((nr, nc), np.nan)
        img[g["row"].to_numpy(), g["col"].to_numpy()] = g["q0.50"].to_numpy()
        ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{label}: median {g['q0.50'].median():.0f} Mg C/ha", fontsize=9)
        ax.set_axis_off()
    cax = fig.add_axes([0.25, 0.08, 0.5, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=bounds)
    cb.set_label("predicted median stock in 0-30 cm, mineral soils (Mg C/ha)", color=SECONDARY, fontsize=8)
    cb.outline.set_edgecolor(BASELINE)
    _title(fig, "Mineral soils' carbon in 0-30 cm across Germany, at 1 km, from the published BZE-LW points",
           "Quantile regression forest on climate (DWD 1991-2020), terrain (BKG DGM200) and the map of organic soils "
           "(Thuenen 2023), each averaged over 4 km. Conditional on the land use, not masked to farmland. Data: Thuenen "
           "BZE-LW (CC BY 4.0); DWD (CC BY 4.0); (c) GeoBasis-DE / BKG (2026) dl-de/by-2-0.", 120)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main(argv: list[str]) -> int:
    """python -m src.report.figures"""
    if argv:
        print(main.__doc__)
        return 2
    for p in (F.OUT_CV, K.OUT_SELECTION, MP.OUT):
        if not p.exists():
            print(f"NOT THERE: {p.relative_to(K.ROOT).as_posix()}")
            return 2
    cv = pd.read_parquet(F.OUT_CV)
    selected = pd.read_parquet(K.OUT_SELECTION).set_index("site")["selected"].sort_index()
    fields, record, aniso, boxcox = K.load_inputs()
    FIGURES.mkdir(parents=True, exist_ok=True)
    mec = mec_by_field(cv, selected)
    site = str(mec.idxmax())
    model = next(m for m in K.models_for(site, record, aniso, boxcox) if m.name == selected[site])
    draw_map(site, pd.read_parquet(MP.OUT), fields[site], model, mec[site], FIGURES / f"map_{site}.png")
    table = calibration_table(cv, selected)
    table.round(4).to_csv(FIGURES / "calibration.csv", index=False)
    draw_calibration(table, selected, FIGURES / "calibration.png")
    curves, points, meta = variogram_table("IL-RS", fields, record, aniso, boxcox)
    curves.round(4).to_csv(FIGURES / "variograms_IL-RS.csv", index=False)
    draw_variograms("IL-RS", curves, points, meta, FIGURES / "variograms_IL-RS.png")
    crps = crps_table(cv, selected)
    crps.round(4).to_csv(FIGURES / "crps_by_field.csv", index=False)
    draw_crps(crps, selected, FIGURES / "crps_by_field.png")
    try:
        from src.transform import field_means as FM
    except ImportError:
        import field_means as FM
    if FM.OUT.exists():
        means, effort = pd.read_parquet(FM.OUT), pd.read_parquet(FM.OUT_EFFORT)
        means.round(4).to_csv(FIGURES / "field_means.csv", index=False)
        effort.round(4).to_csv(FIGURES / "detectable_change.csv", index=False)
        draw_field_means(means, FIGURES / "field_means.png")
        draw_detectable_change(effort, means, FIGURES / "detectable_change.png")
    try:
        from src.transform import bze_map as BMAP
    except ImportError:
        import bze_map as BMAP
    if BMAP.OUT.exists():
        bt = bze_map_table(pd.read_parquet(BMAP.OUT))
        bt.to_csv(FIGURES / "bze_map.csv.gz", index=False, compression="gzip")
        draw_bze_map(bt, FIGURES / "bze_map.png")
    print(f"the map's field: {site}, MEC {mec[site]:.2f} (next: {mec.drop(site).idxmax()}, "
          f"{mec.drop(site).max():.2f}); figures in {FIGURES.relative_to(K.ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
