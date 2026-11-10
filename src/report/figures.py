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

Needs requirements-figures.txt (matplotlib). The colours are the portfolio's
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
    from src.transform import forests as F, kriging as K, maps as MP, scores as SC
except ImportError:         # imported with src/* on sys.path
    import forests as F
    import kriging as K
    import maps as MP
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
    print(f"the map's field: {site}, MEC {mec[site]:.2f} (next: {mec.drop(site).idxmax()}, "
          f"{mec.drop(site).max():.2f}); figures in {FIGURES.relative_to(K.ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
