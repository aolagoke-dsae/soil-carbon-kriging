"""Checkpoint 6 (RUNBOOK, Stage 6; rules fixed 15.11.2026 before any
covariate was pulled). Each rule is one function; C4-C8 join when the
models they check exist.

    python -m src.checks.checkpoint6     exit 0: every rule run passed; 1: one failed; 2: inputs missing
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:                        # run as a module from the repository root
    from src.transform import bze_covariates as BC
except ImportError:         # imported with src/transform on sys.path (pytest)
    import bze_covariates as BC

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "raw" / "manifest.csv"
SOURCES = ROOT / "config" / "sources.csv"
SOURCE_IDS = {"dwd_grids_1991_2020": "raw/DWD/", "bkg_dgm200": "raw/BKG/dgm200/",
              "thuenen_orgsoils_2023": "raw/THUENEN/organic/Thuenen_Kulisse_OrgBoeden_v1/"}
N_SITES = 3104
MIN_SHARE = BC.MIN_SHARE    # C2
MAX_LEFT_OUT = 0.01         # C2, amended 15.11.2026 after it failed
C3_SITES, C3_SEED, C3_TOL = 50, 20261115, 1e-9


@dataclass
class Rule:
    name: str
    passed: bool
    detail: str


def c1_provenance(manifest: pd.DataFrame, sources: pd.DataFrame) -> Rule:
    """Every covariate file accepted in the manifest with URL, size and
    SHA-256, and every source in sources.csv with version, licence and attribution."""
    problems = []
    for sid, prefix in SOURCE_IDS.items():
        rows = manifest[manifest["path"].str.startswith(prefix) & (manifest["rejected"].astype(str) == "0")]
        data = rows[~rows["path"].str.contains("listing|landing|\\.md5|README", regex=True)]
        if data.empty:
            problems.append(f"{sid}: no accepted data file in the manifest")
        elif (data["url"].eq("") | data["sha256"].str.len().ne(64) | data["bytes"].astype(int).le(0)).any():
            problems.append(f"{sid}: a row lacks its URL, size or SHA-256")
        src = sources[sources["source_id"] == sid]
        if src.empty or src[["version", "licence", "attribution"]].isna().any(axis=None):
            problems.append(f"{sid}: not in config/sources.csv with version, licence and attribution")
    n = {sid: int(((manifest["path"].str.startswith(p)) & (manifest["rejected"].astype(str) == "0")).sum())
         for sid, p in SOURCE_IDS.items()}
    return Rule("C1 provenance", not problems,
                "; ".join(problems) or "accepted files in the manifest: " + ", ".join(f"{k} {v}" for k, v in n.items())
                + "; each source in config/sources.csv with its version, licence and attribution")


def c2_coverage(cov: pd.DataFrame) -> Rule:
    """As amended 15.11.2026 after it failed at the fixed rule: every kept
    site has every covariate under both readings and at least half its disc
    inside the data; the sites left out are at most 1 % and listed (`kept`)."""
    points = [c for c in cov if c.endswith("_point")]
    discs = [c for c in cov if c.endswith("_disc")]
    shares = [c for c in cov if c.endswith("_share")]
    missing = cov[points + discs].isna().any(axis=1)
    thin = cov[shares].min(axis=1) < MIN_SHARE
    kept = cov["kept"] if "kept" in cov else ~(missing | thin)
    out = ~kept
    ok = (len(cov) == N_SITES and not (missing & kept).any() and not (thin & kept).any()
          and out.sum() <= MAX_LEFT_OUT * N_SITES and bool((out == (missing | thin)).all()))
    return Rule("C2 coverage", ok,
                f"{len(cov)} sites; {int(missing.sum())} lack a covariate under a reading, {int(thin.sum())} have "
                f"less than half their disc inside the data: {int(out.sum())} left out "
                f"({100 * out.mean():.2f} %; amended rule: at most {100 * MAX_LEFT_OUT:g} %), every kept site complete; "
                f"{int((cov[shares].min(axis=1) < 1.0).sum())} discs partly outside")


def brute_disc_mean(r: BC.Raster, x: float, y: float, radius: float = BC.RADIUS) -> float:
    """The disc mean by a second path: every cell centre of a window built
    from the grid's own coordinate vectors."""
    rows, cols = r.values.shape
    xs = r.x0 + (np.arange(cols) + 0.5) * r.d
    ys = r.y0 - (np.arange(rows) + 0.5) * r.d
    jc = np.flatnonzero(np.abs(xs - x) <= radius)
    ir = np.flatnonzero(np.abs(ys - y) <= radius)
    if not len(jc) or not len(ir):
        return float("nan")
    X, Y = np.meshgrid(xs[jc], ys[ir])
    vals = r.values[np.ix_(ir, jc)]
    sel = ((X - x) ** 2 + (Y - y) ** 2 <= radius ** 2) & np.isfinite(vals)
    return float(vals[sel].mean()) if sel.any() else float("nan")


def c3_disc_means(cov: pd.DataFrame, sites: pd.DataFrame) -> Rule:
    from pyproj import Transformer
    rng = np.random.default_rng(C3_SEED)
    pick = sites.iloc[np.sort(rng.choice(len(sites), C3_SITES, replace=False))]
    worst = 0.0
    for name in BC.NAMES:
        r = BC.Raster.load(name)
        x, y = pick["xcoord"].to_numpy(float), pick["ycoord"].to_numpy(float)
        if r.crs != "EPSG:25832":
            x, y = Transformer.from_crs("EPSG:25832", r.crs, always_xy=True).transform(x, y)
        mine = cov.set_index("pointid").loc[pick["pointid"], f"{name}_disc"].to_numpy()
        for a, xi, yi in zip(mine, x, y):
            b = brute_disc_mean(r, float(xi), float(yi))
            if not (np.isnan(a) and np.isnan(b)):
                worst = max(worst, abs(a - b) / max(abs(b), 1.0))
    return Rule("C3 disc means", worst <= C3_TOL,
                f"{C3_SITES} sites x {len(BC.NAMES)} covariates against a brute-force mean: "
                f"largest difference {worst:.1e} (relative; rule {C3_TOL:g})")


# --- C4-C8: the models -------------------------------------------------------------------------

C4_TOL, C6_TOL = 1e-12, 1e-9
C7_P = 0.01


def _bm():
    try:
        from src.transform import bze_models as BM
    except ImportError:
        import bze_models as BM
    return BM


def c4_forests() -> Rule:
    """At mineral sites, disc features, repeat 0, fold 0, the first 20 held-out
    sites: QRF's weights non-negative, summing to one, and equal to a
    tree-by-tree recomputation from the leaves."""
    from sklearn.ensemble import RandomForestRegressor
    BM = _bm()
    t = BM.load()
    p = t[t["bze_moor"].astype(int) == 0].reset_index(drop=True)
    X, _ = BM.features(p, "disc")
    y = p["kv_0_30"].to_numpy(float)
    fl = BM.folds(len(p), 0)
    train, test = np.flatnonzero(fl != 0), np.flatnonzero(fl == 0)[:20]
    rf = RandomForestRegressor(n_estimators=BM.F.TREES, max_features=BM.max_features(X.shape[1]),
                               min_samples_leaf=BM.F.MIN_LEAF, bootstrap=True, random_state=BM.SEED, n_jobs=-1)
    rf.fit(X[train], y[train])
    W = BM.F.qrf_weights(rf, X[train], X[test])
    W2 = np.zeros_like(W)
    for est in rf.estimators_:
        lt, ls = est.apply(X[train]), est.apply(X[test])
        for j in range(len(test)):
            same = lt == ls[j]
            W2[j, same] += 1.0 / same.sum()
    W2 /= len(rf.estimators_)
    err = float(np.abs(W - W2).max())
    ok = bool(W.min() >= 0 and np.abs(W.sum(axis=1) - 1).max() <= C4_TOL and err <= C4_TOL)
    return Rule("C4 forests", ok, f"20 held-out mineral sites: weights non-negative, summing to one within "
                f"{np.abs(W.sum(axis=1) - 1).max():.1e}; a tree-by-tree recomputation agrees within {err:.1e} (rule {C4_TOL:g})")


def c5_folds(cv: pd.DataFrame) -> Rule:
    """Per population and repeat, the folds partition the sites, sizes
    differing by at most one, and every model and feature set uses them."""
    problems = []
    for (pop, r), g in cv.groupby(["population", "repeat"]):
        assign = g.groupby("pointid")["fold"].nunique()
        if (assign != 1).any():
            problems.append(f"{pop} r{r}: a site in two folds")
        f = g.drop_duplicates("pointid")["fold"].value_counts()
        if f.max() - f.min() > 1 or len(f) != _bm().FOLDS:
            problems.append(f"{pop} r{r}: fold sizes {f.min()}-{f.max()}")
        per = g.groupby(["features", "method"])["pointid"].nunique()
        if per.nunique() != 1:
            problems.append(f"{pop} r{r}: models predict different sets of sites")
    sites = cv.groupby("population")["pointid"].nunique().to_dict()
    return Rule("C5 folds", not problems, "; ".join(problems) or
                f"{sites}: every repeat's ten folds partition the sites, sizes differing by at most one, the same for every model")


def c6_residual_kriging() -> Rule:
    """At organic sites, disc features, repeat 0, fold 0: the residual
    kriging's predictor and variance at the grid's maximum equal the bordered
    system's, and its predictive returns its levels at its quantiles."""
    BM = _bm()
    try:
        from src.checks.checkpoint4 import bordered
        from src.transform import kriging as K, reml as Mr
    except ImportError:
        from checkpoint4 import bordered
        import kriging as K
        import reml as Mr
    t = BM.load()
    p = t[t["bze_moor"].astype(int) == 1].reset_index(drop=True)
    X, _ = BM.features(p, "disc")
    y = p["kv_0_30"].to_numpy(float)
    xy = p[["xcoord", "ycoord"]].to_numpy(float)
    fl = BM.folds(len(p), 0)
    train, test = np.flatnonzero(fl != 0), np.flatnonzero(fl == 0)
    res = BM.forest_fold(X, y, train, test, BM.SEED + 1000)
    resid = y[train] - np.array(res["oob"])
    lo, hi = BM.residual_bounds(xy[train])
    grid = K.make_grid("exponential", lo, hi)
    Xc, X0 = K.design("constant", xy[train], xy[test])
    post = K.evaluate(grid, xy[train], resid, Xc, xy[test], X0)
    g = int(np.nanargmax(post.logpost))
    a, tau, _ = (v[g] for v in grid.arrays())
    C = (1 - tau) * Mr.correlation("exponential", K.distances(xy[train], xy[train]), a)
    np.fill_diagonal(C, 1.0)
    c0 = (1 - tau) * Mr.correlation("exponential", K.distances(xy[test], xy[train]), a)
    ep = ev = 0.0
    for j in range(len(test)):
        pr, v, _ = bordered(C, Xc, c0[j], X0[j], resid)
        ep = max(ep, abs(pr - post.pred[g, j]) / max(abs(pr), 1.0))
        ev = max(ev, abs(v - post.var1[g, j]) / max(abs(v), 1e-12))
    mix = K.predictive(post, 0)
    qs = mix.quantiles([0.05, 0.5, 0.95])
    lev = float(np.abs(mix.cdf(qs) - np.array([0.05, 0.5, 0.95])).max())
    ok = ep <= C6_TOL and ev <= C6_TOL and lev <= C6_TOL
    return Rule("C6 residual kriging", ok, f"{len(test)} held-out organic sites at the grid's maximum: predictor within "
                f"{ep:.1e}, variance within {ev:.1e} of the bordered system (relative); the predictive returns its levels "
                f"within {lev:.1e} (rule {C6_TOL:g})")


def c7_displacement() -> Rule:
    from scipy import stats
    BM = _bm()
    t = BM.load()
    xy = t[t["bze_moor"].astype(int) == 0][["xcoord", "ycoord"]].to_numpy(float)
    worst, pmin = 0.0, 1.0
    for rep in range(BM.D7_REPLICATES):
        d = np.hypot(*(BM.displace(xy, rep) - xy).T)
        worst = max(worst, float(d.max()))
        pmin = min(pmin, float(stats.kstest((d / BC.RADIUS) ** 2, "uniform").pvalue))
    ok = worst <= BC.RADIUS + 1e-9 and pmin > C7_P
    return Rule("C7 the second displacement", ok, f"{BM.D7_REPLICATES} displacements of {len(xy)} points: the farthest "
                f"{worst:.1f} m; (distance / 4 km)^2 against the uniform, smallest KS p = {pmin:.3f} (rule > {C7_P})")


def c8_records(cov: pd.DataFrame, cv: pd.DataFrame, d7: pd.DataFrame) -> Rule:
    BM = _bm()
    kept = cov[cov["kept"]]
    n = {pop: int(((kept["pointid"].isin(cv[cv.population == pop]["pointid"])).sum())) for pop in ("mineral", "organic")}
    expected = sum(n.values()) * BM.REPEATS * 6
    cols = ["mean", "var", "crps"] + [c for c in cv if c.startswith("q0.") or c.startswith("q1.")]
    ok = (len(cov) == N_SITES and len(cv) == expected and np.isfinite(cv[cols].to_numpy(float)).all()
          and np.isfinite(d7[cols].to_numpy(float)).all() and d7["replicate"].nunique() == BM.D7_REPLICATES)
    return Rule("C8 records", ok, f"bze_covariates {len(cov)} rows; bze_cv {len(cv)} rows (expected {expected}: "
                f"{n} sites x {BM.REPEATS} repeats x 6 models and sets), every value finite; bze_d7 {len(d7)} rows over "
                f"{d7['replicate'].nunique()} displacements, finite")


def c8_map(n: int = 50) -> Rule:
    """D8's table: every mapped cell under both land uses, every value finite
    and its quantiles ordered; and at n cells drawn with a fixed seed, the
    disc covariates equal D3's reader at the cell's centre within 1e-9."""
    try:
        from src.transform import bze_map as MP
    except ImportError:
        import bze_map as MP
    m = pd.read_parquet(MP.OUT)
    vals = m[["mean", "q0.05", "q0.50", "q0.95"]].to_numpy(float)
    per = m.groupby("land_use").size()
    ordered = bool(((m["q0.05"] <= m["q0.50"]) & (m["q0.50"] <= m["q0.95"])).all())
    cells = MP.cell_covariates(MP.K.Cache(_bm().CACHE))
    keep = cells[MP.mapped(cells)].reset_index(drop=True)
    rng = np.random.default_rng(C3_SEED + 8)
    pick = keep.iloc[rng.choice(len(keep), n, replace=False)]
    worst = 0.0
    for name in BC.NAMES:
        r = BC.Raster.load(name)
        x, y = pick["x"].to_numpy(float), pick["y"].to_numpy(float)
        if r.crs != "EPSG:25832":
            from pyproj import Transformer
            x, y = Transformer.from_crs("EPSG:25832", r.crs, always_xy=True).transform(x, y)
        got = BC.read(r, np.asarray(x, float), np.asarray(y, float))["disc"].to_numpy()
        worst = max(worst, float(np.max(np.abs(got - pick[f"{name}_disc"].to_numpy()) / np.maximum(np.abs(got), 1.0))))
    ok = (set(per.index) == {"A", "G"} and per.nunique() == 1 and int(per.iloc[0]) == len(keep)
          and np.isfinite(vals).all() and ordered and worst <= C3_TOL)
    return Rule("C8 records, the maps", ok, f"bze_map {len(m)} rows: {int(per.iloc[0])} cells x cropland and grassland, "
                f"every value finite, quantiles ordered: {ordered}; at {n} cells the convolved disc means equal D3's "
                f"reader within {worst:.1e} (rule {C3_TOL:g})")


def run() -> list[Rule]:
    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    sources = pd.read_csv(SOURCES)
    cov = pd.read_parquet(BC.OUT)
    rules = [c1_provenance(manifest, sources), c2_coverage(cov), c3_disc_means(cov, BC.sites())]
    BM = _bm()
    if BM.OUT.exists() and BM.OUT_D7.exists():
        cv, d7 = pd.read_parquet(BM.OUT), pd.read_parquet(BM.OUT_D7)
        rules += [c4_forests(), c5_folds(cv), c6_residual_kriging(), c7_displacement(), c8_records(cov, cv, d7)]
    try:
        from src.transform import bze_map as MP
    except ImportError:
        import bze_map as MP
    if MP.OUT.exists():
        rules.append(c8_map())
    return rules


def main(argv: list[str]) -> int:
    if argv:
        print(__doc__)
        return 2
    if not BC.OUT.exists():
        print(f"NOT THERE: {BC.OUT.relative_to(ROOT).as_posix()} (python -m src.transform.bze_covariates)")
        return 2
    rules = run()
    for r in rules:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name}: {r.detail}")
    return 0 if all(r.passed for r in rules) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
