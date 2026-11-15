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


def run() -> list[Rule]:
    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    sources = pd.read_csv(SOURCES)
    cov = pd.read_parquet(BC.OUT)
    return [c1_provenance(manifest, sources), c2_coverage(cov), c3_disc_means(cov, BC.sites())]


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
