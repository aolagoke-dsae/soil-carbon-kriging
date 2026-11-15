"""Rules L1 and L2 (RUNBOOK, Stage 2, Addendum on m4): this repository's REML
against R's nlme (Pinheiro & Bates 2000).

    python -m src.checks.nlme --write   writes tests/fixtures/reml_fields.json from curated/
    python -m src.checks.nlme           runs the check from the fixture
        exit 0: L1 passed; 1: it failed; 2: the fixture is missing; 3: R or nlme is not installed

The fixture holds the stocks (Potash et al. 2023, CC BY 4.0, as Stage 2
computed them), in metres from each field's centroid, and this repository's
REML estimates for L2; both sides' criteria are computed from it.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:                        # run as a module from the repository root
    from src.transform import reml as M
except ImportError:         # imported with src/* on sys.path (pytest)
    import reml as M

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "reml_fields.json"
SCRIPT = Path(__file__).with_name("nlme_check.R")
STOCKS = ROOT / "curated" / "potash_stocks_0_30.parquet"
FAMILY = "exponential"
RANGES = (50.0, 200.0, 800.0)          # L1, m
TAUS = (0.1, 0.4, 0.7)                 # L1
MEANS = ("constant", "quadratic")      # L1
LARGE = ("IL-RD", "IL-RS", "IL-RT", "NE")   # L2
TOL = 1e-6                             # L1
SOURCE = ("Stocks 0-30 cm computed by this repository (Stage 2) from Potash, E., et al. (2023), data on figshare, "
          "https://doi.org/10.6084/m9.figshare.23669304.v2, CC BY 4.0. Coordinates in metres from each field's centroid.")


def parameter_sets() -> list[tuple[float, float]]:
    return [(a, t) for a in RANGES for t in TAUS]


def write_fixture(stocks: Path = STOCKS) -> dict:
    t = pd.read_parquet(stocks)
    fields = []
    for site, g in t.groupby("site", sort=True):
        xy = g[["x", "y"]].to_numpy(float)
        xy = xy - xy.mean(axis=0)
        z = g["stock"].to_numpy(float)
        row = {"site": site, "x": xy[:, 0].tolist(), "y": xy[:, 1].tolist(), "z": z.tolist()}
        if site in LARGE:
            f = M.fit([M.Field.of(xy, z, "constant")], FAMILY)
            row["ours"] = {"eff_range": float(f.eff_range), "tau": float(f.tau), "sill": float(f.sill)}
        fields.append(row)
    out = {"source": SOURCE, "family": FAMILY, "fields": fields}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(out), encoding="utf-8")
    return out


def ours(fx: dict) -> list[dict]:
    """L1, this side: reml_profile at every field, mean model and parameter set."""
    out = []
    for f in fx["fields"]:
        xy, z = np.c_[f["x"], f["y"]], np.array(f["z"])
        for mean in MEANS:
            field = M.Field.of(xy, z, mean)
            ll = [M.reml_profile(field, FAMILY, a, t)[0] for a, t in parameter_sets()]
            out.append({"site": f["site"], "mean": mean, "X": field.X.tolist(), "loglik": ll})
    return out


def nlme(fx: dict, mine: list[dict]) -> dict:
    sets = parameter_sets()
    by_site = {f["site"]: f for f in fx["fields"]}
    params = {"ranges": [a for a, _ in sets], "taus": [t for _, t in sets],
              "cases": [{"site": m["site"], "mean": m["mean"], "x": by_site[m["site"]]["x"],
                         "y": by_site[m["site"]]["y"], "z": by_site[m["site"]]["z"], "X": m["X"]} for m in mine],
              "estimate": [{"site": f["site"], "x": f["x"], "y": f["y"], "z": f["z"],
                            "start_range": f["ours"]["eff_range"], "start_tau": max(f["ours"]["tau"], 0.05)}
                           for f in fx["fields"] if "ours" in f]}
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(params), encoding="utf-8")
        subprocess.run(["Rscript", str(SCRIPT), str(src), str(dst)], check=True)
        return json.loads(dst.read_text(encoding="utf-8"))


def compare(mine: list[dict], theirs: dict) -> tuple[float, list[float]]:
    """L1: the largest disagreement in the differences from the first
    parameter set, and each case's offset between the two criteria."""
    worst, offsets = 0.0, []
    for m, r in zip(mine, theirs["l1"]):
        assert (m["site"], m["mean"]) == (r["site"], r["mean"])
        a, b = np.array(m["loglik"]), np.array(r["loglik"], dtype=float)
        worst = max(worst, float(np.max(np.abs((a - a[0]) - (b - b[0])))))
        offsets.append(float(np.mean(b - a)))
    return worst, offsets


def main(argv: list[str]) -> int:
    """python -m src.checks.nlme [--write]"""
    if argv == ["--write"]:
        fx = write_fixture()
        print(f"{FIXTURE.relative_to(ROOT).as_posix()}: {sum(len(f['z']) for f in fx['fields'])} stocks, "
              f"{len(fx['fields'])} fields")
        return 0
    if argv:
        print(main.__doc__)
        return 2
    if not FIXTURE.exists():
        print(f"NOT THERE: {FIXTURE.relative_to(ROOT).as_posix()}")
        return 2
    if shutil.which("Rscript") is None:
        print("R is not installed: L1 not run")
        return 3
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mine = ours(fx)
    theirs = nlme(fx, mine)
    worst, offsets = compare(mine, theirs)
    ok = worst <= TOL
    v = theirs["versions"]
    print(f"{'PASS' if ok else 'FAIL'}  L1 REML against nlme: {v['R']}, nlme {v['nlme']}: 8 fields x 2 mean models "
          f"x {len(parameter_sets())} parameter sets; differences from the first set agree within {worst:.1e} "
          f"(rule: {TOL:g}); offset between the criteria {min(offsets):.6f} to {max(offsets):.6f}")
    ests = {f["site"]: f["ours"] for f in fx["fields"] if "ours" in f}
    for e in theirs["l2"]:
        o = ests[e["site"]]
        if e["ok"]:
            print(f"      L2 {e['site']}: effective range {o['eff_range']:.0f} m here, {e['eff_range']:.0f} m in nlme; "
                  f"nugget ratio {o['tau']:.3f} here, {e['tau']:.3f} in nlme")
        else:
            print(f"      L2 {e['site']}: nlme did not converge ({e['message'].strip()[:80]})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
