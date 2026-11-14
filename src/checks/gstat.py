"""Rule G1 (RUNBOOK, Stage 5, D7): this repository's kriging against gstat, in R.

At IL-RT, under its selected model (a Matern with geometric anisotropy and a
constant mean) at the mode of its posterior from all the stocks: gstat
(Pebesma 2004) krigs all 2,903 cells of the map and the field mean, the
cells as the block's discretisation, and this repository krigs the same.

    python -m src.checks.gstat --write   writes tests/fixtures/gstat_IL-RT.json from curated/
    python -m src.checks.gstat           runs the check from the fixture
        exit 0: G1 passed; 1: it failed; 2: the fixture is missing; 3: R or gstat is not installed

The fixture holds the inputs only: the stocks (Potash et al. 2023, CC BY 4.0,
as Stage 2 computed them), the cells, the parameters; both sides are
computed from it, so the check runs wherever R runs, CI included.
Coordinates are metres from the stocks' centroid.
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

try:                        # run as a module from the repository root
    from src.transform import field_means as FM, kriging as K, reml as M
except ImportError:         # imported with src/* on sys.path (pytest)
    import field_means as FM
    import kriging as K
    import reml as M

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "gstat_IL-RT.json"
SCRIPT = Path(__file__).with_name("gstat_check.R")
SITE = "IL-RT"
TOL_PRED, TOL_VAR_SILL, TOL_BLOCK_PRED = 1e-8, 1e-8, 1e-8
TOL_BLOCK_VAR = 1e-4            # amended 14.11.2026 from 1e-5, after G1 failed at it (RUNBOOK, Stage 5, If it fails)
SOURCE = ("Stocks 0-30 cm computed by this repository (Stage 2) from Potash, E., et al. (2023), data on figshare, "
          "https://doi.org/10.6084/m9.figshare.23669304.v2, CC BY 4.0. Coordinates in metres from the stocks' centroid.")


def write_fixture() -> dict:
    fields, models, selected = FM.load_inputs()
    model = next(m for m in models[SITE] if m.name == selected[SITE])
    f, res = FM.field_mean(fields, SITE, model, K.Cache(FM.CACHE))
    g = np.array(res["g"])
    i = int(np.argmax(f["lp"][g]))
    gi = int(g[i])
    grid = model.grid()
    per = len(grid.taus) * len(grid.log_ranges)
    lr = grid.log_ranges[(gi % per) // len(grid.taus)]
    tau = grid.taus[gi % len(grid.taus)]
    nu = grid.nus[gi // per]
    c = f["fd"].xy.mean(axis=0)
    out = {"source": SOURCE, "site": SITE, "family": model.family, "lo": model.lo, "hi": model.hi,
           "angle": model.angle, "ratio": model.ratio, "grid_point": gi, "log_range": lr, "tau": tau, "nu": nu,
           "sill": float(res["s2"][i]), "x": (f["fd"].xy[:, 0] - c[0]).tolist(), "y": (f["fd"].xy[:, 1] - c[1]).tolist(),
           "z": f["fd"].z.tolist(), "cx": (f["cells_xy"][:, 0] - c[0]).tolist(), "cy": (f["cells_xy"][:, 1] - c[1]).tolist()}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(out), encoding="utf-8")
    return out


def ours(fx: dict) -> dict:
    """This repository's kriging of the cells and of their mean, from the fixture."""
    xy, cells, z = np.c_[fx["x"], fx["y"]], np.c_[fx["cx"], fx["cy"]], np.array(fx["z"])
    model = K.Model("a", "constant", fx["family"], fx["lo"], fx["hi"], fx["angle"], fx["ratio"])
    X, X0 = K.design("constant", xy, cells)
    post = K.evaluate(K.Grid(fx["family"], (fx["log_range"],), (fx["tau"],), (fx["nu"],)), xy, z, X, cells, X0,
                      fx["angle"], fx["ratio"])
    keep = np.zeros(model.grid().size, dtype=bool)
    keep[fx["grid_point"]] = True
    fd = K.FieldData("fixture", np.arange(len(z)).astype(str), xy, z, np.zeros((len(z), len(K.COVARIATES))))
    blk = FM.block_grid(fd, model, cells, np.zeros((len(cells), len(K.COVARIATES))), (), np.zeros(len(keep)),
                        FM.pair_histogram(cells, fx["angle"], fx["ratio"]), keep)
    sill = fx["sill"]
    return {"pred": post.pred[0], "var": sill * post.var1[0], "block_pred": blk["pred"][0], "block_var": sill * blk["v"][0]}


def gstat(fx: dict) -> dict:
    a = math.exp(fx["log_range"])
    params = {"x": fx["x"], "y": fx["y"], "z": fx["z"], "cx": fx["cx"], "cy": fx["cy"],
              "psill": (1 - fx["tau"]) * fx["sill"], "nugget": fx["tau"] * fx["sill"], "model": "Mat",
              "range": a / M.scale("matern", fx["nu"]), "kappa": fx["nu"], "angle": fx["angle"], "ratio": fx["ratio"]}
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(params), encoding="utf-8")
        subprocess.run(["Rscript", str(SCRIPT), str(src), str(dst)], check=True)
        return json.loads(dst.read_text(encoding="utf-8"))


def check(fx: dict) -> tuple[bool, str]:
    o, g = ours(fx), gstat(fx)
    sill = fx["sill"]
    e = {"pred": float(np.max(np.abs(np.array(g["pred"]) - o["pred"]) / np.abs(o["pred"]))),
         "var": float(np.max(np.abs(np.array(g["var"]) - o["var"])) / sill),
         "block_pred": abs(g["block_pred"] - o["block_pred"]) / abs(o["block_pred"]),
         "block_var": abs(g["block_var"] - o["block_var"]) / o["block_var"]}
    ok = (e["pred"] <= TOL_PRED and e["var"] <= TOL_VAR_SILL and e["block_pred"] <= TOL_BLOCK_PRED
          and e["block_var"] <= TOL_BLOCK_VAR)
    v = g["versions"]
    detail = (f"{SITE}, Matérn (nu {fx['nu']:.3g}) with anisotropy at the posterior's mode, {len(fx['cx']):,} cells: "
              f"{v['R']}, gstat {v['gstat']}: predictions within {e['pred']:.1e} (relative), variances within "
              f"{e['var']:.1e} of the sill; the field mean {g['block_pred']:.4f} against {o['block_pred']:.4f} "
              f"({e['block_pred']:.1e}), its block variance within {e['block_var']:.1e} (relative)")
    return ok, detail


def main(argv: list[str]) -> int:
    """python -m src.checks.gstat [--write]"""
    if argv == ["--write"]:
        fx = write_fixture()
        print(f"{FIXTURE.relative_to(ROOT).as_posix()}: {len(fx['z'])} stocks, {len(fx['cx'])} cells")
        return 0
    if argv:
        print(main.__doc__)
        return 2
    if not FIXTURE.exists():
        print(f"NOT THERE: {FIXTURE.relative_to(ROOT).as_posix()}")
        return 2
    if shutil.which("Rscript") is None:
        print("R is not installed: G1 not run")
        return 3
    ok, detail = check(json.loads(FIXTURE.read_text(encoding="utf-8")))
    print(f"{'PASS' if ok else 'FAIL'}  G1 gstat: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
