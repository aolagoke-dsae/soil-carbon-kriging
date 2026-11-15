"""Review of v1.0, M2: the field means against the range's upper bound
(RUNBOOK, Stage 5, Addendum on M2, W1-W4; rules fixed 15.11.2026 before this
was written).

    python -m src.transform.range_bound [--budget SECONDS]
        writes curated/potash_range_bound.parquet and prints W2-W4. Every
        step is cached in Stage 5's cache, so a run stopped by --budget
        (exit 4) continues where it stopped, and the original bound's block
        kriging is read from Stage 5's own steps.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dataclasses
import sys
import time

import numpy as np
import pandas as pd

try:                        # run as a module from the repository root
    from src.transform import field_means as FM, kriging as K, maps as MP
except ImportError:         # imported with src/transform on sys.path (pytest)
    import field_means as FM
    import kriging as K
    import maps as MP

OUT = K.CURATED / "potash_range_bound.parquet"
SITES = ("IL-DG", "IL-MC", "IL-RT")         # W1
FACTORS = (1, 2, 3)                          # W1; 1 reproduces Stage 5 (W2)
TOL_REPRO = 1e-9                             # W2, Mg C/ha
SHIFT_SD = 0.5                               # W4: the mean moves by less than half its sd
WIDTH = 0.20                                 # W4: the width changes by less than a fifth
ORCHESTRATION = ("main", "run")
_CODE = K.code_hash(__file__, FM.__file__, MP.__file__, skip=ORCHESTRATION)


def widened(model: K.Model, factor: float) -> K.Model:
    """W1: the same model with its upper bound on the effective range multiplied."""
    return model if factor == 1 else dataclasses.replace(model, hi=model.hi * factor)


def posterior(fields: dict, site: str, model: K.Model, drift: tuple) -> np.ndarray:
    """D2's log posterior on the model's grid, from all the field's stocks,
    with the drift given (W1), and the shared posterior under (b) at a small
    field."""
    fd = fields[site]
    others = MP.shared_others_all(fields, site, model) if model.shared else None
    return MP.log_posterior(fd, model, drift, others)


def block(f: dict, site: str, model: K.Model, lp: np.ndarray, cache: K.Cache) -> dict:
    """D2's block kriging in Stage 5's cached steps (field_means.field_mean,
    with the posterior given), so that the original bound's steps are Stage 5's."""
    res = None
    for chunk in FM.pair_chunks(model, lp >= lp.max() + np.log(FM.KEEP)):
        mask = np.zeros(len(lp), dtype=bool)
        mask[chunk] = True
        part = cache.get(["block", FM._CODE, site, model.key(), int(chunk[0]), int(chunk[-1])] + f["data"],
                         lambda: FM.block_grid(f["fd"], model, f["cells_xy"], f["cells_cov"], f["drift"], lp,
                                               f["hist"], mask))
        res = part if res is None else {k: (res[k] + part[k] if isinstance(part[k], list) else part[k]) for k in part}
    return res


def mass_above(model: K.Model, lp: np.ndarray, hi: float) -> float:
    """W3: the posterior weight on effective ranges above hi."""
    a, _, _ = model.grid().arrays()
    w = np.exp(lp - lp.max())
    return float(w[a > hi * (1 + 1e-12)].sum() / w.sum())


def verdict(rows: pd.DataFrame) -> pd.DataFrame:
    """W4 per field."""
    out = []
    for site, g in rows.groupby("site", sort=True):
        base = g[g["factor"] == 1].iloc[0]
        wider = g[g["factor"] != 1]
        shift = (wider["mean"] - base["mean"]).abs().max()
        change = ((wider["q975"] - wider["q025"]) / (base["q975"] - base["q025"]) - 1).abs().max()
        out.append({"site": site, "max_shift": shift, "shift_in_sd": shift / base["sd"], "max_width_change": change,
                    "robust": bool(shift < SHIFT_SD * base["sd"] and change < WIDTH)})
    return pd.DataFrame(out)


def run(fields: dict, models: dict, selected, cache: K.Cache, log=lambda m: None) -> pd.DataFrame:
    stage5 = pd.read_parquet(FM.OUT).set_index("site")
    rows = []
    for site in SITES:
        model = next(m for m in models[site] if m.name == selected[site])
        f = FM.field_inputs(fields, site, model, cache)
        for factor in FACTORS:
            t0 = time.monotonic()
            m = widened(model, factor)
            lp = np.array(cache.get(["m2-posterior", _CODE, site, m.key(), list(f["drift"])] + f["data"],
                                    lambda: posterior(fields, site, m, f["drift"]).tolist()), dtype=float)
            res = block(f, site, m, lp, cache)
            s = FM.summary(res, lp, f["fd"])
            row = {"site": site, "model": m.name, "factor": factor, "hi": m.hi, "grid": m.grid().size,
                   "mean": s["mean"], "sd": s["sd"], "q025": s["q025"], "q975": s["q975"],
                   "mass_above": mass_above(m, lp, model.hi)}
            if factor == 1:
                row["repro"] = max(abs(s["mean"] - stage5.loc[site, "mean"]), abs(s["q025"] - stage5.loc[site, "q025"]),
                                   abs(s["q975"] - stage5.loc[site, "q975"]))
            rows.append(row)
            log(f"{site} x{factor}: {time.monotonic() - t0:.0f} s")
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    budget = None
    if argv[:1] == ["--budget"] and len(argv) == 2:
        budget = float(argv[1])
    elif argv:
        print(__doc__)
        return 2
    fields, models, selected = FM.load_inputs()
    cache = K.Cache(FM.CACHE, None if budget is None else time.monotonic() + budget)
    try:
        rows = run(fields, models, selected, cache, log=print)
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget; run again to continue from the cache.")
        return 4
    rows.to_parquet(OUT, index=False)
    pd.set_option("display.width", 220)
    print(rows.round(4).to_string(index=False))
    repro = rows["repro"].dropna().max()
    print(f"\nW2 reproduction: largest difference from Stage 5 {repro:.2e} Mg C/ha "
          f"({'PASS' if repro <= TOL_REPRO else 'FAIL'}, rule {TOL_REPRO:g})")
    print(verdict(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
