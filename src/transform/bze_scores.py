"""Stage 6, D6 and D7's report: the scores of every model, feature set and
population, and the cost of the second displacement (RUNBOOK, Stage 6).

    python -m src.transform.bze_scores
        reads curated/bze_cv.parquet and curated/bze_d7.parquet, writes
        curated/bze_scores.parquet and prints the tables
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:                        # run as a module from the repository root
    from src.transform import bze_models as BM, kriging as K
except ImportError:         # imported with src/transform on sys.path (pytest)
    import bze_models as BM
    import kriging as K

OUT = BM.BC.CURATED / "bze_scores.parquet"
CENTRAL = (0.5, 0.8, 0.9, 0.95)


def _q(level: float) -> str:
    return f"q{level:.3f}"


def picp(g: pd.DataFrame, p: float) -> float:
    lo, hi = round((1 - p) / 2, 3), round(1 - (1 - p) / 2, 3)
    return float(((g["y"] >= g[_q(lo)]) & (g["y"] <= g[_q(hi)])).mean())


def scores(g: pd.DataFrame) -> dict:
    """D6 for one group: CRPS, ME, RMSE and MEC per repeat and averaged;
    PICP and the largest QCP departure pooled over the repeats."""
    per = []
    for _, d in g.groupby("repeat"):
        e = d["y"] - d["mean"]
        per.append({"crps": d["crps"].mean(), "me": e.mean(), "rmse": float(np.sqrt((e ** 2).mean())),
                    "mec": 1 - (e ** 2).sum() / ((d["y"] - d["y"].mean()) ** 2).sum()})
    out = pd.DataFrame(per).mean().to_dict()
    out["crps_spread"] = float(pd.DataFrame(per)["crps"].max() - pd.DataFrame(per)["crps"].min())
    for p in CENTRAL:
        out[f"picp{int(p * 100)}"] = picp(g, p)
    out["qcp_max"] = max(abs(float((g["y"] < g[c]).mean()) - float(c[1:])) for c in K.QCOLS)
    out["n"] = int(g["pointid"].nunique())
    return out


def table(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pop, s, m), g in cv.groupby(["population", "features", "method"], sort=True):
        rows.append({"population": pop, "features": s, "method": m, **scores(g)})
    return pd.DataFrame(rows)


def paired(cv: pd.DataFrame, pop: str, a: tuple, b: tuple) -> dict:
    """The mean paired CRPS difference a - b over the same sites and repeats,
    and the share of sites where a is lower."""
    key = ["pointid", "repeat"]
    A = cv[(cv.population == pop) & (cv.features == a[0]) & (cv.method == a[1])].set_index(key)["crps"]
    B = cv[(cv.population == pop) & (cv.features == b[0]) & (cv.method == b[1])].set_index(key)["crps"]
    d = (A - B.reindex(A.index)).dropna()
    return {"population": pop, "a": "/".join(a), "b": "/".join(b), "mean_diff": float(d.mean()),
            "a_lower_share": float((d < 0).mean())}


def d7_table(cv: pd.DataFrame, d7: pd.DataFrame) -> pd.DataFrame:
    """D7: QRF's scores after one displacement (the published points, on the
    sites D7 could read) and after two, per replicate."""
    rows = []
    for s in ("point", "disc"):
        for rep, g in d7[d7.features == s].groupby("replicate"):
            once = cv[(cv.population == "mineral") & (cv.features == s) & (cv.method == "qrf")
                      & cv.set_index(["pointid", "repeat"]).index.isin(g.set_index(["pointid", "repeat"]).index)]
            a, b = scores(once), scores(g)
            rows.append({"features": s, "replicate": int(rep), "n": b["n"], "crps_once": a["crps"],
                         "crps_twice": b["crps"], "mec_once": a["mec"], "mec_twice": b["mec"],
                         "picp90_twice": b["picp90"]})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    if argv:
        print(__doc__)
        return 2
    cv, d7 = pd.read_parquet(BM.OUT), pd.read_parquet(BM.OUT_D7)
    t = table(cv)
    t.to_parquet(OUT, index=False)
    pd.set_option("display.width", 240)
    cols = ["population", "features", "method", "n", "crps", "crps_spread", "me", "rmse", "mec", "picp50", "picp80",
            "picp90", "picp95", "qcp_max"]
    print(t[cols].round(3).to_string(index=False))
    pairs = []
    for pop in ("mineral", "organic"):
        pairs += [paired(cv, pop, ("disc", "qrf"), ("point", "qrf")), paired(cv, pop, ("disc", "rfrk"), ("point", "rfrk")),
                  paired(cv, pop, ("disc", "rfrk"), ("disc", "qrf")), paired(cv, pop, ("desc", "qrf"), ("disc", "qrf")),
                  paired(cv, pop, ("disc", "qrf"), ("none", "baseline"))]
    print("\npaired CRPS differences (a - b):")
    print(pd.DataFrame(pairs).round(3).to_string(index=False))
    print("\nD7, QRF after one and two displacements:")
    d = d7_table(cv, d7)
    print(d.round(3).to_string(index=False))
    print(d.groupby("features")[["crps_once", "crps_twice", "mec_once", "mec_twice"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
