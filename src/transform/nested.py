"""Review of v1.0, M1: each held-out fold's mean model chosen on the other
folds' scores (RUNBOOK, Stage 4, Addendum on M1, N1-N4; rules fixed
15.11.2026 before this was written).

    python -m src.transform.nested
        writes curated/potash_cv_nested.parquet (the predictions of N1's
        kriging, method "kriging_n") and prints N2-N4

Nothing is refitted: it reads the cross-validation's record
(curated/potash_cv.parquet) and Stage 4's selection.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:                        # run as a module from the repository root
    from src.transform import kriging as K
except ImportError:         # imported with src/transform on sys.path (pytest)
    import kriging as K

CV = K.CURATED / "potash_cv.parquet"
OUT = K.CURATED / "potash_cv_nested.parquet"
MODELS = ("a", "b", "c")
TIE = 0.01                  # N3: a difference below 1 % of the larger CRPS


def choices(cv: pd.DataFrame) -> pd.DataFrame:
    """N1: per (site, repeat, fold), the model with the lowest mean CRPS over
    the site's locations in the other folds of the same repeat."""
    k = cv[cv["method"].isin([f"kriging_{m}" for m in MODELS])]
    s = k.groupby(["site", "repeat", "fold", "method"])["crps"].agg(["sum", "count"]).reset_index()
    rows = []
    for (site, rep), g in s.groupby(["site", "repeat"], sort=True):
        tot = g.groupby("method")[["sum", "count"]].sum()
        for fold, gf in g.groupby("fold"):
            rest = tot - gf.set_index("method")[["sum", "count"]].reindex(tot.index, fill_value=0)
            mean = (rest["sum"] / rest["count"]).sort_index()
            rows.append({"site": site, "repeat": rep, "fold": fold, "chosen": mean.idxmin().split("_")[1]})
    return pd.DataFrame(rows)


def nested_predictions(cv: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    """N1's kriging: each location's prediction under its fold's choice."""
    k = cv[cv["method"].isin([f"kriging_{m}" for m in MODELS])].merge(chosen, on=["site", "repeat", "fold"])
    out = k[k["method"] == "kriging_" + k["chosen"]].drop(columns="chosen").copy()
    out["method"] = "kriging_n"
    return out


def field_scores(cv: pd.DataFrame, method: str) -> pd.DataFrame:
    """N2 per field: CRPS and MEC per repeat, averaged; PICP 90 % pooled."""
    g = cv[cv["method"] == method]
    rows = []
    for site, gs in g.groupby("site", sort=True):
        crps = gs.groupby("repeat")["crps"].mean().mean()
        mec = float(np.mean([1 - ((d["y"] - d["mean"]) ** 2).sum() / ((d["y"] - d["y"].mean()) ** 2).sum()
                             for _, d in gs.groupby("repeat")]))
        picp = ((gs["y"] >= gs["q0.050"]) & (gs["y"] <= gs["q0.950"])).mean()
        rows.append({"site": site, "method": method, "crps": crps, "mec": mec, "picp90": picp})
    return pd.DataFrame(rows)


def pooled(cv: pd.DataFrame, method: str) -> dict:
    """N2 pooled: the largest |QCP - level| and MSDR (per repeat, averaged)."""
    g = cv[cv["method"] == method]
    levels = [c for c in g.columns if c.startswith("q0.") or c.startswith("q1.")]
    dep = max(abs((g["y"] < g[c]).mean() - float(c[1:])) for c in levels)
    msdr = g.assign(r=(g["y"] - g["mean"]) ** 2 / g["var"]).groupby("repeat")["r"].mean().mean()
    return {"method": method, "qcp_max_departure": dep, "msdr": msdr}


def outcome(a: float, b: float, tie: float = TIE) -> str:
    """N3 for the first against the second: win, tie or loss (lower CRPS wins)."""
    if abs(a - b) < tie * max(a, b):
        return "tie"
    return "win" if a < b else "loss"


def run(cv: pd.DataFrame, selected: pd.Series) -> dict:
    chosen = choices(cv)
    nested = nested_predictions(cv, chosen)
    all_cv = pd.concat([cv, nested], ignore_index=True)
    sel = cv[cv["method"].isin([f"kriging_{m}" for m in MODELS])].copy()
    sel = sel[sel["method"] == "kriging_" + sel["site"].map(selected)].assign(method="kriging_d5")
    all_cv = pd.concat([all_cv, sel], ignore_index=True)
    methods = ["kriging_n", "kriging_d5"] + [f"kriging_{m}" for m in MODELS] + ["qrf", "rfrk"]
    scores = pd.concat([field_scores(all_cv, m) for m in methods], ignore_index=True)
    wide = scores.pivot(index="site", columns="method", values="crps")
    n3 = pd.DataFrame({"vs_qrf": [outcome(a, b) for a, b in zip(wide["kriging_n"], wide["qrf"])],
                       "vs_rfrk": [outcome(a, b) for a, b in zip(wide["kriging_n"], wide["rfrk"])]},
                      index=wide.index)
    stability = chosen.groupby(["site", "chosen"]).size().unstack(fill_value=0).reindex(columns=list(MODELS),
                                                                                        fill_value=0)
    pool = pd.DataFrame([pooled(all_cv, m) for m in ("kriging_n", "kriging_d5", "plugin_a")])
    return {"nested": nested, "scores": scores, "wide": wide, "n3": n3, "stability": stability, "pooled": pool}


def main(argv: list[str]) -> int:
    if argv:
        print(__doc__)
        return 2
    for p in (CV, K.OUT_SELECTION):
        if not p.exists():
            print(f"NOT THERE: {p.relative_to(K.ROOT).as_posix()}")
            return 2
    cv = pd.read_parquet(CV)
    selected = pd.read_parquet(K.OUT_SELECTION).set_index("site")["selected"]
    r = run(cv, selected)
    r["nested"].to_parquet(OUT, index=False)
    pd.set_option("display.width", 220)
    print("N2, CRPS (Mg C/ha):")
    print(r["wide"].round(3).to_string())
    print("\nN2, MEC and PICP 90 %:")
    print(r["scores"].pivot(index="site", columns="method", values="mec").round(2).to_string())
    print(r["scores"].pivot(index="site", columns="method", values="picp90").round(3).to_string())
    print("\nN3, N1's kriging against the forests:")
    print(r["n3"].to_string())
    for col in ("vs_qrf", "vs_rfrk"):
        print(col, r["n3"][col].value_counts().to_dict())
    print("\nN4, choices over the 30 (repeat, fold) pairs:")
    print(r["stability"].to_string())
    print("\nN2 pooled:")
    print(r["pooled"].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
