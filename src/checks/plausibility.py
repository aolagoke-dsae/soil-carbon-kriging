"""Review of v1.0, M5: the plausibility of the Potash segments (RUNBOOK,
Stage 1, Addendum, Q1-Q5; rules fixed 15.11.2026, after the review had seen
IL-BR's bulk densities, and applied to all eight fields alike).

    python -m src.checks.plausibility     prints Q2-Q5 and exits 0; it
                                          changes no stock and drops nothing

Reads raw/ through Checkpoint 1's readers; makes no request.
"""
from __future__ import annotations

import sys

import pandas as pd

try:                        # run as a module from the repository root
    from src.checks import checkpoint1 as C1
    from src.transform import stocks as S
except ImportError:         # imported with src/checks and src/transform on sys.path (pytest)
    import checkpoint1 as C1
    import stocks as S

BD_LOOSE = 0.9              # g/cm3 (Q1)
SOC_LOW = 2.0               # % (Q1)
VERDICT_MG = 1.0            # Mg C/ha (Q5)
# Table 1 of Potash et al. (2023), "Sample size", read 15.11.2026 (Q4)
TABLE1_N = {"IL-BR": 59, "IL-DG": 86, "IL-MC": 50, "IL-PT": 89,
            "IL-RD": 247, "IL-RS": 223, "IL-RT": 229, "NE": 144}


def above(meas: pd.DataFrame, depth: float = C1.STOCK_DEPTH_CM) -> pd.DataFrame:
    """The segments that start above `depth`, with numeric BD and SOCc."""
    m = meas.copy()
    for c in ("sample_depth_min", "sample_depth_max", "SOCc", "BD"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m[m["sample_depth_min"] < depth]


def flag(seg: pd.DataFrame) -> pd.Series:
    """Q1: loose and low in carbon."""
    return (seg["BD"] < BD_LOOSE) & (seg["SOCc"] < SOC_LOW)


def per_field(meas: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Q2-Q4, one row per field. `table` is Stage 2's stock table."""
    seg = above(meas)
    seg = seg.assign(flagged=flag(seg))
    flagged_keys = set(map(tuple, seg.loc[seg["flagged"], C1.KEY].to_numpy()))
    keys = list(zip(table["site"], table["location_id"]))
    t = table.assign(flagged=[k in flagged_keys for k in keys])
    rows = []
    for site, g in t.groupby("site", sort=True):
        sg = seg[seg["site"] == site]
        clean = g.loc[~g["flagged"], "stock"]
        rows.append({
            "site": site,
            "segments": len(sg), "flagged": int(sg["flagged"].sum()),
            "stocks": len(g), "stocks_flagged": int(g["flagged"].sum()),
            "soc_median": float(sg["SOCc"].median()), "bd_median": float(sg["BD"].median()),
            "mean_all": float(g["stock"].mean()),
            "mean_unflagged": float(clean.mean()) if len(clean) else float("nan"),
            "table1_n": TABLE1_N.get(site),
        })
    out = pd.DataFrame(rows)
    out["shift"] = out["mean_unflagged"] - out["mean_all"]
    return out


def verdict(row: pd.Series) -> str:
    """Q5 for one field."""
    if abs(row["shift"]) < VERDICT_MG:
        return (f"{row['site']}: the flagged segments move the sample mean by {row['shift']:+.2f} Mg C/ha, "
                f"less than {VERDICT_MG:g}: its level is the carbon content's, not the bulk density's. "
                "Reported as low in carbon in the published data, unconfirmed by the authors")
    return (f"{row['site']}: the flagged segments move the sample mean by {row['shift']:+.2f} Mg C/ha: "
            "its field mean is provisional")


def main(argv: list[str]) -> int:
    if argv:
        print(__doc__)
        return 2
    try:
        doc, files = C1.potash()
    except C1.NotPulled as e:
        print(f"NOT PULLED: {e}")
        return 2
    _, loc, meas = C1.check_potash(doc, files)
    table = S.stock_table(loc, meas)
    out = per_field(meas, table)
    pd.set_option("display.width", 200)
    print(out.round(2).to_string(index=False))
    print()
    print(verdict(out[out["site"] == "IL-BR"].iloc[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
