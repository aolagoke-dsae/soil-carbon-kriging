"""IL-BR against the soil survey (RUNBOOK, Stage 1, Addendum on the soil
survey, U2-U4; rules fixed 15.11.2026 before any SSURGO data was pulled).

    python -m src.checks.ssurgo     prints U3's table and U4's verdict;
                                    exit 0, or 2 when raw/SDA/ is incomplete

Reads raw/SDA/ (src/ingest/sda_client.py), raw/FIGSHARE/ through
Checkpoint 1's readers, and Stage 2's stock table; makes no request.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from shapely.strtree import STRtree

try:                        # run as a module from the repository root
    from src.checks import checkpoint1 as C1
    from src.ingest import sda_client as SDA
except ImportError:         # imported with src/checks and src/ingest on sys.path (pytest)
    import checkpoint1 as C1
    import sda_client as SDA

DEPTH = 30.0                # cm
OM_TO_OC = 0.58             # U2 (Pribyl 2010: conventional, and often too low)
FOCUS = "IL-BR"


def component_values(h: pd.DataFrame, depth: float = DEPTH) -> pd.DataFrame:
    """U2 per component: thickness-weighted organic matter above `depth`
    (low, representative, high) times OM_TO_OC, and the stock where both OM
    and bulk density cover the whole layer."""
    h = h.copy()
    for c in ("comppct_r", "hzdept_r", "hzdepb_r", "om_l", "om_r", "om_h", "dbthirdbar_r"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    top, bottom = h["hzdept_r"].clip(lower=0), h["hzdepb_r"].clip(upper=depth)
    h["w"] = (bottom - top).clip(lower=0).fillna(0)
    rows = []
    for (mukey, cokey), g in h.groupby(["mukey", "cokey"], sort=True):
        row = {"mukey": str(mukey), "cokey": str(cokey), "comppct_r": float(g["comppct_r"].iloc[0])}
        for k in ("om_l", "om_r", "om_h"):
            m = g["w"] > 0
            m &= g[k].notna()
            row[k.replace("om", "oc")] = (float((g.loc[m, k] * g.loc[m, "w"]).sum() / g.loc[m, "w"].sum()) * OM_TO_OC
                                          if g.loc[m, "w"].sum() > 0 else np.nan)
        full = (g["w"] > 0) & g["om_r"].notna() & g["dbthirdbar_r"].notna()
        row["stock"] = (float((g.loc[full, "om_r"] * OM_TO_OC * g.loc[full, "dbthirdbar_r"] * g.loc[full, "w"]).sum())
                        if np.isclose(g.loc[full, "w"].sum(), depth) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def mapunit_values(comp: pd.DataFrame) -> pd.DataFrame:
    """U2 per map unit: the components' values weighted by comppct_r, over the
    components that have each value, the weights renormalised."""
    out = []
    for mukey, g in comp.groupby("mukey", sort=True):
        row = {"mukey": mukey}
        for k in ("oc_l", "oc_r", "oc_h", "stock"):
            m = g[k].notna() & (g["comppct_r"] > 0)
            row[k] = float(np.average(g.loc[m, k], weights=g.loc[m, "comppct_r"])) if m.any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def locate(points: pd.DataFrame, polys: pd.DataFrame) -> pd.Series:
    """U3: the mukey of the polygon containing each (lon, lat); NaN where none does."""
    geoms = [shapely_wkt.loads(w) for w in polys["wkt"]]
    tree = STRtree(geoms)
    keys = polys["mukey"].astype(str).to_numpy()
    out = []
    for lon, lat in zip(points["lon"], points["lat"]):
        p = Point(lon, lat)
        hit = [i for i in tree.query(p) if geoms[i].covers(p)]
        out.append(keys[hit[0]] if hit else np.nan)
    return pd.Series(out, index=points.index, dtype=object)


def observed_soc(meas: pd.DataFrame, depth: float = DEPTH) -> pd.Series:
    """The thickness-weighted SOCc (%) in 0-depth cm per (site, location_id)."""
    m = meas.copy()
    for c in ("sample_depth_min", "sample_depth_max", "SOCc"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m[m["sample_depth_min"] < depth]
    m["w"] = m["sample_depth_max"].clip(upper=depth) - m["sample_depth_min"]
    g = m.assign(x=m["SOCc"] * m["w"]).groupby(C1.KEY)
    return g["x"].sum() / g["w"].sum()


def compare(stocks: pd.DataFrame, meas: pd.DataFrame, polys: pd.DataFrame, horizons: pd.DataFrame) -> pd.DataFrame:
    """U3, one row per field, over the locations whose map unit has a value."""
    mu = mapunit_values(component_values(horizons)).set_index("mukey")
    t = stocks.copy()
    t["mukey"] = locate(t, polys)
    obs = observed_soc(meas)
    t["soc_obs"] = [obs.get((s, l), np.nan) for s, l in zip(t["site"], t["location_id"])]
    for k in ("oc_l", "oc_r", "oc_h"):
        t[k] = t["mukey"].map(mu[k])
    t["stock_ssurgo"] = t["mukey"].map(mu["stock"])
    rows = []
    for site, g in t.groupby("site", sort=True):
        g = g[g["oc_r"].notna()]
        rows.append({"site": site, "locations": int(len(t[t["site"] == site])), "matched": len(g),
                     "mapunits": int(g["mukey"].nunique()),
                     "soc_obs": float(g["soc_obs"].mean()), "soc_ssurgo": float(g["oc_r"].mean()),
                     "soc_low": float(g["oc_l"].mean()), "soc_high": float(g["oc_h"].mean()),
                     "stock_obs": float(g["stock"].mean()), "stock_ssurgo": float(g["stock_ssurgo"].mean())})
    out = pd.DataFrame(rows)
    out["ratio"] = out["soc_obs"] / out["soc_ssurgo"]
    out["within"] = (out["soc_obs"] >= out["soc_low"]) & (out["soc_obs"] <= out["soc_high"])
    return out


def verdict(out: pd.DataFrame, focus: str = FOCUS) -> tuple[bool, str]:
    """U4: the focus field's ratio against the range of the others'."""
    others = out[out["site"] != focus]["ratio"]
    r = float(out.set_index("site").loc[focus, "ratio"])
    lo, hi = float(others.min()), float(others.max())
    ok = lo <= r <= hi
    word = "CONSISTENT with" if ok else "INCONSISTENT with"
    return ok, (f"{focus}: observed / SSURGO = {r:.2f}, the other seven {lo:.2f} to {hi:.2f}: "
                f"{word} the soil survey (U4)")


def main(argv: list[str]) -> int:
    if argv:
        print(__doc__)
        return 2
    try:
        doc, files = C1.potash()
    except C1.NotPulled as e:
        print(f"NOT PULLED: {e}")
        return 2
    stocks = pd.read_parquet(SDA.STOCKS)
    sites = sorted(stocks["site"].unique())
    polys = [SDA.stored(SDA.polygons_item(s), SDA.POLYGON_COLUMNS) for s in sites]
    horizons = SDA.stored(SDA.horizons_item(), SDA.HORIZON_COLUMNS)
    catalog = SDA.stored(SDA.catalog_item(), SDA.CATALOG_COLUMNS)
    if any(p is None for p in polys) or horizons is None or catalog is None:
        print("NOT PULLED: raw/SDA/ is incomplete (python -m src.ingest.sda_client)")
        return 2
    _, _, meas = C1.check_potash(doc, files)
    out = compare(stocks, meas, pd.concat(polys, ignore_index=True), horizons)
    pd.set_option("display.width", 220)
    print(out.round(2).to_string(index=False))
    print()
    print("survey areas: " + ", ".join(f"{a} saved {d}" for a, d in zip(catalog["areasymbol"], catalog["saverest"])))
    print(verdict(out)[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
