"""Stage 2, D1 and D2: the 0-30 cm SOC stock at every Potash location whose
segments run from the surface to 30 cm without a gap or an overlap.

    python -m src.transform.stocks     writes curated/potash_stocks_0_30.parquet

D1  stock (Mg C/ha) = the sum over the segments above 30 cm of
        SOCc (%) x BD (g/cm3) x thickness (cm);
    the units cancel: 1 % x 1 g/cm3 x 1 cm = 0.01 g C/cm2 = 1 Mg C/ha.
    No coarse-fragment correction: the files give none. A segment that
    crosses 30 cm would need splitting, which D1 does not do. The files have
    none, and one would stop the run rather than be prorated.
D2  A location without contiguous segments from 0 to 30 cm is left out, not
    filled in. `checkpoint1.covers` decides: the test Checkpoint 1 counted
    the 1,163 complete locations with.

Coordinates: X and Y are longitude and latitude, read as WGS 84. Each field
is projected to WGS 84 / UTM in the zone of its mean longitude, so that REML
works in metres. IL-BR lies just west of 90 degrees W and falls in zone 15,
the other Illinois fields in 16, NE in 14.

Reads raw/ through Checkpoint 1's readers and refuses to run unless
Checkpoint 1's Potash rules pass. Makes no request.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

try:                        # run as a module from the repository root
    from src.checks import checkpoint1 as C1
except ImportError:         # imported with src/checks on sys.path (pytest)
    import checkpoint1 as C1

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "curated" / "potash_stocks_0_30.parquet"
DEPTH_CM = C1.STOCK_DEPTH_CM
KEY = C1.KEY
COLUMNS = ["site", "location_id", "lon", "lat", "epsg", "x", "y", "stock", "segments"]


def utm_epsg(lon: float) -> int:
    """EPSG code of WGS 84 / UTM north for a longitude. All eight fields lie
    north of the equator."""
    return 32600 + int((lon + 180.0) // 6.0) + 1


def complete(loc: pd.DataFrame, meas: pd.DataFrame,
             depth: float = DEPTH_CM) -> list[tuple[str, str]]:
    """The (site, location_id) whose segments cover 0 to `depth` cm (D2), in
    the order of locations.csv."""
    by = {k: g for k, g in meas.groupby(KEY)}
    return [k for k in zip(loc["site"], loc["location_id"])
            if k in by and C1.covers(by[k], depth)]


def stocks(loc: pd.DataFrame, meas: pd.DataFrame, depth: float = DEPTH_CM) -> pd.Series:
    """D1 for the complete locations: one stock per key, Mg C/ha."""
    keys = pd.MultiIndex.from_tuples(complete(loc, meas, depth), names=KEY)
    m = meas.set_index(KEY)
    m = m[m.index.isin(keys)]
    top = m["sample_depth_min"].astype(float)
    bottom = m["sample_depth_max"].astype(float)
    above = top < depth
    crossing = above & (bottom > depth)
    if crossing.any():
        raise ValueError(f"segments cross {depth:g} cm at {sorted(set(m.index[crossing]))}; "
                         "D1 sums whole segments only")
    seg = m["SOCc"].astype(float) * m["BD"].astype(float) * (bottom - top)
    return seg[above].groupby(level=KEY).sum().reindex(keys)


def stock_table(loc: pd.DataFrame, meas: pd.DataFrame, depth: float = DEPTH_CM) -> pd.DataFrame:
    """One row per complete location: keys, lon/lat, the field's UTM EPSG
    code and x/y in it (m), the stock (Mg C/ha) and its segments."""
    s = stocks(loc, meas, depth)
    by = {k: g for k, g in meas.groupby(KEY)}
    t = loc.set_index(KEY).loc[s.index].reset_index()
    t["lon"] = pd.to_numeric(t["X"])
    t["lat"] = pd.to_numeric(t["Y"])
    t["stock"] = s.to_numpy()
    t["segments"] = [C1.scheme(by[k], depth) for k in s.index]
    parts = []
    for site, g in t.groupby("site", sort=True):
        epsg = utm_epsg(float(g["lon"].mean()))
        x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(
            g["lon"].to_numpy(), g["lat"].to_numpy())
        parts.append(g.assign(epsg=epsg, x=np.asarray(x, float), y=np.asarray(y, float)))
    out = pd.concat(parts, ignore_index=True)[COLUMNS]
    return out.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def main(argv: list[str]) -> int:
    """python -m src.transform.stocks"""
    if argv:
        print(main.__doc__)
        return 2
    try:
        doc, files = C1.potash()
    except C1.NotPulled as e:
        print(f"NOT PULLED: {e}")
        return 2
    rules, loc, meas = C1.check_potash(doc, files)
    if not all(r.passed for r in rules):
        print("Checkpoint 1's Potash rules do not pass: python -m src.checks.checkpoint1")
        return 1
    t = stock_table(loc, meas)
    OUT.parent.mkdir(exist_ok=True)
    t.to_parquet(OUT, index=False)
    print(f"{len(t)} stocks in 0-{DEPTH_CM} cm -> {OUT.relative_to(ROOT).as_posix()}")
    summary = t.groupby("site").agg(stocks=("stock", "size"), epsg=("epsg", "first"),
                                    mean=("stock", "mean"), sd=("stock", "std"))
    print(summary.round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
