"""Stage 4, D1: the cross-validation folds.

    python -m src.transform.folds    writes curated/potash_folds.parquet

Ten folds per field, assigned at random within each field and repeated three
times with different assignments. Each repeat and field draws its own
permutation of the field's locations, sorted by location_id, from a seed
recorded beside it, so the folds can be rebuilt from the table alone. A
location's fold is its position in the permutation modulo ten, so the folds
of a field differ in size by at most one. Fold k of every field is held out
together: the forests of D7, trained across the fields, then never see a
held-out stock.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPEATS = 3
FOLDS = 10
SEED = 20261108
ROOT = Path(__file__).resolve().parents[2]
STOCKS = ROOT / "curated" / "potash_stocks_0_30.parquet"
OUT = ROOT / "curated" / "potash_folds.parquet"
COLUMNS = ["site", "location_id", "repeat", "fold", "seed"]


def seed_of(repeat: int, site_index: int) -> int:
    """The seed of one repeat's permutation of one field (sites in sorted order)."""
    return SEED + 1000 * repeat + site_index


def assign(stocks: pd.DataFrame, repeats: int = REPEATS, folds: int = FOLDS) -> pd.DataFrame:
    """One row per repeat and location: its fold and the seed that drew it."""
    rows = []
    for i, site in enumerate(sorted(stocks["site"].unique())):
        ids = sorted(stocks.loc[stocks["site"] == site, "location_id"])
        for r in range(repeats):
            seed = seed_of(r, i)
            perm = np.random.default_rng(seed).permutation(len(ids))
            fold = np.empty(len(ids), dtype=int)
            fold[perm] = np.arange(len(ids)) % folds
            rows += [{"site": site, "location_id": lid, "repeat": r, "fold": int(f), "seed": seed}
                     for lid, f in zip(ids, fold)]
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv: list[str]) -> int:
    """python -m src.transform.folds"""
    if argv:
        print(main.__doc__)
        return 2
    if not STOCKS.exists():
        print("NOT THERE: the stock table: python -m src.transform.stocks")
        return 2
    table = assign(pd.read_parquet(STOCKS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT, index=False)
    sizes = table.groupby(["site", "repeat", "fold"]).size()
    print(f"{OUT.relative_to(ROOT).as_posix()}: {table['site'].nunique()} fields, {REPEATS} repeats of "
          f"{FOLDS} folds, {sizes.min()} to {sizes.max()} locations per fold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
