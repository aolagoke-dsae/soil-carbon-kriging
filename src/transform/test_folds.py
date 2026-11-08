"""Stage 4's folds: a partition per field and repeat, reproducible from its seeds."""
import numpy as np
import pandas as pd

import folds as F


def stocks():
    return pd.DataFrame({"site": ["A"] * 23 + ["B"] * 40,
                         "location_id": [f"a{i:02d}" for i in range(23)] + [f"b{i:02d}" for i in range(40)]})


def test_folds_partition_each_field_and_differ_between_repeats():
    t = F.assign(stocks())
    assert len(t) == 3 * 63
    for (site, r), g in t.groupby(["site", "repeat"]):
        sizes = g["fold"].value_counts()
        assert set(sizes.index) == set(range(10))
        assert sizes.max() - sizes.min() <= 1
        assert g["location_id"].is_unique
    a = t[t["site"] == "B"].pivot(index="location_id", columns="repeat", values="fold")
    assert not (a[0] == a[1]).all() and not (a[1] == a[2]).all()


def test_the_recorded_seed_rebuilds_the_folds():
    t = F.assign(stocks())
    g = t[(t["site"] == "A") & (t["repeat"] == 2)].sort_values("location_id")
    perm = np.random.default_rng(int(g["seed"].iloc[0])).permutation(len(g))
    fold = np.empty(len(g), dtype=int)
    fold[perm] = np.arange(len(g)) % 10
    assert list(g["fold"]) == list(fold)
