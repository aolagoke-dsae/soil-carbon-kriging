"""Make the package layout importable from a plain `pytest` at the repo root,
and share one small stand-in for the Stage 2 REML run between the tests."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
for _p in ("src/ingest", "src/checks", "src/transform"):
    sys.path.insert(0, str(ROOT / _p))


@pytest.fixture(scope="session")
def small_stage():
    """Three hand-made fields (two large, one small) run through Stage 2's
    REML steps with the exponential family only."""
    import numpy as np
    import pandas as pd
    import reml as M
    rng = np.random.default_rng(21)
    rows = []
    for site, n in (("A", 45), ("B", 40), ("C", 30)):
        xy, z = M.simulate(rng, n, (300.0, 250.0), "exponential", 25.0, 90.0, 0.3, mean=50.0)
        rows += [{"site": site, "location_id": str(i), "x": xy[i, 0], "y": xy[i, 1], "stock": z[i]}
                 for i in range(n)]
    t = pd.DataFrame(rows)
    return t, M.run_stage(t, families=("exponential",), large=("A", "B"), small=("C",))
