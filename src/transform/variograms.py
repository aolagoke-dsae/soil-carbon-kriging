"""Stage 2, D8: the moment-method baseline, gstat as the author ran it in R.

    python -m src.transform.variograms   writes curated/potash_variograms.parquet,
                                         the semivariances, and beside it
                                         curated/potash_variogram_fits.parquet

For each field and each of D4's mean models the values are the stocks for
(a), and for (b) their ordinary-least-squares residuals from the
second-order trend, as gstat's variogram() takes them for ordinary and for
universal kriging.

Bins (D8) are centred on multiples of the field's median nearest-neighbour
spacing w, with edges 0, w/2, 3w/2, ..., and the last edge no further than
half the largest pairwise distance, so that no pair in any bin lies further
apart than that. A pair at distance d falls in the bin [lo, hi) that holds
it. A bin's lag is the mean distance of its pairs, as gstat reports it.

Estimators: Matheron's, gamma = sum (z_i - z_j)^2 / (2N), and beside it the
robust estimator of Cressie and Hawkins (1980; Lark 2000),
gamma = (mean |z_i - z_j|^(1/2))^4 / (2 (0.457 + 0.494/N + 0.045/N^2)).
Omnidirectional, and in four directions, 0, 45, 90 and 135 degrees
clockwise from north with 22.5 degrees either side, as gstat's alpha and
tol.hor.

Fits (D8): spherical and exponential, by weighted least squares with gstat's
default weights N_j / h_j^2 (fit.method = 7), over the omnidirectional bins
of 30 pairs or more:

    gamma(h) = c0 + c g(h; a),  spherical    g = 1.5 h/a - 0.5 (h/a)^3, h < a; 1 beyond
                                exponential  g = 1 - exp(-h/a)

with nugget c0 >= 0, partial sill c > 0 and 0 < a <= 2 x the largest lag;
a is gstat's range parameter, the spherical's range and the exponential's
distance parameter. For a given a the model is linear in (c0, c), solved
exactly by non-negative least squares, so the search runs over a alone: a
grid, then Brent's method between the best point's neighbours. "c > 0" is
c >= 1e-9 x the largest semivariance and "0 < a" is a >= 0.001 x the
largest lag. The effective range is reported as REML defines it: a for the
spherical, a ln 20 for the exponential. The family with the lower weighted
sum of squares is the baseline's model. Stage 4 does not krig with it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

try:                        # run as a module from the repository root
    from src.transform import reml as M
    from src.transform import stocks as S
except ImportError:         # imported with src/transform on sys.path (pytest)
    import reml as M
    import stocks as S

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "curated" / "potash_variograms.parquet"
OUT_FITS = ROOT / "curated" / "potash_variogram_fits.parquet"
MIN_PAIRS = 30                      # D8: bins of fewer pairs are not fitted
FAMILIES = ("spherical", "exponential")
DIRECTIONS = (0.0, 45.0, 90.0, 135.0)   # degrees clockwise from north
TOLERANCE = 22.5                    # degrees either side
A_GRID = 200
A_MIN_FACTOR = 1e-3                 # "0 < a": a >= 0.001 x the largest lag
A_MAX_FACTOR = 2.0                  # D8: a <= 2 x the largest lag
C_FLOOR = 1e-9                      # "c > 0": c >= 1e-9 x the largest semivariance
LN20 = math.log(20.0)


def values(xy, z, mean: str) -> np.ndarray:
    """(a) the stocks; (b) their OLS residuals from the second-order trend."""
    z = np.asarray(z, dtype=float)
    if mean == "constant":
        return z
    X = M.design(xy, "quadratic")
    return z - X @ np.linalg.lstsq(X, z, rcond=None)[0]


def edges(xy) -> np.ndarray:
    """D8's edges: 0, w/2, 3w/2, ..., none beyond half the largest separation."""
    xy = np.asarray(xy, dtype=float)
    w = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    half = 0.5 * float(pdist(xy).max())
    e = [0.0]
    k = 1
    while (k - 0.5) * w <= half:
        e.append((k - 0.5) * w)
        k += 1
    return np.array(e)


def pairs(xy, v):
    """Every pair once: distance, azimuth in [0, 180) degrees clockwise from
    north, and the difference of the values."""
    xy = np.asarray(xy, dtype=float)
    i, j = np.triu_indices(len(v), 1)
    dx, dy = xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]
    return np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy)) % 180.0, v[j] - v[i]


def cressie_hawkins(diff: np.ndarray) -> float:
    n = diff.size
    return float(np.mean(np.sqrt(np.abs(diff))) ** 4 / (2.0 * (0.457 + 0.494 / n + 0.045 / n ** 2)))


def in_sector(azimuth: np.ndarray, direction: float) -> np.ndarray:
    """Pairs within 22.5 degrees of `direction`, either way; the four sectors
    partition the pairs."""
    off = (azimuth - direction + 90.0) % 180.0 - 90.0
    return (off >= -TOLERANCE) & (off < TOLERANCE)


def semivariances(d, diff, e) -> list[dict]:
    """Per bin of edges e: pairs, lag, Matheron's and Cressie-Hawkins' gamma."""
    b = np.searchsorted(e, d, side="right") - 1
    rows = []
    for k in range(len(e) - 1):
        sel = b == k
        n = int(sel.sum())
        rows.append({"bin": k, "lo": float(e[k]), "hi": float(e[k + 1]), "pairs": n,
                     "lag": float(d[sel].mean()) if n else np.nan,
                     "gamma": float((diff[sel] ** 2).sum() / (2 * n)) if n else np.nan,
                     "gamma_ch": cressie_hawkins(diff[sel]) if n else np.nan})
    return rows


def shape(family: str, h, a: float) -> np.ndarray:
    """g(h; a), the model's structured part rising from 0 to 1."""
    h = np.asarray(h, dtype=float)
    if family == "spherical":
        x = np.minimum(h / a, 1.0)
        return x * (1.5 - 0.5 * x * x)
    if family == "exponential":
        return 1.0 - np.exp(-h / a)
    raise ValueError(f"unknown family {family!r}")


def _linear(family, h, gamma, w, a, floor):
    """For one a: the weighted least-squares (c0, c), c0 >= 0 and c >= floor."""
    g = shape(family, h, a)
    sw = np.sqrt(w)
    (c0, extra), _ = optimize.nnls(np.column_stack([sw, sw * g]), sw * (gamma - floor * g))
    c = floor + extra
    resid = gamma - c0 - c * g
    return float(w @ (resid * resid)), float(c0), float(c)


def fit(family: str, h, gamma, n) -> dict:
    """D8 for one family over the fitted bins (lags h, semivariances, pairs)."""
    h, gamma, n = (np.asarray(x, dtype=float) for x in (h, gamma, n))
    w = n / h ** 2
    largest = float(h.max())
    floor = C_FLOOR * float(gamma.max())
    lo, hi = A_MIN_FACTOR * largest, A_MAX_FACTOR * largest
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), A_GRID))
    wss = np.array([_linear(family, h, gamma, w, a, floor)[0] for a in grid])
    i = int(np.argmin(wss))
    b_lo, b_hi = math.log(grid[max(i - 1, 0)]), math.log(grid[min(i + 1, A_GRID - 1)])
    r = optimize.minimize_scalar(lambda la: _linear(family, h, gamma, w, math.exp(la), floor)[0],
                                 bounds=(b_lo, b_hi), method="bounded", options={"xatol": 1e-10})
    a = math.exp(r.x) if r.fun <= wss[i] else float(grid[i])
    a = min(max(a, lo), hi)
    total, c0, c = _linear(family, h, gamma, w, a, floor)
    at_bound = [name for name, flag in (("a", a <= lo * (1 + 1e-6) or a >= hi * (1 - 1e-6)),
                                        ("nugget", c0 == 0.0), ("partial_sill", c <= floor * (1 + 1e-9)))
                if flag]
    return {"family": family, "nugget": c0, "partial_sill": c, "sill": c0 + c,
            "tau": c0 / (c0 + c), "a": a,
            "eff_range": a if family == "spherical" else a * LN20,
            "a_min": lo, "a_max": hi, "largest_lag": largest, "bins_fitted": int(len(h)),
            "wss": total, "converged": bool(r.success), "at_bound": ",".join(at_bound) or "none"}


def field_variograms(site: str, mean: str, xy, z) -> tuple[list[dict], list[dict]]:
    """One field and mean model: its semivariance rows and its two fits."""
    v = values(xy, z, mean)
    e = edges(xy)
    d, azimuth, diff = pairs(xy, v)
    rows = []
    for direction, sel in [("omni", np.ones_like(d, dtype=bool))] + \
                          [(f"{a:g}", in_sector(azimuth, a)) for a in DIRECTIONS]:
        for r in semivariances(d[sel], diff[sel], e):
            r.update(site=site, mean=mean, direction=direction,
                     fitted=bool(direction == "omni" and r["pairs"] >= MIN_PAIRS))
            rows.append(r)
    omni = [r for r in rows if r["fitted"]]
    h = [r["lag"] for r in omni]
    g = [r["gamma"] for r in omni]
    n = [r["pairs"] for r in omni]
    fits = [dict(fit(f, h, g, n), site=site, mean=mean) for f in FAMILIES]
    best = min(fits, key=lambda x: x["wss"])
    for f in fits:
        f["chosen"] = f is best
    return rows, fits


COLUMNS = ["site", "mean", "direction", "bin", "lo", "hi", "pairs", "lag", "gamma", "gamma_ch", "fitted"]
FIT_COLUMNS = ["site", "mean", "family", "nugget", "partial_sill", "sill", "tau", "a", "eff_range",
               "a_min", "a_max", "largest_lag", "bins_fitted", "wss", "converged", "at_bound", "chosen"]


def run_stage(stocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, fits = [], []
    for site, g in stocks.groupby("site"):
        for mean in M.MEANS:
            r, f = field_variograms(site, mean, g[["x", "y"]].to_numpy(float), g["stock"].to_numpy(float))
            rows += r
            fits += f
    return pd.DataFrame(rows, columns=COLUMNS), pd.DataFrame(fits, columns=FIT_COLUMNS)


def main(argv: list[str]) -> int:
    """python -m src.transform.variograms"""
    if argv:
        print(main.__doc__)
        return 2
    if not S.OUT.exists():
        print("NOT THERE: the stock table: python -m src.transform.stocks")
        return 2
    vario, fits = run_stage(pd.read_parquet(S.OUT))
    vario.to_parquet(OUT, index=False)
    fits.to_parquet(OUT_FITS, index=False)
    print(f"{len(vario)} semivariances -> {OUT.relative_to(ROOT).as_posix()}; "
          f"{len(fits)} fits -> {OUT_FITS.relative_to(ROOT).as_posix()}")
    chosen = fits[fits["chosen"]]
    for _, x in chosen.iterrows():
        print(f"  {x['site']:6s} {x['mean']:9s} {x['family']:11s} effective range {x['eff_range']:6.0f} m, "
              f"nugget/sill {x['tau']:.2f}, {x['bins_fitted']} bins, at bound: {x['at_bound']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
