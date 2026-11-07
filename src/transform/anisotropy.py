"""Stage 2, reported beside the fits: REML with geometric anisotropy for the
four large fields.

For each large field and mean model, the family D5 chose is fitted again
with the distance between two points replaced by

    h' = sqrt(u^2 + (v / ratio)^2),   u = dx sin(theta) + dy cos(theta),
                                      v = dx cos(theta) - dy sin(theta),

so that the effective range a holds along the azimuth theta (degrees
clockwise from north, as gstat's anis) and ratio x a across it. At ratio = 1
the model is the isotropic one, so the anisotropic fit can only match or
better it, and the change in AIC, with two parameters more, says whether
the anisotropy earns them. Stage 4 adopts it only where the AIC falls by
more than 2.

    python -m src.transform.anisotropy [--budget SECONDS]
        writes curated/potash_anisotropy.parquet; reads the REML record

Bounds: the isotropic fit's, and the ratio from 0.1 to 1; the angle is
searched without a bound and reported in [0, 180). Starts: the isotropic
estimate with the angle at 0, 45, 90 and 135 degrees and the ratio at 0.5,
each refined by L-BFGS-B, with a bounded Nelder-Mead where L-BFGS-B stops
without converging. Each fit is cached as the REML run's steps are.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import linalg, optimize

try:                        # run as a module from the repository root
    from src.transform import reml as M
    from src.transform import stocks as S
except ImportError:         # imported with src/transform on sys.path (pytest)
    import reml as M
    import stocks as S

RATIO_BOUNDS = (0.1, 1.0)
ANGLES = (0.0, 45.0, 90.0, 135.0)
START_RATIO = 0.5
ADOPT_AIC = -2.0            # Stage 4 adopts anisotropy only where the AIC falls by more than 2
OUT = M.CURATED / "potash_anisotropy.parquet"
CACHE = M.CURATED / "anisotropy_cache"
_SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def rotated_distances(dx, dy, angle: float, ratio: float) -> np.ndarray:
    """h' for pair differences (dx east, dy north), the major axis at `angle`."""
    t = math.radians(angle)
    u = dx * math.sin(t) + dy * math.cos(t)
    v = dx * math.cos(t) - dy * math.sin(t)
    return np.sqrt(u * u + (v / ratio) ** 2)


@dataclass(frozen=True)
class AnisoFit:
    family: str
    mean: str
    n: int
    sill: float
    tau: float
    eff_range: float            # along the major axis
    nu: Optional[float]
    angle: float                # of the major axis, degrees clockwise from north, in [0, 180)
    ratio: float                # minor range / major range
    loglik: float
    k: int
    converged: bool
    at_bound: tuple
    message: str
    evaluations: int
    range_bounds: tuple

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.k


class _Search:
    """x = (log a, tau, [nu], angle, ratio) for one field and family."""

    def __init__(self, field: M.Field, family: str):
        self.field, self.family, self.matern = field, family, family == "matern"
        i, j = np.triu_indices(field.n, 1)          # the condensed order of pdist
        self.dx = field.xy[j, 0] - field.xy[i, 0]
        self.dy = field.xy[j, 1] - field.xy[i, 1]
        self.range_bounds = field.range_bounds()
        self.bounds = ([(math.log(self.range_bounds[0]), math.log(self.range_bounds[1])), M.TAU_BOUNDS]
                       + ([M.NU_BOUNDS] if self.matern else []) + [(None, None), RATIO_BOUNDS])
        self.names = ("eff_range", "tau") + (("nu",) if self.matern else ()) + ("angle", "ratio")
        self.count = 0

    def unpack(self, x):
        return (math.exp(x[0]), float(x[1]), float(x[2]) if self.matern else None,
                float(x[-2]), float(x[-1]))

    def profile(self, x):
        a, tau, nu, angle, ratio = self.unpack(x)
        rotated = dataclasses.replace(self.field, d=rotated_distances(self.dx, self.dy, angle, ratio))
        return M.reml_profile(rotated, self.family, a, tau, nu)

    def objective(self, x) -> float:
        self.count += 1
        try:
            ll = self.profile(x)[0]
        except (linalg.LinAlgError, np.linalg.LinAlgError, ValueError, FloatingPointError):
            return M._FAIL
        return -ll if math.isfinite(ll) else M._FAIL

    def refine(self, x0):
        r = optimize.minimize(self.objective, x0, method="L-BFGS-B", bounds=self.bounds)
        if not r.success:
            nm = optimize.minimize(self.objective, r.x, method="Nelder-Mead", bounds=self.bounds,
                                   options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
            if nm.success and nm.fun <= r.fun + 1e-9:
                r = nm
        return r


def fit_anisotropic(field: M.Field, family: str, eff_range: float, tau: float,
                    nu: Optional[float] = None) -> AnisoFit:
    """The family refitted with geometric anisotropy, started from the
    isotropic estimate (eff_range, tau, nu) at four angles."""
    s = _Search(field, family)
    base = [math.log(eff_range), float(tau)] + ([float(nu)] if s.matern else [])
    best = None
    for angle in ANGLES:
        r = s.refine(np.array(base + [angle, START_RATIO]))
        if best is None or r.fun < best.fun:
            best = r
    a, tau_hat, nu_hat, angle, ratio = s.unpack(best.x)
    ll, sill, _ = s.profile(best.x)
    at_bound = tuple(name for name, v, (lo, hi) in zip(s.names, best.x, s.bounds)
                     if lo is not None and min(v - lo, hi - v) <= 1e-6 * max(1.0, abs(v)))
    k = (3 if s.matern else 2) + 1 + 2          # shared, the sill, angle and ratio
    return AnisoFit(family, field.mean, field.n, float(sill), tau_hat, a, nu_hat, angle % 180.0,
                    ratio, float(ll), k, bool(best.success), at_bound, str(best.message), s.count,
                    tuple(s.range_bounds))


def _aniso_from(d: dict) -> AnisoFit:
    d = dict(d)
    d["at_bound"], d["range_bounds"] = tuple(d["at_bound"]), tuple(d["range_bounds"])
    return AnisoFit(**d)


COLUMNS = ["site", "mean", "family", "nu", "sill", "tau", "eff_range_major", "eff_range_minor",
           "angle", "ratio", "loglik", "loglik_iso", "aic", "aic_iso", "delta_aic", "adopt", "k",
           "converged", "at_bound", "range_bound_lo", "range_bound_hi"]


def run_stage(stocks: pd.DataFrame, record: pd.DataFrame, large=M.LARGE,
              cache: Optional[M.Cache] = None, log=lambda message: None) -> pd.DataFrame:
    """Every large field and mean model: the anisotropic refit of its own REML model."""
    cache = cache or M.Cache()
    own = record[record["fit"] == "own"].set_index(["site", "mean"])
    rows = []
    for site in [s for s in large if s in set(stocks["site"])]:
        g = stocks[stocks["site"] == site]
        for mean in M.MEANS:
            iso = own.loc[(site, mean)]
            field = M.Field.of(g[["x", "y"]].to_numpy(float), g["stock"].to_numpy(float), mean)
            nu = float(iso["nu"]) if iso["family"] == "matern" else None
            key = ["aniso", _SOURCE, M._data_key(field), iso["family"], float(iso["eff_range"]),
                   float(iso["tau"]), nu]
            f = cache.get(key, lambda: fit_anisotropic(field, iso["family"], float(iso["eff_range"]),
                                                       float(iso["tau"]), nu), _aniso_from)
            delta = f.aic - float(iso["aic"])
            rows.append({"site": site, "mean": mean, "family": f.family,
                         "nu": np.nan if f.nu is None else f.nu, "sill": f.sill, "tau": f.tau,
                         "eff_range_major": f.eff_range, "eff_range_minor": f.eff_range * f.ratio,
                         "angle": f.angle, "ratio": f.ratio, "loglik": f.loglik,
                         "loglik_iso": float(iso["loglik"]), "aic": f.aic, "aic_iso": float(iso["aic"]),
                         "delta_aic": delta, "adopt": bool(delta < ADOPT_AIC), "k": f.k,
                         "converged": f.converged, "at_bound": ",".join(f.at_bound) or "none",
                         "range_bound_lo": f.range_bounds[0], "range_bound_hi": f.range_bounds[1]})
            log(f"{site} {mean}: angle {f.angle:.0f}, ratio {f.ratio:.2f}, change in AIC {delta:+.1f}")
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv: list[str]) -> int:
    """python -m src.transform.anisotropy [--budget SECONDS]

    exit 0: written; 2: the stock table or the REML record is missing;
    4: stopped at the budget, run again to continue from the cache"""
    budget = None
    if argv:
        if len(argv) == 2 and argv[0] == "--budget":
            budget = float(argv[1])
        else:
            print(main.__doc__)
            return 2
    if not S.OUT.exists() or not M.OUT.exists():
        print("NOT THERE: python -m src.transform.stocks, then python -m src.transform.reml")
        return 2
    t0 = time.monotonic()
    cache = M.Cache(CACHE, None if budget is None else t0 + budget)
    try:
        table = run_stage(pd.read_parquet(S.OUT), pd.read_parquet(M.OUT), cache=cache,
                          log=lambda m: print(f"[{time.monotonic() - t0:4.0f} s] {m}", flush=True))
    except M.OutOfTime:
        print(f"Stopped at the {budget:g} s budget: {cache.computed} fits computed in this run, "
              f"{cache.hits} taken from the cache. Run again to continue.")
        return 4
    table.to_parquet(OUT, index=False)
    print(f"{len(table)} anisotropic fits -> {OUT.relative_to(M.ROOT).as_posix()} "
          f"({cache.computed} computed, {cache.hits} from the cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
