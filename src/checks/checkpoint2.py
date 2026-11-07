"""Checkpoint 2: the stock table and the covariance models (Potash fields).

The rules were fixed in RUNBOOK.md on 05.11.2026, before any variogram was
computed, and each rule built so far is one function here: S1 and S2 for the
stock table, R1 and R2 for the REML code. R3, R4 and V1-V3 belong to the
fits, their record and the moment baseline; until those are built they are
reported as not built, and the checkpoint cannot pass.

    python -m src.checks.checkpoint2    exit 0: every rule built and passed
                                        exit 1: a rule failed
                                        exit 2: raw/ or the stock table is missing
                                        exit 3: no rule failed, some not built yet

Reads raw/ and curated/; makes no request. R2 fits 50 simulated fields and
takes about half a minute.
"""
from __future__ import annotations

import datetime as dt
import math
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

try:                        # run as a module from the repository root
    from src.checks import checkpoint1 as C1
    from src.transform import reml as M
    from src.transform import stocks as S
except ImportError:         # imported with src/checks and src/transform on sys.path (pytest)
    import checkpoint1 as C1
    import reml as M
    import stocks as S

Rule = C1.Rule
KEY = C1.KEY

# S1 and S2
N_STOCKS = 1163
INCOMPLETE = frozenset({          # Checkpoint 1's seven, listed so that S1 does not lean on covers()
    ("IL-BR", "516-1"), ("IL-BR", "520-3"), ("IL-BR", "521-3"), ("IL-PT", "1"),
    ("IL-RD", "RD-A1"), ("IL-RT", "RT-B12"), ("IL-RT", "RT-M6")})
TOL_STOCK = 1e-9                  # Mg C/ha

# R1: parameters set from each field's geometry and sample variance, never fitted
TOL_R1 = 1e-8                     # relative
R1_RANGES = (0.1, 0.3)            # x the field's largest separation
R1_TAUS = (0.1, 0.5)
R1_NUS = (0.3, 0.5, 1.5, 2.2)     # scikit-learn's closed forms (0.5, 1.5) and its Bessel path

# R2, as the rule states it
R2_FIELDS, R2_POINTS, R2_EXTENT = 50, 400, (600.0, 400.0)
R2_TRUTH = {"eff_range": 150.0, "tau": 0.3, "sill": 100.0}
R2_TOL = 0.10
R2_SEED = 20261105

NOT_BUILT = ("R3 fits", "R4 record", "V1 bins", "V2 independent estimate", "V3 baseline fits")


class NotThere(Exception):
    """raw/ or the stock table is not there yet."""


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stored stock table and the raw measurements it was built from."""
    try:
        _, files = C1.potash()
    except C1.NotPulled as e:
        raise NotThere(str(e)) from e
    if not S.OUT.exists():
        raise NotThere("the stock table: python -m src.transform.stocks")
    _, meas = C1.potash_tables(files)
    return pd.read_parquet(S.OUT), meas


# --- S1, S2: the stock table ------------------------------------------------------

def s1_stocks(t: pd.DataFrame) -> Rule:
    """1,163 stocks, one per key, none of the seven incomplete locations,
    each finite and above 0."""
    problems = []
    if len(t) != N_STOCKS:
        problems.append(f"{len(t)} stocks, the rule says {N_STOCKS}")
    dup = int(t.duplicated(KEY).sum())
    if dup:
        problems.append(f"{dup} repeated (site, location_id)")
    present = sorted(INCOMPLETE & set(zip(t["site"], t["location_id"])))
    if present:
        problems.append(f"incomplete locations present: {present}")
    s = pd.to_numeric(t["stock"], errors="coerce")
    bad = int((~np.isfinite(s) | (s <= 0)).sum())
    if bad:
        problems.append(f"{bad} stocks not finite or not above 0")
    return Rule("S1 stocks", not problems, "; ".join(problems) or
                f"{len(t)} stocks, one per (site, location_id), none of the seven incomplete "
                f"locations; all finite, {s.min():.1f} to {s.max():.1f} Mg C/ha")


def loop_stocks(meas: pd.DataFrame, keys, depth: float = S.DEPTH_CM) -> dict:
    """S2's second path: one plain loop over the measurement rows."""
    total = {k: 0.0 for k in keys}
    for site, lid, top, bottom, soc, bd in zip(meas["site"], meas["location_id"],
                                               meas["sample_depth_min"], meas["sample_depth_max"],
                                               meas["SOCc"], meas["BD"]):
        k = (site, lid)
        if k in total and float(top) < depth:
            total[k] += float(soc) * float(bd) * (float(bottom) - float(top))
    return total


def s2_arithmetic(t: pd.DataFrame, meas: pd.DataFrame) -> Rule:
    """Every stored stock equals the loop's within 1e-9 Mg C/ha."""
    keys = list(zip(t["site"], t["location_id"]))
    total = loop_stocks(meas, keys)
    diff = max((abs(total[k] - float(s)) for k, s in zip(keys, t["stock"])), default=0.0)
    return Rule("S2 arithmetic", diff <= TOL_STOCK,
                f"largest difference from the loop-based recomputation {diff:.1e} Mg C/ha "
                f"(rule: {TOL_STOCK:g}); the hand-worked example is in src/transform/test_stocks.py")


# --- R1, R2: the REML code -----------------------------------------------------------

def _sklearn_kernel(sill: float, eff_range: float, tau: float, nu: float):
    """The same Matern-plus-white-noise covariance, as scikit-learn writes it:
    its length scale l puts sqrt(2 nu) h / l where reml.py puts h / phi."""
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    phi = eff_range / M.scale("matern", nu)
    return (ConstantKernel(sill * (1.0 - tau), "fixed")
            * Matern(math.sqrt(2.0 * nu) * phi, "fixed", nu)
            + WhiteKernel(sill * tau, "fixed"))


def _design_direct(xy: np.ndarray, mean: str) -> np.ndarray:
    """X again, without reml.design: about the field's corner rather than its
    centroid, so the comparison also tests that l_R ignores that choice."""
    if mean == "constant":
        return np.ones((len(xy), 1))
    c = (xy - xy.min(axis=0)) / 1000.0
    x, y = c[:, 0], c[:, 1]
    return np.column_stack([np.ones(len(xy)), x, y, x ** 2, x * y, y ** 2])


def reml_direct(V: np.ndarray, X: np.ndarray, z: np.ndarray) -> float:
    """l_R as its formula reads, with explicit inverses: slow, and independent
    of reml.py's Cholesky, QR and profiling."""
    n, p = X.shape
    Vi = np.linalg.inv(V)
    A = X.T @ Vi @ X
    P = Vi - Vi @ X @ np.linalg.inv(A) @ X.T @ Vi

    def logdet(m):
        return np.linalg.slogdet(m)[1]

    return -0.5 * ((n - p) * math.log(2.0 * math.pi) + logdet(V) + logdet(A)
                   - logdet(X.T @ X) + float(z @ P @ z))


def r1_likelihood(t: pd.DataFrame) -> Rule:
    """On every field's stocks: the Gaussian log-likelihood against
    scikit-learn's, and l_R against reml_direct, within 1e-8 relative."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    def rel(a, b):
        return abs(a - b) / abs(b)

    worst_ml = worst_reml = 0.0
    n_ml = n_reml = 0
    for _, g in t.groupby("site"):
        xy = g[["x", "y"]].to_numpy(float)
        z = g["stock"].to_numpy(float)
        y = z - z.mean()
        sill = float(np.var(z, ddof=1))
        dmax = float(pdist(xy).max())
        fields = {m: M.Field.of(xy, z, m) for m in M.MEANS}
        designs = {m: _design_direct(xy, m) for m in M.MEANS}
        for frac in R1_RANGES:
            a = frac * dmax
            for tau in R1_TAUS:
                for nu in R1_NUS:
                    kernel = _sklearn_kernel(sill, a, tau, nu)
                    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer=None).fit(xy, y)
                    worst_ml = max(worst_ml, rel(M.gaussian_loglik(xy, y, "matern", sill, a, tau, nu),
                                                 gp.log_marginal_likelihood_value_))
                    n_ml += 1
                    V = kernel(xy)
                    for m in M.MEANS:
                        worst_reml = max(worst_reml, rel(M.reml_loglik(fields[m], "matern", sill, a, tau, nu),
                                                         reml_direct(V, designs[m], z)))
                        n_reml += 1
                for family in ("exponential", "spherical"):
                    V = M.covariance(xy, family, sill, a, tau)
                    for m in M.MEANS:
                        worst_reml = max(worst_reml, rel(M.reml_loglik(fields[m], family, sill, a, tau),
                                                         reml_direct(V, designs[m], z)))
                        n_reml += 1
    passed = worst_ml <= TOL_R1 and worst_reml <= TOL_R1
    return Rule("R1 likelihood", passed,
                f"Gaussian log-likelihood against scikit-learn's GaussianProcessRegressor, largest relative "
                f"difference {worst_ml:.1e} over {n_ml} parameter sets; REML criterion against its formula "
                f"with explicit inverses, {worst_reml:.1e} over {n_reml}, both mean models and all three "
                f"families (rule: {TOL_R1:g})")


def r2_recovery(fields: int = R2_FIELDS, points: int = R2_POINTS, seed: int = R2_SEED) -> Rule:
    """Simulated exponential fields, the family given: the median REML
    estimates within 10 % of the truth."""
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(fields):
        xy, z = M.simulate(rng, points, R2_EXTENT, "exponential", R2_TRUTH["sill"],
                           R2_TRUTH["eff_range"], R2_TRUTH["tau"])
        f = M.fit(M.Field.of(xy, z, "constant"), "exponential")
        est.append({"eff_range": f.eff_range, "tau": f.tau, "sill": f.sill, "converged": f.converged})
    e = pd.DataFrame(est)
    med = {k: float(e[k].median()) for k in R2_TRUTH}
    off = {k: med[k] / R2_TRUTH[k] - 1.0 for k in R2_TRUTH}

    def iqr(k):
        return f"{e[k].quantile(0.25):.3g} to {e[k].quantile(0.75):.3g}"

    return Rule("R2 recovery", all(abs(v) <= R2_TOL for v in off.values()),
                f"medians over {fields} fields of {points} points: effective range {med['eff_range']:.1f} m "
                f"({off['eff_range']:+.1%}; quartiles {iqr('eff_range')}), nugget-to-sill {med['tau']:.3f} "
                f"({off['tau']:+.1%}; {iqr('tau')}), sill {med['sill']:.1f} ({off['sill']:+.1%}; "
                f"{iqr('sill')}); {int(e['converged'].sum())}/{fields} fits converged "
                f"(rule: each median within {R2_TOL:.0%})")


# --- reported, not judged ------------------------------------------------------------

def skewness_se(n: int) -> float:
    """Standard error of the adjusted sample skewness G1 under normality."""
    return math.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))


def stock_diagnostics(t: pd.DataFrame) -> pd.DataFrame:
    """Per field: stocks, mean, SD, CV, skewness with its standard error, and
    the R2 of the second-order trend (ordinary least squares)."""
    rows = []
    for site, g in t.groupby("site"):
        z = g["stock"].to_numpy(float)
        X = M.design(g[["x", "y"]].to_numpy(float), "quadratic")
        resid = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
        centred = z - z.mean()
        rows.append({"site": site, "stocks": len(z), "mean": f"{z.mean():.1f}",
                     "sd": f"{z.std(ddof=1):.1f}", "cv (%)": f"{100 * z.std(ddof=1) / z.mean():.1f}",
                     "skewness ± se": f"{stats.skew(z, bias=False):+.2f} ± {skewness_se(len(z)):.2f}",
                     "trend R2": f"{1 - (resid @ resid) / (centred @ centred):.2f}"})
    return pd.DataFrame(rows)


# --- the checkpoint -------------------------------------------------------------------

def run() -> tuple[list[Rule], pd.DataFrame]:
    """Every built rule, and the stock diagnostics. Raises NotThere."""
    t, meas = load()
    rules = [s1_stocks(t), s2_arithmetic(t, meas)]
    rules.append(r1_likelihood(t) if rules[0].passed else
                 Rule("R1 likelihood", False, "not evaluated: S1 failed"))
    rules.append(r2_recovery())
    return rules, stock_diagnostics(t)


def report(rules: list[Rule], diagnostics: Optional[pd.DataFrame],
           when: Optional[dt.datetime] = None) -> str:
    """The text Checkpoint 2 prints, and the RUNBOOK records."""
    when = when or dt.datetime.now(dt.timezone.utc)
    lines = [f"Checkpoint 2, run {when:%Y-%m-%dT%H:%M:%SZ}", ""]
    lines += [f"{'PASS' if r.passed else 'FAIL'}  {r.rule}: {r.detail}" for r in rules]
    lines += [f"----  {name}: not built yet" for name in NOT_BUILT]
    if diagnostics is not None:
        lines += ["", "Stocks in 0-30 cm per field, Mg C/ha (reported, not judged; "
                      "the Box-Cox profile of D3 comes with the fits):",
                  diagnostics.to_string(index=False)]
    passed = sum(r.passed for r in rules)
    verdict = "FAILED" if passed < len(rules) else ("NOT COMPLETE" if NOT_BUILT else "PASSED")
    lines += ["", f"Checkpoint 2 {verdict}: {passed}/{len(rules)} built rules passed, "
                  f"{len(NOT_BUILT)} not built yet"]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """python -m src.checks.checkpoint2"""
    if argv:
        print(main.__doc__)
        return 2
    try:
        rules, diagnostics = run()
    except NotThere as e:
        print(f"NOT THERE: {e}")
        return 2
    print(report(rules, diagnostics))
    if not all(r.passed for r in rules):
        return 1
    return 3 if NOT_BUILT else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
