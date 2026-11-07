"""Checkpoint 2: the stock table and the covariance models (Potash fields).

The rules were fixed in RUNBOOK.md on 05.11.2026, before any variogram was
computed, and each rule built so far is one function here: S1 and S2 for the
stock table, R1 and R2 for the REML code, R3 and R4 for the fits and their
record, V1-V3 for the moment baseline. scikit-learn and scikit-gstat give
the independent estimates R1, V2 and V3 compare with.

    python -m src.checks.checkpoint2    exit 0: every rule built and passed
                                        exit 1: a rule failed
                                        exit 2: raw/ or a curated table is missing
                                        exit 3: no rule failed, some not built yet
                                                (none: every rule is built)

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
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

try:                        # run as a module from the repository root
    from src.checks import checkpoint1 as C1
    from src.transform import anisotropy as AN
    from src.transform import reml as M
    from src.transform import stocks as S
    from src.transform import variograms as VG
except ImportError:         # imported with src/checks and src/transform on sys.path (pytest)
    import anisotropy as AN
    import checkpoint1 as C1
    import reml as M
    import stocks as S
    import variograms as VG

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

NOT_BUILT: tuple = ()
TOL_V2 = 1e-9                     # relative
TOL_V3 = 0.01                     # relative, of the weighted sum of squares
TOL_BOUND = 1e-9                  # relative: "inside the bounds"
TOL_ABOVE = 1e-4                  # l_R units: a profile higher than its fit means the fit was not the maximum


class NotThere(Exception):
    """raw/ or a curated table is not there yet."""


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


# --- R3, R4: the fits and their record -----------------------------------------------------

def load_reml() -> dict:
    """The record and the three tables `python -m src.transform.reml` writes."""
    paths = {"record": M.OUT, "families": M.OUT_FAMILIES, "pooling": M.OUT_POOLING,
             "boxcox": M.OUT_BOXCOX}
    if not all(p.exists() for p in paths.values()):
        raise NotThere("the REML record: python -m src.transform.reml")
    return {k: pd.read_parquet(p) for k, p in paths.items()}


def r3_fits(families: pd.DataFrame, record: pd.DataFrame,
            anisotropic: Optional[pd.DataFrame] = None) -> Rule:
    """Every fit converged inside its bounds, the anisotropic refits too; every
    D6 interval contains its estimate, and no profile rose above the fit it
    started from."""
    f, r = families, record
    problems = []
    an_text = ""
    if anisotropic is not None:
        a = anisotropic
        lo_, hi_ = 1.0 - TOL_BOUND, 1.0 + TOL_BOUND
        bad = (~a["converged"].astype(bool)
               | (a["eff_range_major"] < a["range_bound_lo"] * lo_) | (a["eff_range_major"] > a["range_bound_hi"] * hi_)
               | (a["tau"] < M.TAU_BOUNDS[0] * lo_) | (a["tau"] > M.TAU_BOUNDS[1] * hi_)
               | (a["ratio"] < AN.RATIO_BOUNDS[0] * lo_) | (a["ratio"] > AN.RATIO_BOUNDS[1] * hi_)
               | (a["family"].eq("matern") & ((a["nu"] < M.NU_BOUNDS[0] * lo_) | (a["nu"] > M.NU_BOUNDS[1] * hi_))))
        if bad.any():
            problems.append(f"{int(bad.sum())} anisotropic refits not converged inside their bounds")
        below = a["loglik"] < a["loglik_iso"] - 1e-6
        if below.any():
            problems.append(f"{int(below.sum())} anisotropic refits below their isotropic fit")
        an_text = f", and so did the {len(a)} anisotropic refits, none below its isotropic fit"
    failed = f[~f["converged"].astype(bool)]
    if len(failed):
        problems.append(f"{len(failed)} fits did not converge: " + ", ".join(
            f"{x['site']} {x['mean']} {x['family']}" for _, x in failed.iterrows()))
    lo, hi = 1.0 - TOL_BOUND, 1.0 + TOL_BOUND
    nu = f["nu"]
    outside = ((f["eff_range"] < f["range_bound_lo"] * lo) | (f["eff_range"] > f["range_bound_hi"] * hi)
               | (f["tau"] < M.TAU_BOUNDS[0] * lo) | (f["tau"] > M.TAU_BOUNDS[1] * hi)
               | (f["family"].eq("matern") & ((nu < M.NU_BOUNDS[0] * lo) | (nu > M.NU_BOUNDS[1] * hi))))
    if outside.any():
        problems.append(f"{int(outside.sum())} fits outside their bounds")
    without = ((r["eff_range"] < r["eff_range_lo"]) | (r["eff_range"] > r["eff_range_hi"])
               | (r["tau"] < r["tau_lo"]) | (r["tau"] > r["tau_hi"]))
    if without.any():
        problems.append(f"{int(without.sum())} record rows whose D6 interval does not contain the estimate")
    above = r["profile_above_fit"] > TOL_ABOVE
    if above.any():
        problems.append(f"{int(above.sum())} record rows whose profile rose up to "
                        f"{r.loc[above, 'profile_above_fit'].max():.2g} above the fit, which was "
                        "therefore not the maximum")
    own = int((f["fit"] == "own").sum())
    models = int((r["fit"] == "own").sum()) + len(r.loc[r["fit"] != "own", ["fit", "mean"]].drop_duplicates())
    return Rule("R3 fits", not problems, "; ".join(problems) or
                f"all {len(f)} fits converged inside their bounds ({own} of one field, "
                f"{len(f) - own} joint){an_text}; the D6 intervals of all {models} chosen models contain "
                f"their estimates, and no profile rose above its fit by more than {TOL_ABOVE:g}")


def r4_record(record: pd.DataFrame, pooling: pd.DataFrame, stocks: pd.DataFrame,
              large=M.LARGE, small=M.SMALL) -> Rule:
    """The record holds what R4 lists, and the small fields take the
    parameters D7's comparison gives them."""
    r = record
    absent = [c for c in M.RECORD_COLUMNS if c not in r.columns]
    if absent:
        return Rule("R4 record", False, f"columns missing: {absent}")
    sites = sorted(stocks["site"].unique())
    large = [s for s in large if s in sites]
    small = [s for s in small if s in sites]
    wanted = sorted((s, m) for s in sites for m in M.MEANS)
    problems = []
    own = r[r["fit"] == "own"]
    if sorted(zip(own["site"], own["mean"])) != wanted:
        problems.append(f"{len(own)} rows of own fits, not one per field and mean model")
    matern = r["family"].eq("matern")
    empty = (int(r.drop(columns=["nu"]).isna().sum().sum())
             + int((r == "").sum().sum())       # pandas 2 and 3 alike, without select_dtypes
             + int((matern & r["nu"].isna()).sum()))
    if empty:
        problems.append(f"{empty} empty cells")
    if (~matern & r["nu"].notna()).any():
        problems.append("nu given for a family other than the Matern")
    counts = stocks.groupby("site").size()
    wrong = int((r["n"].to_numpy() != r["site"].map(counts).to_numpy()).sum())
    if wrong:
        problems.append(f"{wrong} rows whose number of stocks differs from the stock table")
    said = []
    for m in M.MEANS:
        joint = r[(r["fit"] == "joint_large") & (r["mean"] == m)]
        if sorted(joint["site"]) != sorted(large) or joint["aic"].nunique() != 1:
            problems.append(f"the joint fit of the large fields under the {m} mean is missing or incomplete")
            continue
        own_m = own[own["mean"] == m]
        delta = float(joint["aic"].iloc[0]) - float(own_m.loc[own_m["site"].isin(large), "aic"].sum())
        pooled = delta <= M.POOL_AIC
        refit = r[(r["fit"] == "joint_all") & (r["mean"] == m)]
        if pooled and small and sorted(refit["site"]) != sorted(large + small):
            problems.append(f"sharing holds under the {m} mean, but the refit on all fields is missing")
        if not pooled and len(refit):
            problems.append(f"a refit on all fields under the {m} mean, which D7 does not call for")
        takes = "shared" if pooled else "own"
        if not (own_m.loc[own_m["site"].isin(small), "takes"] == takes).all():
            problems.append(f"under the {m} mean the small fields should take {takes} parameters")
        if not (own_m.loc[own_m["site"].isin(large), "takes"] == "own").all():
            problems.append(f"under the {m} mean a large field does not keep its own fit")
        p = pooling[pooling["mean"] == m]
        if len(p) != 1 or bool(p["pooled"].iloc[0]) != pooled:
            problems.append(f"the pooling table disagrees with the record under the {m} mean")
        said.append(f"{m} mean, sharing {delta:+.1f} AIC: the small fields take {takes} parameters")
    stage4 = r[r["for_stage4"].astype(bool)]
    if sorted(zip(stage4["site"], stage4["mean"])) != wanted:
        problems.append("not exactly one model per field and mean model is marked for Stage 4")
    return Rule("R4 record", not problems, "; ".join(problems) or
                f"{len(own)} rows, one per field and mean model, and {int((r['fit'] != 'own').sum())} "
                f"rows for D7's joint fits; no empty cell but nu outside the Matern; the numbers of "
                f"stocks as in the stock table; {'; '.join(said)}; one model per field and mean "
                f"model marked for Stage 4")


def cambardella(tau: float) -> str:
    """Spatial dependence from the nugget-to-sill ratio (Cambardella et al. 1994)."""
    return "strong" if tau < 0.25 else ("moderate" if tau <= 0.75 else "weak")


def _with_interval(x, name: str, digits: int) -> str:
    """'estimate [lo, hi]', an end that is the bound starred."""
    opened = set(str(x["interval_open"]).split(","))
    lo = f"{x[name + '_lo']:.{digits}f}{'*' if name + '_lo' in opened else ''}"
    hi = f"{x[name + '_hi']:.{digits}f}{'*' if name + '_hi' in opened else ''}"
    return f"{x[name]:.{digits}f} [{lo}, {hi}]"


def _family_text(x) -> str:
    return x["family"] + (f" nu {x['nu']:.2f}" if x["family"] == "matern" else "")


def model_diagnostics(reml: dict) -> dict:
    """The REML run's tables, as Checkpoint 2 reports them (not judged)."""
    r, fam, pool, bc = reml["record"], reml["families"], reml["pooling"], reml["boxcox"]
    label = {"constant": "(a)", "quadratic": "(b)"}
    own_fits = fam[fam["fit"] == "own"]
    rows = []
    for _, x in r[r["fit"] == "own"].sort_values(["mean", "site"]).iterrows():
        aics = own_fits.loc[(own_fits["site"] == x["site"]) & (own_fits["mean"] == x["mean"]), "aic"].sort_values()
        rows.append({"field": x["site"], "mean": label[x["mean"]], "family": _family_text(x),
                     "next family": f"+{aics.iloc[1] - aics.iloc[0]:.1f} AIC" if len(aics) > 1 else "-",
                     "nugget": f"{x['nugget']:.1f}", "partial sill": f"{x['partial_sill']:.1f}",
                     "effective range (m)": _with_interval(x, "eff_range", 0),
                     "nugget/sill": f"{_with_interval(x, 'tau', 2)} {cambardella(x['tau'])}",
                     "at bound": "-" if x["at_bound"] == "none" else x["at_bound"],
                     "takes": x["takes"]})
    tables = {"REML models, (a) constant and (b) second-order mean; 95 % profile intervals, "
              "* where an end is the bound:": pd.DataFrame(rows)}
    rows = []
    for _, p in pool.iterrows():
        joint = r[(r["fit"] == "joint_large") & (r["mean"] == p["mean"])]
        own = r[(r["fit"] == "own") & (r["mean"] == p["mean"])].set_index("site")["eff_range"]
        x = joint.iloc[0]
        rows.append({"mean": label[p["mean"]], "shared": _family_text(x),
                     "AIC jointly": f"{p['aic_joint']:.1f}", "AIC own, summed": f"{p['aic_own']:.1f}",
                     "difference": f"{p['delta_aic']:+.1f}",
                     "small fields take": "shared" if p["pooled"] else "own",
                     "shared range (m)": _with_interval(x, "eff_range", 0),
                     "shared nugget/sill": _with_interval(x, "tau", 2),
                     "shared/own range": ", ".join(f"{s} {x['eff_range'] / own[s]:.2f}" for s in joint["site"])})
    tables["D7, the large fields' correlation parameters shared:"] = pd.DataFrame(rows)
    refits = r[r["fit"] == "joint_all"]
    if len(refits):
        rows = []
        for m, g in refits.groupby("mean"):
            x = g.iloc[0]
            rows.append({"mean": label[m], "family": _family_text(x),
                         "range (m)": _with_interval(x, "eff_range", 0),
                         "nugget/sill": _with_interval(x, "tau", 2),
                         "small fields' sills": ", ".join(f"{s} {v:.1f}" for s, v in
                                                          zip(g["site"], g["sill"]) if s in M.SMALL)})
        tables["D7, the shared parameters re-estimated from all eight fields:"] = pd.DataFrame(rows)
    rows = [{"field": b["site"], "family": b["family"],
             "lambda": f"{b['lambda']:.2f} [{b['lambda_lo']:.2f}{'*' if b['lo_open'] else ''}, "
                       f"{b['lambda_hi']:.2f}{'*' if b['hi_open'] else ''}]",
             "interval excludes 1": "yes" if b["excludes_1"] else "no"}
            for _, b in bc.iterrows()]
    tables["D3, the Box-Cox lambda under mean model (a), 95 % profile interval:"] = pd.DataFrame(rows)
    return tables


# --- V1-V3: the moment baseline -------------------------------------------------------------

def load_baseline() -> dict:
    """The tables `python -m src.transform.variograms` and
    `python -m src.transform.anisotropy` write."""
    paths = {"variograms": VG.OUT, "fits": VG.OUT_FITS, "anisotropy": AN.OUT}
    if not all(p.exists() for p in paths.values()):
        raise NotThere("the baseline and the anisotropic fits: python -m src.transform.variograms, "
                       "then python -m src.transform.anisotropy")
    return {k: pd.read_parquet(p) for k, p in paths.items()}


def _d8_edges(xy: np.ndarray) -> np.ndarray:
    """D8's edges again, without variograms.edges: 0, w/2, 3w/2, ... up to
    half the largest separation."""
    w = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    half = float(pdist(xy).max()) / 2.0
    e = np.r_[0.0, (np.arange(1, int(half / w + 0.5) + 2) - 0.5) * w]
    return e[e <= half]


def _values_direct(xy: np.ndarray, z: np.ndarray, mean: str) -> np.ndarray:
    """The stocks, or their OLS residuals from the trend built as in R1."""
    if mean == "constant":
        return z
    X = _design_direct(xy, "quadratic")
    return z - X @ np.linalg.lstsq(X, z, rcond=None)[0]


def _omni(vario: pd.DataFrame, site: str, mean: str) -> pd.DataFrame:
    return vario[(vario["direction"] == "omni") & (vario["site"] == site)
                 & (vario["mean"] == mean)].sort_values("bin")


def v1_bins(vario: pd.DataFrame, stocks: pd.DataFrame) -> Rule:
    """D8's edges; only, and all, bins of 30 pairs or more fitted; no lag
    beyond half the field's largest separation."""
    problems, count, fitted = [], 0, 0
    for site, g in stocks.groupby("site"):
        xy = g[["x", "y"]].to_numpy(float)
        expected = _d8_edges(xy)
        half = float(pdist(xy).max()) / 2.0
        for m in M.MEANS:
            b = _omni(vario, site, m)
            if b.empty:
                problems.append(f"{site} {m}: no baseline variogram")
                continue
            count += 1
            lo, hi = b["lo"].to_numpy(float), b["hi"].to_numpy(float)
            if (len(lo) != len(expected) - 1 or not np.allclose(lo, expected[:-1], rtol=1e-12, atol=1e-9)
                    or not np.allclose(hi, expected[1:], rtol=1e-12, atol=1e-9)):
                problems.append(f"{site} {m}: bin edges differ from D8's")
            flags = b["fitted"].astype(bool).to_numpy()
            if (flags != (b["pairs"].to_numpy() >= VG.MIN_PAIRS)).any():
                problems.append(f"{site} {m}: the fitted bins are not those of {VG.MIN_PAIRS} pairs or more")
            fitted += int(flags.sum())
            if (b["hi"] > half * (1 + 1e-12)).any() or (b["lag"].dropna() > half).any():
                problems.append(f"{site} {m}: a lag beyond half the largest separation")
    return Rule("V1 bins", not problems, "; ".join(problems) or
                f"{count} baseline variograms, every field and mean model, with D8's edges; only and all "
                f"bins of {VG.MIN_PAIRS} pairs or more fitted ({fitted} bins); no lag beyond half its "
                "field's largest separation")


def v2_independent(vario: pd.DataFrame, stocks: pd.DataFrame) -> Rule:
    """scikit-gstat's semivariances for the same coordinates, values and
    edges, within 1e-9 relative, and the same pair counts."""
    import warnings
    import skgstat as skg
    problems, worst, bins = [], 0.0, 0
    for site, g in stocks.groupby("site"):
        xy, z = g[["x", "y"]].to_numpy(float), g["stock"].to_numpy(float)
        for m in M.MEANS:
            b = _omni(vario, site, m)
            upper = b["hi"].to_numpy(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk = skg.Variogram(xy, _values_direct(xy, z, m), estimator="matheron",
                                   bin_func=list(upper), maxlag=float(upper[-1]), use_nugget=True,
                                   fit_method=None)
            counts = np.asarray(sk.bin_count, dtype=int)
            ours = b["pairs"].to_numpy(int)
            if len(counts) != len(ours) or (counts != ours).any():
                problems.append(f"{site} {m}: pair counts differ from scikit-gstat's")
                continue
            ok = ours > 0
            mine = b["gamma"].to_numpy(float)[ok]
            worst = max(worst, float(np.max(np.abs(np.asarray(sk.experimental, float)[ok] - mine) / np.abs(mine))))
            bins += int(ok.sum())
    passed = not problems and worst <= TOL_V2
    return Rule("V2 independent estimate", passed, "; ".join(problems) or
                f"scikit-gstat {skg.__version__}, given the same coordinates, values and edges: the same "
                f"pair counts, and semivariances within {worst:.1e} relative over {bins} bins "
                f"(rule: {TOL_V2:g})")


def v3_fits(fits: pd.DataFrame, vario: pd.DataFrame) -> Rule:
    """Every baseline fit converged inside D8's bounds, and scikit-gstat's
    model, fitted to the same bins with sigma = h / sqrt(N), reaches a
    weighted sum of squares within 1 % of ours."""
    import warnings
    from scipy.optimize import curve_fit
    from skgstat import models
    problems, worst = [], 0.0
    fitted = vario[(vario["direction"] == "omni") & vario["fitted"].astype(bool)]
    for _, f in fits.iterrows():
        b = fitted[(fitted["site"] == f["site"]) & (fitted["mean"] == f["mean"])].sort_values("bin")
        h, gam, n = (b[c].to_numpy(float) for c in ("lag", "gamma", "pairs"))
        floor = VG.C_FLOOR * float(gam.max())
        a_max = VG.A_MAX_FACTOR * float(h.max())
        if not (bool(f["converged"]) and f["nugget"] >= 0 and f["partial_sill"] >= floor * (1 - 1e-9)
                and 0 < f["a"] <= a_max * (1 + 1e-9)):
            problems.append(f"{f['site']} {f['mean']} {f['family']}: not converged inside D8's bounds")
        scale = 1.0 if f["family"] == "spherical" else 3.0      # scikit-gstat's r: the range, or 3a
        r_lo, r_hi = scale * VG.A_MIN_FACTOR * float(h.max()), scale * a_max
        model = getattr(models, f["family"])
        sigma = h / np.sqrt(n)
        best = np.inf
        for r0, c0, b0 in ((h.mean(), gam.mean(), 0.0), (h.max(), np.ptp(gam), gam.min()),
                           (np.median(h), 0.5 * gam.max(), 0.5 * gam.min())):
            p0 = [min(max(r0, r_lo), r_hi), max(c0, 2 * floor), max(b0, 0.0)]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(model, h, gam, p0=p0, sigma=sigma, method="trf",
                                        bounds=([r_lo, floor, 0.0], [r_hi, np.inf, np.inf]), max_nfev=20000)
            except (RuntimeError, ValueError):
                continue
            res = (gam - np.asarray(model(h, *popt), float)) / sigma
            best = min(best, float(res @ res))
        rel = abs(best - float(f["wss"])) / float(f["wss"])
        worst = max(worst, rel)
        if rel > TOL_V3:
            problems.append(f"{f['site']} {f['mean']} {f['family']}: scikit-gstat {best:.4g} against ours "
                            f"{f['wss']:.4g}")
    return Rule("V3 baseline fits", not problems, "; ".join(problems) or
                f"all {len(fits)} baseline fits converged inside D8's bounds; scikit-gstat's models, fitted "
                f"to the same bins with sigma = h/sqrt(N), reach weighted sums of squares within "
                f"{worst:.2%} of ours (rule: {TOL_V3:.0%})")


def baseline_diagnostics(base: dict, record: pd.DataFrame) -> dict:
    """The baseline beside REML, Cressie-Hawkins beside Matheron, and the
    anisotropic refits (reported, not judged)."""
    label = {"constant": "(a)", "quadratic": "(b)"}
    fits, vario, an = base["fits"], base["variograms"], base["anisotropy"]
    own = record[record["fit"] == "own"].set_index(["site", "mean"])
    rows = []
    for _, f in fits[fits["chosen"].astype(bool)].sort_values(["mean", "site"]).iterrows():
        x = own.loc[(f["site"], f["mean"])]
        rows.append({"field": f["site"], "mean": label[f["mean"]],
                     "baseline": f["family"], "range (m)": f"{f['eff_range']:.0f}",
                     "nugget/sill": f"{f['tau']:.2f}",
                     "at bound": "-" if f["at_bound"] == "none" else f["at_bound"],
                     "REML": _family_text(x), "REML range (m)": _with_interval(x, "eff_range", 0),
                     "REML nugget/sill": _with_interval(x, "tau", 2)})
    tables = {"D8, the moment baseline beside REML (effective ranges as REML defines them):": pd.DataFrame(rows)}
    rows = []
    fitted = vario[(vario["direction"] == "omni") & vario["fitted"].astype(bool)]
    for (site, m), g in fitted.groupby(["site", "mean"]):
        ratio = g["gamma_ch"] / g["gamma"]
        rows.append({"field": site, "mean": label[m], "bins": len(g),
                     "Cressie-Hawkins / Matheron, median": f"{ratio.median():.2f}",
                     "smallest": f"{ratio.min():.2f}", "largest": f"{ratio.max():.2f}"})
    tables["Cressie-Hawkins beside Matheron over the fitted bins:"] = (
        pd.DataFrame(rows).sort_values(["mean", "field"]))
    rows = [{"field": x["site"], "mean": label[x["mean"]], "family": _family_text(x),
             "major axis": f"{x['angle']:.0f} deg", "ratio": f"{x['ratio']:.2f}",
             "ranges (m)": f"{x['eff_range_major']:.0f} / {x['eff_range_minor']:.0f}",
             "change in AIC": f"{x['delta_aic']:+.1f}", "Stage 4 adopts": "yes" if x["adopt"] else "no",
             "at bound": "-" if x["at_bound"] == "none" else x["at_bound"]}
            for _, x in an.sort_values(["mean", "site"]).iterrows()]
    tables["Geometric anisotropy by REML, the large fields (azimuth of the major axis, clockwise from "
           "north; Stage 4 adopts it only where the AIC falls by more than 2; directional moment "
           "variograms are in curated/potash_variograms.parquet and notebook 02):"] = pd.DataFrame(rows)
    return tables


# --- the checkpoint -------------------------------------------------------------------

def run() -> tuple[list[Rule], dict]:
    """Every built rule, and the tables reported beside them. Raises NotThere."""
    t, meas = load()
    reml = load_reml()
    rules = [s1_stocks(t), s2_arithmetic(t, meas)]
    rules.append(r1_likelihood(t) if rules[0].passed else
                 Rule("R1 likelihood", False, "not evaluated: S1 failed"))
    base = load_baseline()
    rules += [r2_recovery(), r3_fits(reml["families"], reml["record"], base["anisotropy"]),
              r4_record(reml["record"], reml["pooling"], t),
              v1_bins(base["variograms"], t), v2_independent(base["variograms"], t),
              v3_fits(base["fits"], base["variograms"])]
    tables = {"Stocks in 0-30 cm per field, Mg C/ha:": stock_diagnostics(t)}
    tables.update(model_diagnostics(reml))
    tables.update(baseline_diagnostics(base, reml["record"]))
    return rules, tables


def report(rules: list[Rule], tables: Optional[dict], when: Optional[dt.datetime] = None) -> str:
    """The text Checkpoint 2 prints, and the RUNBOOK records."""
    when = when or dt.datetime.now(dt.timezone.utc)
    lines = [f"Checkpoint 2, run {when:%Y-%m-%dT%H:%M:%SZ}", ""]
    lines += [f"{'PASS' if r.passed else 'FAIL'}  {r.rule}: {r.detail}" for r in rules]
    lines += [f"----  {name}: not built yet" for name in NOT_BUILT]
    if tables:
        lines += ["", "Reported, not judged:"]
        for title, table in tables.items():
            lines += ["", title, table.to_string(index=False)]
    passed = sum(r.passed for r in rules)
    verdict = "FAILED" if passed < len(rules) else ("NOT COMPLETE" if NOT_BUILT else "PASSED")
    lines += ["", f"Checkpoint 2 {verdict}: {passed}/{len(rules)} "
                  + (f"built rules passed, {len(NOT_BUILT)} not built yet" if NOT_BUILT else "rules")]
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
