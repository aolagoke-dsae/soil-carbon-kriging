"""Stage 4, D2 to D6: Bayesian kriging, cross-validated.

A field's stocks are z = X b + e, e ~ N(0, s2 C), C = (1 - tau) R(h) + tau I,
as in Stage 2 (reml.py). D3 puts a flat prior on b and 1/s2 on s2, both
integrated out, and a prior uniform in log a, tau and log nu on a grid inside
Stage 2's bounds. On the grid, the posterior of (a, tau, nu) is then

    log p = -1/2 log|C| - 1/2 log|X'C^-1 X| - (n - p)/2 log S2 + const,
    S2 = (z - X b^)' C^-1 (z - X b^),

the restricted likelihood with the sill integrated rather than maximised
(Kitanidis 1986), and a held-out stock's predictive distribution is the
posterior mixture of Student-t distributions with n - p degrees of freedom,
each centred on the universal-kriging predictor at its grid point with the
scale sqrt(S2 / (n - p) v0), where v0 is the kriging variance per unit sill
of a new observation, nugget included. Plug-in kriging is the Gaussian at
the grid's maximum.

    (a), (b)  Stage 2's models (D2): the family, the anisotropy where Stage 2
              adopted it, and under (b) the small fields' correlation
              parameters shared by all eight: their posterior is the product
              of the eight fields' integrated likelihoods, each from its
              training locations in the same repeat and fold
    (c)       D4: a constant plus terrain covariates, exponential, isotropic;
              the drift chosen in each training fold by the AIC of the
              maximum likelihood among 256 candidates
    D6        (a) on a Box-Cox scale, and at IL-RT on the log scale; where
              the Box-Cox scale's predictive reaches below a stock of zero,
              that part is cut off (scores.Transformed)

    python -m src.transform.kriging [--budget SECONDS] [--sites A,B] [--models a,b]
        writes curated/potash_cv_kriging.parquet, the drifts of (c) in
        curated/potash_cv_drifts.parquet and D5's selection in
        curated/potash_cv_selection.parquet. Every fold's result is cached
        in curated/kriging_cache/, so a run stopped by --budget (exit 4)
        continues where it stopped; --sites and --models split the work
        between processes, and a run without them writes the record.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.distance import squareform

try:                        # run as a module from the repository root
    from src.transform import anisotropy as A, folds as FO, reml as M, scores as SC
except ImportError:         # imported with src/transform on sys.path (pytest)
    import anisotropy as A
    import folds as FO
    import reml as M
    import scores as SC

LOG_RANGE_STEP = 0.1
TAU_STEP = 0.05
NU_COUNT = 7
DRIFT_FAMILY = "exponential"                # D4, and RF-RK's residuals (D7)
ATTRIBUTES = (("elevation",), ("slope_30", "slope_90", "slope_270"),
              ("curvature_30", "curvature_90", "curvature_270"),
              ("tpi_50", "tpi_150", "tpi_450"), ("twi",))
COVARIATES = tuple(c for group in ATTRIBUTES for c in group)
MEAN_OF = {"a": "constant", "b": "quadratic", "c": "drift"}
ROOT = M.ROOT
CURATED = M.CURATED
STOCKS = CURATED / "potash_stocks_0_30.parquet"
TERRAIN = CURATED / "potash_terrain.parquet"
OUT = CURATED / "potash_cv_kriging.parquet"
OUT_DRIFTS = CURATED / "potash_cv_drifts.parquet"
OUT_SELECTION = CURATED / "potash_cv_selection.parquet"
CACHE = CURATED / "kriging_cache"


ORCHESTRATION = ("main", "run_stage", "_rows", "selection", "load_inputs", "load_fields", "models_for",
                 "group_models")


def code_hash(*paths, skip=()) -> str:
    """A hash of the code in these files, without their docstrings and
    comments and without the top-level functions named in skip: the cache
    follows what is computed, not how it is described or scheduled (the
    cache keys carry each step's model and data)."""
    h = hashlib.sha256()
    for path in paths:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in skip)]
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and body and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        h.update(ast.dump(tree, include_attributes=False).encode())
    return h.hexdigest()


_SOURCE = code_hash(__file__, SC.__file__, M.__file__, A.__file__, skip=ORCHESTRATION)


# --- the grid of D3 ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """The prior's support: every (a, tau[, nu]), nu outermost, then a, then tau."""
    family: str
    log_ranges: tuple
    taus: tuple
    nus: tuple

    @property
    def size(self) -> int:
        return len(self.log_ranges) * len(self.taus) * len(self.nus)

    def points(self):
        """(eff_range, nu, taus) for each correlation matrix the grid needs."""
        for nu in self.nus:
            for lr in self.log_ranges:
                yield math.exp(lr), nu, self.taus

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """eff_range, tau and nu at every grid point, in the order points() visits them."""
        a, t, n = [], [], []
        for nu in self.nus:
            for lr in self.log_ranges:
                for tau in self.taus:
                    a.append(math.exp(lr))
                    t.append(tau)
                    n.append(np.nan if nu is None else nu)
        return np.array(a), np.array(t), np.array(n)


def make_grid(family: str, lo: float, hi: float, refine: int = 1) -> Grid:
    """D3's grid between the effective-range bounds lo and hi: steps of at most
    0.1 in log a with both bounds included, 0.05 in tau (its 0 standing for
    Stage 2's 1e-6), and seven values of nu evenly spaced in log nu over
    Stage 2's bounds. refine = 2 doubles the points along every axis (K2)."""
    span = math.log(hi) - math.log(lo)
    n = int(math.ceil(span / LOG_RANGE_STEP - 1e-9)) + 1
    n = (n - 1) * refine + 1
    log_ranges = tuple(np.linspace(math.log(lo), math.log(hi), n))
    steps = int(round(1.0 / TAU_STEP)) * refine
    taus = tuple([M.TAU_BOUNDS[0]] + [k / steps for k in range(1, steps + 1)])
    if family == "matern":
        k = (NU_COUNT - 1) * refine + 1
        nus = tuple(np.exp(np.linspace(math.log(M.NU_BOUNDS[0]), math.log(M.NU_BOUNDS[1]), k)))
    else:
        nus = (None,)
    return Grid(family, log_ranges, taus, nus)


# --- distances, designs ---------------------------------------------------------------------

def distances(xa: np.ndarray, xb: np.ndarray, angle: Optional[float] = None,
              ratio: Optional[float] = None) -> np.ndarray:
    """Distances between the rows of xa and xb (m), anisotropic as Stage 2's
    rotated_distances when an angle is given."""
    dx = xb[None, :, 0] - xa[:, None, 0]
    dy = xb[None, :, 1] - xa[:, None, 1]
    if angle is None:
        return np.hypot(dx, dy)
    return A.rotated_distances(dx, dy, angle, ratio)


def condensed(xy: np.ndarray, angle: Optional[float] = None, ratio: Optional[float] = None) -> np.ndarray:
    i, j = np.triu_indices(len(xy), 1)
    dx, dy = xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]
    return np.hypot(dx, dy) if angle is None else A.rotated_distances(dx, dy, angle, ratio)


def corr_matrix(d_cond: np.ndarray, family: str, eff_range: float, nu) -> np.ndarray:
    R = squareform(M.correlation(family, d_cond, eff_range, nu), checks=False)
    np.fill_diagonal(R, 1.0)
    return R


def standardise(train: np.ndarray, *others: np.ndarray) -> tuple:
    """Covariates centred and scaled by the training locations' means and
    standard deviations; a constant column is left centred only."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return tuple((a - mu) / sd for a in (train,) + others)


def design(mean: str, xy_train: np.ndarray, xy_test: np.ndarray, cov_train=None, cov_test=None,
           drift: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    """X at the training and the held-out locations. (b)'s quadratic is
    centred on the training locations' centroid; (c)'s covariates are
    standardised on them."""
    if mean in ("constant", "quadratic"):
        centre = xy_train.mean(axis=0)
        return M.design(xy_train, mean, centre), M.design(xy_test, mean, centre)
    if mean == "drift":
        ct, cs = standardise(cov_train[:, list(drift)], cov_test[:, list(drift)]) if drift else (
            np.empty((len(xy_train), 0)), np.empty((len(xy_test), 0)))
        return (np.column_stack([np.ones(len(xy_train)), ct]),
                np.column_stack([np.ones(len(xy_test)), cs]))
    raise ValueError(f"unknown mean model {mean!r}")


# --- the posterior on the grid --------------------------------------------------------------

@dataclass
class Posterior:
    """At every grid point: the log posterior (up to a constant), S2, and at
    every held-out location the predictor and the variance per unit sill."""
    grid: Grid
    n: int
    p: int
    logpost: np.ndarray                 # (G,)
    s2: np.ndarray                      # (G,)
    pred: np.ndarray                    # (G, m)
    var1: np.ndarray                    # (G, m)
    failed: int = 0
    beta: Optional[np.ndarray] = None   # (G, p): the GLS coefficients
    beta_v: Optional[np.ndarray] = None  # (G, p): the diagonal of (X'C^-1 X)^-1

    @property
    def df(self) -> int:
        return self.n - self.p

    def weights(self, logpost: Optional[np.ndarray] = None) -> np.ndarray:
        lp = self.logpost if logpost is None else logpost
        w = np.exp(lp - np.max(lp))
        return w / w.sum()


def evaluate(grid: Grid, xy: np.ndarray, z: np.ndarray, X: np.ndarray,
             xy0: Optional[np.ndarray] = None, X0: Optional[np.ndarray] = None,
             angle: Optional[float] = None, ratio: Optional[float] = None,
             coefficients: bool = False) -> Posterior:
    """The posterior at every grid point, and the universal-kriging predictor
    and variance at the held-out locations xy0 (none if xy0 is None). With
    n = p there is no posterior (log p is NaN), only the kriging."""
    return evaluate_many(grid, xy, [z], X, xy0, X0, angle, ratio, coefficients)[0]


def evaluate_many(grid: Grid, xy: np.ndarray, zs: list, X: np.ndarray,
                  xy0: Optional[np.ndarray] = None, X0: Optional[np.ndarray] = None,
                  angle: Optional[float] = None, ratio: Optional[float] = None,
                  coefficients: bool = False) -> list[Posterior]:
    """evaluate() for several value vectors at the same locations, one
    Cholesky factor per grid point for all of them: (a) and D6's transforms
    of it share every matrix but the values."""
    n, p = X.shape
    Z = np.column_stack(zs)
    k = Z.shape[1]
    m = 0 if xy0 is None else len(xy0)
    d_cond = condensed(xy, angle, ratio)
    d0 = distances(xy0, xy, angle, ratio) if m else None
    G = grid.size
    logpost, s2 = np.full((k, G), -np.inf), np.full((k, G), np.nan)
    pred, var1 = np.full((k, G, m), np.nan), np.full((G, m), np.nan)
    beta_all = np.full((k, G, p), np.nan) if coefficients else None
    beta_v = np.full((G, p), np.nan) if coefficients else None
    failed, g = 0, 0
    for a, nu, taus in grid.points():
        R = corr_matrix(d_cond, grid.family, a, nu)
        r0 = M.correlation(grid.family, d0, a, nu) if m else None
        for tau in taus:
            C = (1.0 - tau) * R
            np.fill_diagonal(C, 1.0)
            try:
                L = linalg.cholesky(C, lower=True, check_finite=False)
            except (linalg.LinAlgError, np.linalg.LinAlgError):
                failed += 1
                g += 1
                continue
            logdet_c = 2.0 * float(np.log(np.diag(L)).sum())
            rhs = np.column_stack([X, Z] + ([(1.0 - tau) * r0.T] if m else []))
            B = linalg.solve_triangular(L, rhs, lower=True, check_finite=False)
            Xw, Zw = B[:, :p], B[:, p:p + k]
            Q, Rx = np.linalg.qr(Xw)
            logdet_xcx = 2.0 * float(np.log(np.abs(np.diag(Rx))).sum())
            QZ = Q.T @ Zw
            resid = Zw - Q @ QZ
            ss = (resid * resid).sum(axis=0)
            logpost[:, g] = (-0.5 * logdet_c - 0.5 * logdet_xcx - 0.5 * (n - p) * np.log(ss)
                             if n > p else np.nan)
            s2[:, g] = ss
            if m or coefficients:
                beta = linalg.solve_triangular(Rx, QZ, lower=False, check_finite=False)
            if coefficients:
                Ri = linalg.solve_triangular(Rx, np.eye(p), lower=False, check_finite=False)
                beta_all[:, g], beta_v[g] = beta.T, (Ri * Ri).sum(axis=1)
            if m:
                Kw = B[:, p + k:]
                pred[:, g] = (X0 @ beta + Kw.T @ resid).T
                u = X0.T - Xw.T @ Kw
                v = linalg.solve_triangular(Rx, u, trans="T", lower=False, check_finite=False)
                var1[g] = np.maximum(1.0 - (Kw * Kw).sum(axis=0) + (v * v).sum(axis=0), 1e-12)
            g += 1
    return [Posterior(grid, n, p, logpost[i], s2[i], pred[i], var1, failed,
                      None if beta_all is None else beta_all[i], beta_v) for i in range(k)]


def predictive(post: Posterior, j: int, logpost: Optional[np.ndarray] = None,
               s2: Optional[np.ndarray] = None) -> SC.MixtureT:
    """Held-out location j's predictive distribution: the posterior mixture of t."""
    w = post.weights(logpost)
    s2 = post.s2 if s2 is None else s2
    ok = np.isfinite(post.logpost) & (w > 0)
    scale = np.sqrt(s2[ok] / post.df * post.var1[ok, j])
    return SC.MixtureT(w[ok], post.pred[ok, j], scale, post.df)


def plug_in(post: Posterior, j: int, logpost: Optional[np.ndarray] = None) -> SC.Normal:
    """The Gaussian at the grid's maximum: REML's estimate to the grid's resolution."""
    g = int(np.argmax(post.logpost if logpost is None else logpost))
    return SC.Normal(post.pred[g, j], math.sqrt(post.s2[g] / post.df * post.var1[g, j]))


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, levels) -> np.ndarray:
    """The quantiles of a discrete distribution: the smallest value whose
    cumulative weight reaches each level."""
    order = np.argsort(values, kind="stable")
    v, cw = values[order], np.cumsum(weights[order])
    cw /= cw[-1]
    i = np.searchsorted(cw, np.asarray(levels, dtype=float) - 1e-12, side="left")
    return v[np.minimum(i, len(v) - 1)]


def parameter_summary(post: Posterior, logpost: Optional[np.ndarray] = None) -> dict:
    """The posterior median and 90 % credible interval of the effective range,
    the nugget-to-sill ratio and nu, from the grid's weights."""
    w = post.weights(logpost)
    a, t, nu = post.grid.arrays()
    out = {}
    for name, v in (("eff_range", a), ("tau", t), ("nu", nu)):
        if np.all(np.isnan(v)):
            continue
        q = weighted_quantiles(v, w, (0.05, 0.5, 0.95))
        out.update({f"{name}_q05": float(q[0]), f"{name}_median": float(q[1]), f"{name}_q95": float(q[2])})
    return out


def coefficient_summary(post: Posterior) -> list[dict]:
    """Each coefficient's posterior mean and 90 % credible interval: a mixture
    over the grid of Student-t with n - p degrees of freedom."""
    w = post.weights()
    ok = np.isfinite(post.logpost) & (w > 0)
    out = []
    for j in range(post.p):
        mix = SC.MixtureT(w[ok], post.beta[ok, j], np.sqrt(post.s2[ok] / post.df * post.beta_v[ok, j]),
                          post.df)
        q = mix.quantiles([0.05, 0.95])
        out.append({"mean": mix.mean(), "q05": float(q[0]), "q95": float(q[1])})
    return out


# --- D4: the drift of (c) ----------------------------------------------------------------------

def drift_candidates() -> list[tuple]:
    """The 256 drifts that take at most one scale of each attribute, as indices
    into COVARIATES; the constant alone first."""
    start, options = 0, []
    for group in ATTRIBUTES:
        options.append([None] + list(range(start, start + len(group))))
        start += len(group)
    return [tuple(i for i in combo if i is not None) for combo in itertools.product(*options)]


def drift_name(drift: tuple) -> str:
    return "+".join(COVARIATES[i] for i in drift) or "constant"


def ml_table(grid: Grid, xy: np.ndarray, z: np.ndarray, cov: np.ndarray,
             candidates: Optional[list] = None) -> pd.DataFrame:
    """Every candidate drift's maximum log-likelihood (b and s2 profiled out,
    the covariance's parameters maximised on the grid) and AIC."""
    candidates = drift_candidates() if candidates is None else candidates
    n = len(z)
    zc = z - z.mean()
    covs = standardise(cov)[0]
    W_all = np.column_stack([np.ones(n), covs])
    d_cond = condensed(xy)
    by_size: dict[int, list] = {}
    for k, c in enumerate(candidates):
        by_size.setdefault(len(c), []).append(k)
    cols = {s: np.array([[0] + [1 + i for i in candidates[k]] for k in ks]) for s, ks in by_size.items()}
    best = np.full(len(candidates), -np.inf)
    arg = np.zeros(len(candidates), dtype=int)
    g = 0
    for a, nu, taus in grid.points():
        R = corr_matrix(d_cond, grid.family, a, nu)
        for tau in taus:
            C = (1.0 - tau) * R
            np.fill_diagonal(C, 1.0)
            try:
                L = linalg.cholesky(C, lower=True, check_finite=False)
            except (linalg.LinAlgError, np.linalg.LinAlgError):
                g += 1
                continue
            logdet_c = 2.0 * float(np.log(np.diag(L)).sum())
            B = linalg.solve_triangular(L, np.column_stack([W_all, zc]), lower=True, check_finite=False)
            Wb, zw = B[:, :-1], B[:, -1]
            Gm, gv, zz = Wb.T @ Wb, Wb.T @ zw, float(zw @ zw)
            for s, ks in by_size.items():
                idx = cols[s]
                Gs = Gm[idx[:, :, None], idx[:, None, :]]
                gs = gv[idx]
                sol = np.linalg.solve(Gs, gs[..., None])[..., 0]
                rss = zz - (gs * sol).sum(axis=1)
                ll = -0.5 * (n * (math.log(2.0 * math.pi) + np.log(rss / n) + 1.0) + logdet_c)
                better = ll > best[ks]
                best[np.array(ks)[better]] = ll[better]
                arg[np.array(ks)[better]] = g
            g += 1
    a_arr, t_arr, _ = grid.arrays()
    p = np.array([1 + len(c) for c in candidates])
    aic = -2.0 * best + 2.0 * (p + 3)
    return pd.DataFrame({"candidate": np.arange(len(candidates)),
                         "drift": [drift_name(c) for c in candidates], "p": p, "loglik": best,
                         "aic": aic, "eff_range": a_arr[arg], "tau": t_arr[arg]})


def choose(table: pd.DataFrame) -> int:
    """The lowest AIC; a tie goes to the fewer covariates, then the earlier candidate."""
    return int(table.sort_values(["aic", "p", "candidate"], kind="stable")["candidate"].iloc[0])


# --- the models of each field -----------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    name: str                       # a, b, c, boxcox_a, log_a
    mean: str
    family: str
    lo: float                       # the effective range's bounds (m)
    hi: float
    angle: Optional[float] = None
    ratio: Optional[float] = None
    shared: bool = False            # (b) at the small fields
    lam: Optional[float] = None     # D6's transform (0: the logarithm)

    def grid(self, refine: int = 1) -> Grid:
        return make_grid(self.family, self.lo, self.hi, refine)

    def key(self) -> list:
        return [self.name, self.mean, self.family, self.lo, self.hi, self.angle, self.ratio,
                self.shared, self.lam]


def models_for(site: str, record: pd.DataFrame, aniso: pd.DataFrame, boxcox: pd.DataFrame) -> list[Model]:
    """D2, D4 and D6's models at one field, from Stage 2's record."""
    out = []
    s4 = record[record["for_stage4"].astype(bool) & (record["site"] == site)].set_index("mean")
    own = record[(record["fit"] == "own") & (record["site"] == site)].iloc[0]
    for name in ("a", "b"):
        row = s4.loc[MEAN_OF[name]]
        ani = aniso[(aniso["site"] == site) & (aniso["mean"] == MEAN_OF[name]) & aniso["adopt"].astype(bool)]
        angle, ratio = (float(ani["angle"].iloc[0]), float(ani["ratio"].iloc[0])) if len(ani) else (None, None)
        out.append(Model(name, MEAN_OF[name], row["family"], float(row["range_bound_lo"]),
                         float(row["range_bound_hi"]), angle, ratio, row["takes"] == "shared"))
    out.append(Model("c", "drift", DRIFT_FAMILY, float(own["range_bound_lo"]), float(own["range_bound_hi"])))
    bc = boxcox[boxcox["site"] == site]
    if len(bc) and bool(bc["excludes_1"].iloc[0]):
        a = out[0]
        lam = float(bc["lambda"].iloc[0])
        if lam < 1.0:
            out.append(Model("log_a", a.mean, a.family, a.lo, a.hi, a.angle, a.ratio, False, 0.0))
        out.append(Model("boxcox_a", a.mean, a.family, a.lo, a.hi, a.angle, a.ratio, False, lam))
    return out


# --- data ---------------------------------------------------------------------------------------

@dataclass
class FieldData:
    site: str
    ids: np.ndarray
    xy: np.ndarray
    z: np.ndarray
    cov: np.ndarray
    folds: dict = dc_field(default_factory=dict)     # repeat -> fold label per location

    def split(self, repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
        f = self.folds[repeat]
        return np.flatnonzero(f != fold), np.flatnonzero(f == fold)


def load_fields(stocks: pd.DataFrame, terrain: pd.DataFrame, folds: pd.DataFrame) -> dict:
    t = stocks.merge(terrain[["site", "location_id"] + list(COVARIATES)], on=["site", "location_id"],
                     how="left", validate="one_to_one")
    if t[list(COVARIATES)].isna().any().any():
        raise ValueError("a location without terrain covariates")
    out = {}
    for site, g in t.groupby("site"):
        g = g.sort_values("location_id").reset_index(drop=True)
        fd = FieldData(site, g["location_id"].to_numpy(), g[["x", "y"]].to_numpy(float),
                       g["stock"].to_numpy(float), g[list(COVARIATES)].to_numpy(float))
        ff = folds[folds["site"] == site]
        for r, fr in ff.groupby("repeat"):
            fd.folds[int(r)] = fr.set_index("location_id").loc[fd.ids, "fold"].to_numpy()
        out[site] = fd
    return out


# --- one fold ----------------------------------------------------------------------------------

def summarise(dist, y: float) -> dict:
    """What the record keeps of one predictive distribution: its mean and
    variance, its 39 quantiles, its CRPS at y, and K3's check, the largest
    distance between the levels and the distribution function (by stdtr) at
    the quantiles."""
    q = dist.quantiles()
    err = float(np.max(np.abs(dist.cdf(q) - SC.LEVELS)))
    var = dist.var() if hasattr(dist, "var") else float("nan")
    out = {"mean": float(dist.mean()), "var": float(var), "q": list(map(float, q)),
           "crps": float(dist.crps(y)), "cdf_error": err}
    if isinstance(dist, SC.Transformed):
        out["outside"] = dist.p_outside
    return out


def group_models(models: list) -> list[list]:
    """Models that differ only in D6's transform share their factorisations:
    (a) with its transforms, every other model alone."""
    groups: dict = {}
    for m in models:
        groups.setdefault((m.mean, m.family, m.lo, m.hi, m.angle, m.ratio, m.shared), []).append(m)
    return list(groups.values())


def fold_results(fd: FieldData, models: list, train: np.ndarray, test: np.ndarray,
                 shared_others: Optional[np.ndarray] = None) -> dict:
    """Everything a group of models (group_models) predicts for one training
    fold's held-out locations, by model name."""
    base = models[0]
    xy, z, cov = fd.xy[train], fd.z[train], fd.cov[train]
    head = {"ids": list(map(str, fd.ids[test])), "y": list(map(float, fd.z[test]))}
    drift, extra = (), {}
    if base.mean == "drift":
        if len(models) > 1:
            raise ValueError("a drift model is fitted alone")
        table = ml_table(base.grid(), xy, z, cov)
        k = choose(table)
        drift = drift_candidates()[k]
        extra["drift"] = {"chosen": k, "aic": list(map(float, table["aic"])),
                          "loglik": list(map(float, table["loglik"])),
                          "eff_range": list(map(float, table["eff_range"])),
                          "tau": list(map(float, table["tau"]))}
    maps = [SC.boxcox_maps(m.lam) if m.lam is not None else None for m in models]
    zs = [g[1](z) if g is not None else z for g in maps]
    X, X0 = design(base.mean, xy, fd.xy[test], cov, fd.cov[test], drift)
    posts = evaluate_many(base.grid(), xy, zs, X, fd.xy[test], X0, base.angle, base.ratio)
    out = {}
    for model, g, post in zip(models, maps, posts):
        lp = post.logpost if shared_others is None else post.logpost + shared_others
        gmax = int(np.argmax(lp))
        a_arr, t_arr, n_arr = post.grid.arrays()
        res = dict(head, **extra)
        res["posterior"] = {"failed": post.failed, "eff_range_max": float(a_arr[gmax]),
                            "tau_max": float(t_arr[gmax]), "nu_max": float(n_arr[gmax]),
                            "n": post.n, "p": post.p}
        w = post.weights(lp)
        ok = np.isfinite(lp) & (w > 0)
        bayes, plug = [], []
        for j, y in enumerate(head["y"]):
            mix = predictive(post, j, lp)
            if g is None:
                b = summarise(mix, y)
                b["mean_post"] = float(np.dot(w[ok], post.pred[ok, j]))
                bayes.append(b)
                nd = plug_in(post, j, lp)
                plug.append({"mean": nd.mean(), "sd": nd.scale, "crps": nd.crps(y)})
            else:
                bayes.append(summarise(SC.Transformed(mix, *g), y))
        res["bayes"] = bayes
        if plug:
            res["plugin"] = plug
        out[model.name] = res
    return out


def fold_result(fd: FieldData, model: Model, train: np.ndarray, test: np.ndarray,
                shared_others: Optional[np.ndarray] = None) -> dict:
    """One model's fold_results."""
    return fold_results(fd, [model], train, test, shared_others)[model.name]


# --- the run ------------------------------------------------------------------------------------

class OutOfTime(Exception):
    """The run reached its --budget; what it finished is in the cache."""


class Cache:
    """One JSON file per finished step, named by a hash of the code, the step
    and its data, as Stage 2's REML run caches its fits."""

    def __init__(self, folder: Optional[Path] = None, deadline: Optional[float] = None):
        self.folder, self.deadline = folder, deadline
        self.hits = self.computed = 0
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)

    def _path(self, key: list) -> Path:
        name = hashlib.sha256(json.dumps([_SOURCE] + key, default=str).encode()).hexdigest()[:24]
        return self.folder / f"{name}.json"

    def get_many(self, keys: dict, compute) -> dict:
        """The values for several keys that one computation gives together."""
        if self.folder is None:
            return compute()
        paths = {name: self._path(key) for name, key in keys.items()}
        if all(p.exists() for p in paths.values()):
            self.hits += 1
            return {name: json.loads(p.read_text(encoding="utf-8")) for name, p in paths.items()}
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise OutOfTime()
        values = compute()
        for name, p in paths.items():
            part = p.with_suffix(".part")
            part.write_text(json.dumps(values[name]), encoding="utf-8")
            part.replace(p)
        self.computed += 1
        return values

    def get(self, key: list, compute):
        return self.get_many({0: key}, lambda: {0: compute()})[0]


def data_key(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def shared_loglik(fd: FieldData, model: Model, idx: np.ndarray, cache: Cache) -> np.ndarray:
    """Under (b), one field's integrated likelihood on the shared grid, from
    the locations idx (a training fold, or all)."""
    xy, z = fd.xy[idx], fd.z[idx]
    X = M.design(xy, "quadratic", xy.mean(axis=0))
    return np.array(cache.get(["shared", fd.site, model.key(), data_key(xy, z)],
                              lambda: list(map(float, evaluate(model.grid(), xy, z, X).logpost))))


def shared_others(fields: dict, site: str, model: Model, repeat: int, fold: int, cache: Cache) -> np.ndarray:
    """The other seven fields' integrated likelihoods from their training
    locations in the same repeat and fold: fold k is held out in every field."""
    total = None
    for other in sorted(fields):
        if other == site:
            continue
        ll = shared_loglik(fields[other], model, fields[other].split(repeat, fold)[0], cache)
        total = ll if total is None else total + ll
    return total


def run_stage(fields: dict, record: pd.DataFrame, aniso: pd.DataFrame, boxcox: pd.DataFrame,
              cache: Optional[Cache] = None, log=lambda message: None, sites=None,
              repeats=None, models=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every field, model, repeat and fold: the long table of predictions, (c)'s
    drifts and D5's selection."""
    cache = cache or Cache()
    rows, drift_rows = [], []
    candidates = drift_candidates()
    for site in (sites or sorted(fields)):
        fd = fields[site]
        chosen = [m for m in models_for(site, record, aniso, boxcox) if models is None or m.name in models]
        for group in group_models(chosen):
            t0 = time.monotonic()
            base = group[0]
            for r in (repeats if repeats is not None else sorted(fd.folds)):
                for k in range(FO.FOLDS):
                    train, test = fd.split(r, k)
                    data = data_key(fd.xy[train], fd.z[train], fd.cov[train], fd.xy[test], fd.z[test])
                    others = shared_others(fields, site, base, r, k, cache) if base.shared else None
                    keys = {m.name: ["fold", site, m.key(), r, k, data]
                            + ([data_key(others)] if base.shared else []) for m in group}
                    results = cache.get_many(keys, lambda: fold_results(fd, group, train, test, others))
                    for model in group:
                        res = results[model.name]
                        rows += _rows(site, model, r, k, res)
                        if "drift" in res:
                            d = res["drift"]
                            drift_rows += [{"site": site, "repeat": r, "fold": k, "candidate": c,
                                            "drift": drift_name(candidates[c]), "p": 1 + len(candidates[c]),
                                            "loglik": d["loglik"][c], "aic": d["aic"][c],
                                            "eff_range": d["eff_range"][c], "tau": d["tau"][c],
                                            "chosen": c == d["chosen"]} for c in range(len(candidates))]
            log(f"{site} {'+'.join(m.name for m in group)}: {time.monotonic() - t0:.0f} s")
    table = pd.DataFrame(rows)
    return table, pd.DataFrame(drift_rows), selection(table)


QCOLS = [f"q{lev:.3f}" for lev in SC.LEVELS]


def _rows(site: str, model: Model, r: int, k: int, res: dict) -> list:
    rows = []
    name = model.name if model.lam is not None else f"kriging_{model.name}"
    for j, (lid, y) in enumerate(zip(res["ids"], res["y"])):
        b = res["bayes"][j]
        row = {"site": site, "location_id": lid, "repeat": r, "fold": k, "method": name, "y": y,
               "mean": b["mean"], "var": b["var"], "crps": b["crps"], "cdf_error": b["cdf_error"],
               "mean_post": b.get("mean_post", np.nan), "outside": b.get("outside", np.nan)}
        row.update(dict(zip(QCOLS, b["q"])))
        rows.append(row)
        if "plugin" in res:
            p = res["plugin"][j]
            nd = SC.Normal(p["mean"], p["sd"])
            prow = {"site": site, "location_id": lid, "repeat": r, "fold": k,
                    "method": f"plugin_{model.name}", "y": y, "mean": p["mean"], "var": p["sd"] ** 2,
                    "crps": p["crps"], "cdf_error": np.nan, "mean_post": np.nan, "outside": np.nan}
            prow.update(dict(zip(QCOLS, map(float, nd.quantiles()))))
            rows.append(prow)
    return rows


def selection(table: pd.DataFrame) -> pd.DataFrame:
    """D5: per field, the mean model whose Bayesian kriging has the lowest CRPS,
    averaged over the locations and the repeats."""
    k = table[table["method"].isin(["kriging_a", "kriging_b", "kriging_c"])]
    per = (k.groupby(["site", "method", "repeat"])["crps"].mean().groupby(["site", "method"]).mean()
           .unstack("method"))
    out = per.rename(columns=lambda c: f"crps_{c[-1]}").reset_index()
    out["selected"] = per.idxmin(axis=1).str[-1].to_numpy()
    return out


def load_inputs() -> tuple:
    for p in (STOCKS, TERRAIN, FO.OUT, M.OUT, A.OUT, M.OUT_BOXCOX):
        if not p.exists():
            raise FileNotFoundError(p)
    fields = load_fields(pd.read_parquet(STOCKS), pd.read_parquet(TERRAIN), pd.read_parquet(FO.OUT))
    return (fields, pd.read_parquet(M.OUT), pd.read_parquet(A.OUT), pd.read_parquet(M.OUT_BOXCOX))


def main(argv: list[str]) -> int:
    """python -m src.transform.kriging [--budget SECONDS] [--sites A,B,...] [--models a,b,...]

    exit 0: written; 2: an input is missing; 4: stopped at the budget, run
    again to continue from the cache. With --sites or --models, only those
    fields or models are computed, into the cache, and nothing is written:
    two such runs can share the work, and a run without either then writes
    the record."""
    budget, sites, models = None, None, None
    args = list(argv)
    while args:
        if len(args) >= 2 and args[0] == "--budget":
            budget = float(args[1])
        elif len(args) >= 2 and args[0] == "--sites":
            sites = args[1].split(",")
        elif len(args) >= 2 and args[0] == "--models":
            models = args[1].split(",")
        else:
            print(main.__doc__)
            return 2
        args = args[2:]
    try:
        fields, record, aniso, boxcox = load_inputs()
    except FileNotFoundError as e:
        print(f"NOT THERE: {e}")
        return 2
    t0 = time.monotonic()
    cache = Cache(CACHE, None if budget is None else t0 + budget)
    try:
        table, drifts, chosen = run_stage(fields, record, aniso, boxcox, cache=cache, sites=sites,
                                          models=models,
                                          log=lambda msg: print(f"[{time.monotonic() - t0:5.0f} s] {msg}",
                                                                flush=True))
    except OutOfTime:
        print(f"Stopped at the {budget:g} s budget: {cache.computed} steps computed in this run, "
              f"{cache.hits} taken from the cache. Run again to continue.")
        return 4
    if sites is not None or models is not None:
        print(f"{', '.join(sites or ['every field'])}, {', '.join(models or ['every model'])}: done, "
              f"{cache.computed} steps computed, {cache.hits} from the cache")
        return 0
    table.to_parquet(OUT, index=False)
    drifts.to_parquet(OUT_DRIFTS, index=False)
    chosen.to_parquet(OUT_SELECTION, index=False)
    print(f"{len(table)} predictions -> {OUT.relative_to(ROOT).as_posix()}; "
          f"{cache.computed} folds computed, {cache.hits} from the cache")
    print(chosen.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
