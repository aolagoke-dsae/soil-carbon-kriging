"""Stage 2, D3 to D7: covariance parameters by residual maximum likelihood.

A field's stocks z are modelled as z = X b + e, e ~ N(0, V), with

    V = s2 C,    C = (1 - tau) R(h) + tau I,

where s2 is the sill (nugget plus partial sill), tau the nugget-to-sill ratio
and R the correlation of the spatially structured part (D5). X is the mean
model of D4: a constant (a) or a second-order polynomial in the coordinates (b).

Each family is written in terms of its effective range a: the distance at
which R falls to 0.05, or, for the spherical, which reaches zero, its range.
Bounds and reported ranges then mean the same thing in every family:

    exponential   R = exp(-h/phi)                                a = phi ln 20
    spherical     R = 1 - 1.5 h/a + 0.5 (h/a)^3 for h < a, else 0
    Matern        R = 2^(1-nu)/Gamma(nu) (h/phi)^nu K_nu(h/phi)   R(a) = 0.05

The Matern with nu = 0.5 is the exponential with the same a.

The restricted log-likelihood is the density of n - p orthonormal error
contrasts L'z, with L'X = 0 and L'L = I (Harville 1974):

    l_R = -1/2 [(n - p) log 2pi + log|V| + log|X'V^-1 X| - log|X'X| + z'Pz]
    P   = V^-1 - V^-1 X (X'V^-1 X)^-1 X'V^-1

The log|X'X| term makes l_R independent of how X is scaled. The sill has a
closed-form maximum, s2 = z'P_C z / (n - p), so the search runs over a and
tau, and nu for the Matern. AIC = -2 l_R + 2k, with k = 3 covariance
parameters, 4 for the Matern. The mean model is the same for every family a
field is compared over: D4 compares (a) with (b) by cross-validation in
Stage 4, never by likelihood.

D7 fits several fields at once, with the effective range, the nugget-to-
sill ratio and nu shared and a mean and a sill of their own each. The
fields are independent, so the joint l_R is the sum of each field's,
maximised over its own sill, and k counts the shared parameters plus one
sill per field. D6's intervals are 95 % profile-likelihood intervals; D3 is
a REML profile of the Box-Cox lambda.

    python -m src.transform.reml [--budget SECONDS]
        writes curated/potash_reml.parquet, rule R4's record of the models
        Stage 4 krigs with, and beside it every family's fit, the pooling
        comparison and the Box-Cox profiles. Each expensive step is cached in
        curated/reml_cache/, so a run stopped by --budget (exit 4) continues
        where it stopped when it is run again.

Bounds, set from the geometry and never from the stocks:
    a     from half the smallest separation to three times the largest
    tau   from 1e-6 to 1: a nugget under a millionth of the sill counts as
          none, and C stays invertible
    nu    from 0.2 to 3 (D5)

The search evaluates l_R on a grid inside the bounds and refines the best
grid points with L-BFGS-B; if L-BFGS-B stops without converging, a bounded
Nelder-Mead from where it stopped gets a second try.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special
from scipy.spatial.distance import pdist, squareform

FAMILIES = ("exponential", "spherical", "matern")
MEANS = ("constant", "quadratic")       # D4 (a) and (b)
R_AT_RANGE = 0.05
TAU_BOUNDS = (1e-6, 1.0)
NU_BOUNDS = (0.2, 3.0)
RANGE_FACTORS = (0.5, 3.0)              # x the smallest and the largest separation
RANGE_GRID = 12                         # log-spaced effective ranges in the start grid
TAU_GRID = (1e-6, 0.1, 0.25, 0.5, 0.75, 0.95)
NU_GRID = (0.25, 0.5, 1.0, 1.5, 2.5)
STARTS = 2                              # grid points refined by L-BFGS-B
LOG2PI = math.log(2.0 * math.pi)
_LN20 = math.log(1.0 / R_AT_RANGE)
_FAIL = 1e300                           # what the search sees where C is not positive definite


# --- correlation ---------------------------------------------------------------

def _matern(u: np.ndarray, nu: float) -> np.ndarray:
    """Matern correlation at scaled distances u = h/phi >= 0; 1 at u = 0."""
    u = np.asarray(u, dtype=float)
    out = np.ones_like(u)
    pos = u > 0
    up = u[pos]
    out[pos] = (2.0 ** (1.0 - nu) / special.gamma(nu)) * up ** nu * special.kv(nu, up)
    return out


@lru_cache(maxsize=4096)
def _matern_scale(nu: float) -> float:
    """h/phi at which the Matern correlation with smoothness nu is 0.05."""
    return optimize.brentq(lambda u: float(_matern(np.array([u]), nu)[0]) - R_AT_RANGE,
                           1e-6, 100.0, xtol=1e-13)


def scale(family: str, nu: Optional[float] = None) -> float:
    """The effective range in units of the family's distance parameter phi."""
    if family == "exponential":
        return _LN20
    if family == "spherical":
        return 1.0
    if family == "matern":
        return _matern_scale(float(nu))
    raise ValueError(f"unknown family {family!r}")


def correlation(family: str, h, eff_range: float, nu: Optional[float] = None) -> np.ndarray:
    """R(h) of a family with effective range eff_range (m)."""
    h = np.asarray(h, dtype=float)
    if family == "exponential":
        return np.exp(-h * (_LN20 / eff_range))
    if family == "spherical":
        x = np.minimum(h / eff_range, 1.0)
        return 1.0 - x * (1.5 - 0.5 * x * x)
    if family == "matern":
        return _matern(h * (scale("matern", nu) / eff_range), float(nu))
    raise ValueError(f"unknown family {family!r}")


def _c_matrix(d: np.ndarray, family: str, eff_range: float, tau: float,
              nu: Optional[float]) -> np.ndarray:
    """C = (1 - tau) R + tau I from condensed distances d."""
    C = squareform((1.0 - tau) * correlation(family, d, eff_range, nu), checks=False)
    np.fill_diagonal(C, 1.0)
    return C


def covariance(xy, family: str, sill: float, eff_range: float, tau: float,
               nu: Optional[float] = None) -> np.ndarray:
    """V = sill [(1 - tau) R + tau I] at the points xy (n x 2, m)."""
    return sill * _c_matrix(pdist(np.asarray(xy, dtype=float)), family, eff_range, tau, nu)


# --- mean model and data -----------------------------------------------------------

def design(xy, mean: str, centre=None) -> np.ndarray:
    """D4's design matrix. The quadratic is in kilometres about `centre` (the
    centroid by default), for conditioning only: l_R does not depend on it."""
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    if mean == "constant":
        return np.ones((n, 1))
    if mean == "quadratic":
        c = (xy - (xy.mean(axis=0) if centre is None else np.asarray(centre, dtype=float))) / 1000.0
        x, y = c[:, 0], c[:, 1]
        return np.column_stack([np.ones(n), x, y, x * x, x * y, y * y])
    raise ValueError(f"unknown mean model {mean!r}")


@dataclass(frozen=True, eq=False)
class Field:
    """One field's coordinates (m), values and mean model, with what every
    evaluation of l_R reuses."""
    xy: np.ndarray
    z: np.ndarray
    mean: str
    X: np.ndarray
    d: np.ndarray               # condensed pairwise distances (m)
    logdet_xtx: float

    @classmethod
    def of(cls, xy, z, mean: str = "constant") -> "Field":
        xy = np.asarray(xy, dtype=float)
        z = np.asarray(z, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(z):
            raise ValueError("xy must be n x 2 and z of length n")
        X = design(xy, mean)
        sign, logdet = np.linalg.slogdet(X.T @ X)
        if sign <= 0 or len(z) <= X.shape[1]:
            raise ValueError("the mean model cannot be estimated from these points")
        return cls(xy, z, mean, X, pdist(xy), float(logdet))

    @property
    def n(self) -> int:
        return len(self.z)

    @property
    def p(self) -> int:
        return self.X.shape[1]

    def range_bounds(self) -> tuple[float, float]:
        sep = self.d[self.d > 0]
        return RANGE_FACTORS[0] * float(sep.min()), RANGE_FACTORS[1] * float(sep.max())


# --- likelihoods ---------------------------------------------------------------------

def _terms(field: Field, C: np.ndarray):
    """log|C|, log|X'C^-1 X|, z'P_C z and the GLS b, by Cholesky and QR."""
    L = linalg.cholesky(C, lower=True, check_finite=False)
    logdet_c = 2.0 * float(np.log(np.diag(L)).sum())
    Xw = linalg.solve_triangular(L, field.X, lower=True, check_finite=False)
    zw = linalg.solve_triangular(L, field.z, lower=True, check_finite=False)
    Q, Rx = np.linalg.qr(Xw)
    logdet_xcx = 2.0 * float(np.log(np.abs(np.diag(Rx))).sum())
    qz = Q.T @ zw
    resid = zw - Q @ qz
    beta = linalg.solve_triangular(Rx, qz, lower=False, check_finite=False)
    return logdet_c, logdet_xcx, float(resid @ resid), beta


def reml_profile(field: Field, family: str, eff_range: float, tau: float,
                 nu: Optional[float] = None) -> tuple[float, float, np.ndarray]:
    """l_R maximised over the sill at (a, tau, nu): (l_R, sill, b)."""
    n, p = field.n, field.p
    logdet_c, logdet_xcx, rss, beta = _terms(field, _c_matrix(field.d, family, eff_range, tau, nu))
    sill = rss / (n - p)
    ll = -0.5 * ((n - p) * (LOG2PI + math.log(sill) + 1.0)
                 + logdet_c + logdet_xcx - field.logdet_xtx)
    return ll, sill, beta


def reml_loglik(field: Field, family: str, sill: float, eff_range: float, tau: float,
                nu: Optional[float] = None) -> float:
    """l_R at given parameters, the sill included."""
    n, p = field.n, field.p
    logdet_c, logdet_xcx, rss, _ = _terms(field, _c_matrix(field.d, family, eff_range, tau, nu))
    return -0.5 * ((n - p) * (LOG2PI + math.log(sill)) + logdet_c + logdet_xcx
                   - field.logdet_xtx + rss / sill)


def gaussian_loglik(xy, y, family: str, sill: float, eff_range: float, tau: float,
                    nu: Optional[float] = None) -> float:
    """log N(y; 0, V): the Gaussian log-likelihood of values with a known zero
    mean. Rule R1 compares it with scikit-learn's."""
    y = np.asarray(y, dtype=float)
    L = linalg.cholesky(covariance(xy, family, sill, eff_range, tau, nu), lower=True, check_finite=False)
    w = linalg.solve_triangular(L, y, lower=True, check_finite=False)
    return -0.5 * (len(y) * LOG2PI + float(w @ w)) - float(np.log(np.diag(L)).sum())


# --- the fit -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Fit:
    """One family's REML fit: to one field, or to several at once with the
    correlation parameters shared and a mean and a sill of their own each (D7)."""
    family: str
    mean: str
    ns: tuple                   # stocks per field
    p: int
    sills: tuple                # one per field
    tau: float
    eff_range: float
    nu: Optional[float]
    loglik: float               # summed over the fields
    betas: tuple                # the GLS coefficients, one tuple per field
    converged: bool
    at_bound: tuple
    message: str
    evaluations: int
    range_bounds: tuple         # the effective range's lower and upper bound (m)

    @property
    def n(self) -> int:
        return int(sum(self.ns))

    @property
    def sill(self) -> float:
        if len(self.sills) != 1:
            raise AttributeError("a joint fit has one sill per field: see .sills")
        return self.sills[0]

    @property
    def beta(self) -> tuple:
        if len(self.betas) != 1:
            raise AttributeError("a joint fit has one mean per field: see .betas")
        return self.betas[0]

    @property
    def k(self) -> int:
        """The shared effective range, nugget-to-sill ratio and, for the
        Matern, nu; plus one sill per field."""
        return (3 if self.family == "matern" else 2) + len(self.sills)

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.k

    @property
    def nugget(self) -> float:
        return self.sill * self.tau

    @property
    def partial_sill(self) -> float:
        return self.sill * (1.0 - self.tau)


def _as_fields(fields) -> tuple:
    return (fields,) if isinstance(fields, Field) else tuple(fields)


def joint_profile(fields, family: str, eff_range: float, tau: float,
                  nu: Optional[float] = None) -> tuple[float, list, list]:
    """The sum over independent fields of l_R, each maximised over its own
    sill: (l_R, sills, GLS coefficients)."""
    total, sills, betas = 0.0, [], []
    for f in _as_fields(fields):
        ll, sill, beta = reml_profile(f, family, eff_range, tau, nu)
        total += ll
        sills.append(sill)
        betas.append(beta)
    return total, sills, betas


class _Problem:
    """One family's search over one or more fields, in x = (log a, tau) and,
    for the Matern, nu. For several fields the effective range may run from
    the lowest of their lower bounds to the highest of their upper bounds."""

    def __init__(self, fields, family: str):
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}")
        self.fields = _as_fields(fields)
        if len({f.mean for f in self.fields}) != 1:
            raise ValueError("a joint fit needs the same mean model in every field")
        self.family = family
        self.matern = family == "matern"
        b = [f.range_bounds() for f in self.fields]
        self.range_bounds = (min(lo for lo, _ in b), max(hi for _, hi in b))
        self.bounds = ([(math.log(self.range_bounds[0]), math.log(self.range_bounds[1])), TAU_BOUNDS]
                       + ([NU_BOUNDS] if self.matern else []))
        self.names = ("eff_range", "tau", "nu")[:len(self.bounds)]
        self.count = 0

    def unpack(self, x):
        return math.exp(x[0]), float(x[1]), (float(x[2]) if self.matern else None)

    def x_of(self, f: "Fit") -> np.ndarray:
        return np.array([math.log(f.eff_range), f.tau] + ([f.nu] if self.matern else []))

    def loglik(self, x) -> float:
        self.count += 1
        a, tau, nu = self.unpack(x)
        try:
            ll = joint_profile(self.fields, self.family, a, tau, nu)[0]
        except (linalg.LinAlgError, np.linalg.LinAlgError, ValueError, FloatingPointError):
            return -_FAIL
        return ll if math.isfinite(ll) else -_FAIL

    def maximize(self, x0, free=None):
        """l_R maximised over the coordinates in `free` (all by default), the
        others held at x0: L-BFGS-B and, where it does not converge, a bounded
        Nelder-Mead from where it stopped. Returns (x, l_R, converged, message)."""
        x0 = np.asarray(x0, dtype=float)
        free = list(range(len(x0))) if free is None else list(free)

        def full(y):
            x = x0.copy()
            x[free] = y
            return x

        def objective(y):
            return -self.loglik(full(y))

        b = [self.bounds[i] for i in free]
        r = optimize.minimize(objective, x0[free], method="L-BFGS-B", bounds=b)
        if not r.success:
            nm = optimize.minimize(objective, r.x, method="Nelder-Mead", bounds=b,
                                   options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
            if nm.success and nm.fun <= r.fun + 1e-9:
                r = nm
        return full(r.x), -float(r.fun), bool(r.success), str(r.message)

    def make_fit(self, x, converged: bool, message: str) -> "Fit":
        a, tau, nu = self.unpack(x)
        ll, sills, betas = joint_profile(self.fields, self.family, a, tau, nu)
        at_bound = tuple(name for name, v, (lo, hi) in zip(self.names, x, self.bounds)
                         if min(v - lo, hi - v) <= 1e-6 * max(1.0, abs(v)))
        return Fit(self.family, self.fields[0].mean, tuple(f.n for f in self.fields),
                   self.fields[0].p, tuple(float(s) for s in sills), tau, a, nu, ll,
                   tuple(tuple(float(c) for c in b) for b in betas), converged, at_bound,
                   message, self.count, self.range_bounds)


def fit(fields, family: str, starts: int = STARTS) -> Fit:
    """D5 for one family, and D7 when `fields` holds several: the REML
    estimate inside the bounds, from a grid of starts refined by L-BFGS-B."""
    pr = _Problem(fields, family)
    grid = [np.array([la, t] + ([nu] if pr.matern else []))
            for la in np.linspace(pr.bounds[0][0], pr.bounds[0][1], RANGE_GRID)
            for t in TAU_GRID
            for nu in (NU_GRID if pr.matern else (None,))]
    values = np.array([pr.loglik(x) for x in grid])
    best = None
    for i in np.argsort(-values, kind="stable")[:starts]:
        if values[i] <= -_FAIL:
            break
        x, ll, ok, message = pr.maximize(grid[i])
        if best is None or ll > best[1]:
            best = (x, ll, ok, message)
    if best is None:
        raise RuntimeError("no grid point gives a positive-definite covariance")
    return pr.make_fit(best[0], best[2], best[3])


def fit_all(fields, families: Sequence[str] = FAMILIES) -> list[Fit]:
    """D5: every family fitted to one field, or jointly to several, under
    one mean model; the lowest AIC first."""
    return sorted((fit(fields, f) for f in families), key=lambda f: f.aic)


def simulate(rng: np.random.Generator, n: int, extent: tuple[float, float], family: str,
             sill: float, eff_range: float, tau: float, nu: Optional[float] = None,
             mean: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """n points uniform over [0, width] x [0, height] (m) and one draw of the
    Gaussian field on them: the simulation rule R2 runs."""
    xy = rng.uniform((0.0, 0.0), extent, size=(n, 2))
    L = linalg.cholesky(covariance(xy, family, sill, eff_range, tau, nu), lower=True)
    return xy, mean + L @ rng.standard_normal(n)


# --- D6: profile-likelihood intervals ---------------------------------------------------

CHI2_95_1 = 3.841458820694124       # the 95 % point of chi-squared with one degree of freedom
PROFILE_STEP = {"eff_range": 0.1, "tau": 0.03}    # the first step, in log(a) or in tau
PROFILE_XTOL = {"eff_range": 1e-4, "tau": 1e-5}
PROFILE_GROWTH = 1.6


@dataclass(frozen=True)
class End:
    """One end of a profile-likelihood interval. An end the profile does not
    reach inside the bounds is the bound itself, and is open."""
    value: float
    open: bool
    above_fit: float            # the most the profile rose above the fit's l_R (0: never)
    evaluations: int


@dataclass(frozen=True)
class Interval:
    parameter: str
    estimate: float
    lo: End
    hi: End

    @property
    def above_fit(self) -> float:
        return max(self.lo.above_fit, self.hi.above_fit)


def _walk(ll_at, t_hat, x_hat, bound, direction, step, xtol, target):
    """From t_hat towards `bound` in growing steps, each started from the
    last one's solution, until ll_at(t, start) -> (x, l) falls below
    `target`; then the crossing by Brent's method. Returns (end, open)."""
    prev_t, prev_x, h = t_hat, x_hat, step
    while direction * (bound - prev_t) > 1e-12:
        t = prev_t + direction * h
        if direction * (t - bound) > 0:
            t = bound
        x, ll = ll_at(t, prev_x)
        if ll < target:
            start = prev_x
            try:
                return optimize.brentq(lambda s: ll_at(s, start)[1] - target,
                                       min(prev_t, t), max(prev_t, t), xtol=xtol), False
            except ValueError:          # re-optimising moved the bracket's sign: keep the outer end
                return t, False
        prev_t, prev_x, h = t, x, h * PROFILE_GROWTH
    return bound, True


def profile_end(fields, fit: Fit, parameter: str, side: str) -> End:
    """One end ('lo' or 'hi') of D6's interval for `parameter` ('eff_range'
    or 'tau'): where the profile l_R, the other parameters re-estimated at
    every step, falls chi2(1, 0.95)/2 below the fit's."""
    pr = _Problem(fields, fit.family)
    j = pr.names.index(parameter)
    x_hat = pr.x_of(fit)
    free = [i for i in range(len(x_hat)) if i != j]
    above = [0.0]

    def ll_at(t, start):
        x0 = np.array(start, dtype=float)
        x0[j] = t
        x, ll, _, _ = pr.maximize(x0, free)
        above[0] = max(above[0], ll - fit.loglik)
        return x, ll

    k, direction = (0, -1.0) if side == "lo" else (1, 1.0)
    t, is_open = _walk(ll_at, x_hat[j], x_hat, pr.bounds[j][k], direction, PROFILE_STEP[parameter],
                       PROFILE_XTOL[parameter], fit.loglik - CHI2_95_1 / 2.0)
    return End(math.exp(t) if parameter == "eff_range" else float(t), bool(is_open),
               float(above[0]), pr.count)


def profile_interval(fields, fit: Fit, parameter: str) -> Interval:
    """D6: the 95 % profile-likelihood interval, both ends."""
    estimate = fit.eff_range if parameter == "eff_range" else fit.tau
    return Interval(parameter, estimate, profile_end(fields, fit, parameter, "lo"),
                    profile_end(fields, fit, parameter, "hi"))


def profile_loglik(fields, fit: Fit, parameter: str, value: float) -> float:
    """The profile l_R at one value of `parameter`, the other parameters
    re-estimated from the fit's."""
    pr = _Problem(fields, fit.family)
    j = pr.names.index(parameter)
    x0 = pr.x_of(fit)
    x0[j] = math.log(value) if parameter == "eff_range" else value
    return pr.maximize(x0, [i for i in range(len(x0)) if i != j])[1]


# --- D3: the Box-Cox profile -----------------------------------------------------------

BOXCOX_GRID = (-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, 3.0)    # walked outward from 1
BOXCOX_RANGE = (-2.0, 4.0)                              # where the interval's ends are sought
BOXCOX_STEP, BOXCOX_XTOL = 0.1, 1e-4


def boxcox(z, lam: float) -> np.ndarray:
    """(z^lambda - 1) / lambda, and log z at lambda = 0."""
    z = np.asarray(z, dtype=float)
    return np.log(z) if abs(lam) < 1e-12 else (z ** lam - 1.0) / lam


@dataclass(frozen=True)
class BoxCox:
    lam: float
    lo: float
    hi: float
    lo_open: bool
    hi_open: bool
    loglik: float               # the profile's maximum
    loglik_at_1: float          # its value on the original scale

    @property
    def excludes_1(self) -> bool:
        """D3's condition for a Stage 4 sensitivity on another scale. Which
        scale, where lambda lies above 1, is for Stage 4's rules to say."""
        return not (self.lo <= 1.0 <= self.hi)


def boxcox_profile(field: Field, fit: Fit) -> BoxCox:
    """D3: l(lambda) = l_R(z^(lambda)) + (lambda - 1) sum(log z), maximised
    over the covariance parameters of `fit`'s family under its mean model.
    The maximum is sought on a grid from -1 to 3 and refined between the best
    point's neighbours, or out to -2 or 4 when the best point is an end of
    the grid; the interval's ends, chi2(1, 0.95)/2 below it, between -2 and 4."""
    if np.any(field.z <= 0):
        raise ValueError("the Box-Cox transform needs positive values")
    slog = float(np.log(field.z).sum())

    def ll_at(lam, start):
        pr = _Problem(dataclasses.replace(field, z=boxcox(field.z, lam)), fit.family)
        x, ll, _, _ = pr.maximize(start)
        return x, ll + (lam - 1.0) * slog

    values = {1.0: ll_at(1.0, _Problem(field, fit.family).x_of(fit))}
    for side in ([l for l in BOXCOX_GRID if l > 1.0], [l for l in reversed(BOXCOX_GRID) if l < 1.0]):
        start = values[1.0][0]
        for lam in side:
            values[lam] = ll_at(lam, start)
            start = values[lam][0]
    lams = sorted(values)
    i = max(range(len(lams)), key=lambda k: values[lams[k]][1])
    start = values[lams[i]][0]
    lo_b = lams[i - 1] if i > 0 else BOXCOX_RANGE[0]
    hi_b = lams[i + 1] if i < len(lams) - 1 else BOXCOX_RANGE[1]
    r = optimize.minimize_scalar(lambda l: -ll_at(l, start)[1], method="bounded",
                                 bounds=(lo_b, hi_b), options={"xatol": BOXCOX_XTOL})
    if -float(r.fun) >= values[lams[i]][1]:
        lam_hat, ll_hat = float(r.x), -float(r.fun)
    else:
        lam_hat, ll_hat = lams[i], values[lams[i]][1]
    x_hat = ll_at(lam_hat, start)[0]
    target = ll_hat - CHI2_95_1 / 2.0
    (lo, lo_open), (hi, hi_open) = (
        _walk(ll_at, lam_hat, x_hat, BOXCOX_RANGE[k], direction, BOXCOX_STEP, BOXCOX_XTOL, target)
        for k, direction in ((0, -1.0), (1, 1.0)))
    return BoxCox(lam_hat, float(lo), float(hi), bool(lo_open), bool(hi_open), ll_hat,
                  values[1.0][1])


# --- the Stage 2 run ----------------------------------------------------------------------

LARGE = ("IL-RD", "IL-RS", "IL-RT", "NE")        # D7: 144 to 248 stocks each
SMALL = ("IL-BR", "IL-DG", "IL-MC", "IL-PT")     # D7: 52 to 97
POOL_AIC = 2.0                                    # D7: sharing at most this much worse
ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT / "curated"
OUT = CURATED / "potash_reml.parquet"             # rule R4's record: the models Stage 4 krigs with
OUT_FAMILIES = CURATED / "potash_reml_families.parquet"
OUT_POOLING = CURATED / "potash_pooling.parquet"
OUT_BOXCOX = CURATED / "potash_boxcox.parquet"
CACHE = CURATED / "reml_cache"
FIT_ORDER = ("own", "joint_large", "joint_all")
RECORD_COLUMNS = ["fit", "site", "mean", "family", "nu", "sill", "nugget", "partial_sill", "tau",
                  "eff_range", "eff_range_lo", "eff_range_hi", "tau_lo", "tau_hi", "interval_open",
                  "aic", "loglik", "k", "n", "n_fit", "converged", "at_bound", "takes",
                  "for_stage4", "range_bound_lo", "range_bound_hi", "profile_above_fit"]


class OutOfTime(Exception):
    """The run reached its --budget. What it finished is in the cache."""


class Cache:
    """The expensive steps' results, one JSON file each, named by a hash of
    this file's source, the step and its data: an interrupted run resumes
    where it stopped, and changed data or code is computed afresh. Without a
    folder nothing is stored."""

    def __init__(self, folder: Optional[Path] = None, deadline: Optional[float] = None):
        self.folder, self.deadline = folder, deadline
        self.source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.hits = self.computed = 0
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)

    def get(self, key: list, compute, load):
        if self.folder is None:
            return compute()
        name = hashlib.sha256(json.dumps([self.source] + key).encode()).hexdigest()[:24]
        path = self.folder / f"{name}.json"
        if path.exists():
            self.hits += 1
            return load(json.loads(path.read_text(encoding="utf-8")))
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise OutOfTime()
        value = compute()
        part = path.with_suffix(".part")
        part.write_text(json.dumps(dataclasses.asdict(value)), encoding="utf-8")
        part.rename(path)
        self.computed += 1
        return value


def _fit_from(d: dict) -> Fit:
    d = dict(d)
    for name in ("ns", "sills", "at_bound", "range_bounds"):
        d[name] = tuple(d[name])
    d["betas"] = tuple(tuple(b) for b in d["betas"])
    return Fit(**d)


def _data_key(fields) -> list:
    return [hashlib.sha256(f.xy.tobytes() + f.z.tobytes()).hexdigest()[:16] + ":" + f.mean
            for f in _as_fields(fields)]


def _record_row(kind: str, site: str, f: Fit, i: int, ivs: dict, takes: str,
                for_stage4: bool) -> dict:
    er, tv = ivs["eff_range"], ivs["tau"]
    opens = [name for name, end in (("eff_range_lo", er.lo), ("eff_range_hi", er.hi),
                                    ("tau_lo", tv.lo), ("tau_hi", tv.hi)) if end.open]
    sill = f.sills[i]
    return {"fit": kind, "site": site, "mean": f.mean, "family": f.family,
            "nu": np.nan if f.nu is None else f.nu, "sill": sill, "nugget": sill * f.tau,
            "partial_sill": sill * (1.0 - f.tau), "tau": f.tau, "eff_range": f.eff_range,
            "eff_range_lo": er.lo.value, "eff_range_hi": er.hi.value, "tau_lo": tv.lo.value,
            "tau_hi": tv.hi.value, "interval_open": ",".join(opens) or "none", "aic": f.aic,
            "loglik": f.loglik, "k": f.k, "n": f.ns[i], "n_fit": f.n, "converged": f.converged,
            "at_bound": ",".join(f.at_bound) or "none", "takes": takes, "for_stage4": for_stage4,
            "range_bound_lo": f.range_bounds[0], "range_bound_hi": f.range_bounds[1],
            "profile_above_fit": max(er.above_fit, tv.above_fit)}


def _family_row(kind: str, site: str, f: Fit, chosen: bool) -> dict:
    return {"fit": kind, "site": site, "mean": f.mean, "family": f.family,
            "nu": np.nan if f.nu is None else f.nu, "tau": f.tau, "eff_range": f.eff_range,
            "aic": f.aic, "loglik": f.loglik, "k": f.k, "n": f.n, "converged": f.converged,
            "at_bound": ",".join(f.at_bound) or "none", "chosen": chosen,
            "range_bound_lo": f.range_bounds[0], "range_bound_hi": f.range_bounds[1],
            "evaluations": f.evaluations, "message": f.message}


def run_stage(stocks: pd.DataFrame, families: Sequence[str] = FAMILIES,
              large: Sequence[str] = LARGE, small: Sequence[str] = SMALL,
              cache: Optional[Cache] = None, log=lambda message: None):
    """D5, D6, D7 and D3 on the stock table. Returns the record (R4), every
    family's fit, the pooling comparison and the Box-Cox profiles."""
    cache = cache or Cache()
    sites = sorted(stocks["site"].unique())
    large = [s for s in large if s in sites]
    small = [s for s in small if s in sites]
    fields = {(s, m): Field.of(g[["x", "y"]].to_numpy(float), g["stock"].to_numpy(float), m)
              for s, g in stocks.groupby("site") for m in MEANS}

    def fitted(members, family):
        return cache.get(["fit", _data_key(members), family], lambda: fit(members, family), _fit_from)

    def intervals(members, f):
        out = {}
        for parameter in ("eff_range", "tau"):
            ends = [cache.get(["end", _data_key(members), f.family, parameter, side],
                              lambda: profile_end(members, f, parameter, side), lambda d: End(**d))
                    for side in ("lo", "hi")]
            out[parameter] = Interval(parameter, f.eff_range if parameter == "eff_range" else f.tau, *ends)
        return out

    fam_rows, rec_rows, pool_rows, bc_rows = [], [], [], []
    own = {}
    for (s, m), field in fields.items():
        fits = sorted((fitted(field, family) for family in families), key=lambda f: f.aic)
        own[s, m] = fits[0]
        fam_rows += [_family_row("own", s, f, i == 0) for i, f in enumerate(fits)]
        log(f"D5 {s} {m}: {fits[0].family}, AIC {fits[0].aic:.1f}")
    own_iv = {}
    for (s, m), f in own.items():
        own_iv[s, m] = intervals(fields[s, m], f)
        er = own_iv[s, m]["eff_range"]
        log(f"D6 {s} {m}: effective range {er.lo.value:.0f} to {er.hi.value:.0f} m")
    pooled = {}
    for m in MEANS:
        members = [fields[s, m] for s in large]
        joint = sorted((fitted(members, family) for family in families), key=lambda f: f.aic)
        shared = joint[0]
        fam_rows += [_family_row("joint_large", "+".join(large), f, i == 0) for i, f in enumerate(joint)]
        shared_iv = intervals(members, shared)
        aic_own = sum(own[s, m].aic for s in large)
        delta = shared.aic - aic_own
        pooled[m] = bool(delta <= POOL_AIC)
        rec_rows += [_record_row("joint_large", s, shared, i, shared_iv, "shared", False)
                     for i, s in enumerate(large)]
        log(f"D7 {m}: the large fields jointly ({shared.family}) {delta:+.1f} AIC against "
            f"their own fits: {'shared' if pooled[m] else 'not shared'}")
        row = {"mean": m, "family": shared.family, "aic_joint": shared.aic, "aic_own": aic_own,
               "delta_aic": delta, "pooled": pooled[m], "eff_range_joint": shared.eff_range,
               "tau_joint": shared.tau, "nu_joint": np.nan if shared.nu is None else shared.nu,
               "eff_range_all": np.nan, "tau_all": np.nan}
        if pooled[m] and small:
            everyone = list(large) + list(small)
            all_members = [fields[s, m] for s in everyone]
            refit = fitted(all_members, shared.family)
            fam_rows.append(_family_row("joint_all", "+".join(everyone), refit, True))
            all_iv = intervals(all_members, refit)
            rec_rows += [_record_row("joint_all", s, refit, i, all_iv, "shared", s in small)
                         for i, s in enumerate(everyone)]
            row.update(eff_range_all=refit.eff_range, tau_all=refit.tau)
            log(f"D7 {m}: the shared parameters re-estimated from all {len(everyone)} fields")
        pool_rows.append(row)
    for (s, m), f in own.items():
        takes = "shared" if (s in small and pooled[m]) else "own"
        rec_rows.append(_record_row("own", s, f, 0, own_iv[s, m], takes, takes == "own"))
    for s in sites:
        field, f = fields[s, "constant"], own[s, "constant"]
        b = cache.get(["boxcox", _data_key(field), f.family], lambda: boxcox_profile(field, f),
                      lambda d: BoxCox(**d))
        bc_rows.append({"site": s, "mean": "constant", "family": f.family, "lambda": b.lam,
                        "lambda_lo": b.lo, "lambda_hi": b.hi, "lo_open": b.lo_open,
                        "hi_open": b.hi_open, "excludes_1": b.excludes_1,
                        "loglik_max": b.loglik, "loglik_at_1": b.loglik_at_1})
        log(f"D3 {s}: lambda {b.lam:.2f} ({b.lo:.2f} to {b.hi:.2f})")
    order = {k: i for i, k in enumerate(FIT_ORDER)}
    record = (pd.DataFrame(rec_rows, columns=RECORD_COLUMNS)
              .sort_values(["fit", "mean", "site"], key=lambda c: c.map(order) if c.name == "fit" else c)
              .reset_index(drop=True))
    return record, pd.DataFrame(fam_rows), pd.DataFrame(pool_rows), pd.DataFrame(bc_rows)


def main(argv: list[str]) -> int:
    """python -m src.transform.reml [--budget SECONDS]

    exit 0: the record and its three tables are written
         2: the stock table is not there yet (python -m src.transform.stocks)
         4: stopped at the budget; run again to continue from the cache"""
    budget = None
    if argv:
        if len(argv) == 2 and argv[0] == "--budget":
            budget = float(argv[1])
        else:
            print(main.__doc__)
            return 2
    try:                        # run as a module from the repository root
        from src.transform import stocks as S
    except ImportError:         # src/transform on sys.path (pytest)
        import stocks as S
    if not S.OUT.exists():
        print("NOT THERE: the stock table: python -m src.transform.stocks")
        return 2
    t0 = time.monotonic()
    cache = Cache(CACHE, None if budget is None else t0 + budget)

    def log(message):
        print(f"[{time.monotonic() - t0:4.0f} s] {message}", flush=True)

    try:
        record, families, pooling, profiles = run_stage(pd.read_parquet(S.OUT), cache=cache, log=log)
    except OutOfTime:
        print(f"Stopped at the {budget:g} s budget: {cache.computed} steps computed in this run, "
              f"{cache.hits} taken from the cache. Run again to continue.")
        return 4
    for table, path in ((record, OUT), (families, OUT_FAMILIES), (pooling, OUT_POOLING),
                        (profiles, OUT_BOXCOX)):
        table.to_parquet(path, index=False)
    print(f"{len(record)} rows -> {OUT.relative_to(ROOT).as_posix()}, with the families, pooling and "
          f"Box-Cox tables beside it ({cache.computed} steps computed, {cache.hits} from the cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
