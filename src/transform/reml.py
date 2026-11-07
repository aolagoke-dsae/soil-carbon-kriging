"""Stage 2, D4 and D5: covariance parameters by residual maximum likelihood.

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

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
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
    """One family's REML fit to one field and mean model."""
    family: str
    mean: str
    n: int
    p: int
    sill: float
    tau: float
    eff_range: float
    nu: Optional[float]
    loglik: float
    beta: tuple
    converged: bool
    at_bound: tuple
    message: str
    evaluations: int

    @property
    def k(self) -> int:
        return 4 if self.family == "matern" else 3

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.k

    @property
    def nugget(self) -> float:
        return self.sill * self.tau

    @property
    def partial_sill(self) -> float:
        return self.sill * (1.0 - self.tau)


def fit(field: Field, family: str, starts: int = STARTS) -> Fit:
    """D5 for one family: the REML estimate inside the bounds."""
    matern = family == "matern"
    lo, hi = field.range_bounds()
    bounds = [(math.log(lo), math.log(hi)), TAU_BOUNDS] + ([NU_BOUNDS] if matern else [])
    count = [0]

    def unpack(v):
        return math.exp(v[0]), float(v[1]), (float(v[2]) if matern else None)

    def objective(v) -> float:
        count[0] += 1
        a, tau, nu = unpack(v)
        try:
            ll = reml_profile(field, family, a, tau, nu)[0]
        except (linalg.LinAlgError, np.linalg.LinAlgError, ValueError, FloatingPointError):
            return _FAIL
        return -ll if math.isfinite(ll) else _FAIL

    grid = [np.array([la, t] + ([nu] if matern else []))
            for la in np.linspace(bounds[0][0], bounds[0][1], RANGE_GRID)
            for t in TAU_GRID
            for nu in (NU_GRID if matern else (None,))]
    values = np.array([objective(v) for v in grid])
    best = None
    for i in np.argsort(values, kind="stable")[:starts]:
        if values[i] >= _FAIL:
            break
        r = optimize.minimize(objective, grid[i], method="L-BFGS-B", bounds=bounds)
        if not r.success:
            nm = optimize.minimize(objective, r.x, method="Nelder-Mead", bounds=bounds,
                                   options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
            if nm.success and nm.fun <= r.fun + 1e-9:
                r = nm
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        raise RuntimeError("no grid point gives a positive-definite covariance")
    a, tau, nu = unpack(best.x)
    ll, sill, beta = reml_profile(field, family, a, tau, nu)
    names = ("eff_range", "tau", "nu")
    at_bound = tuple(name for name, v, (b_lo, b_hi) in zip(names, best.x, bounds)
                     if min(v - b_lo, b_hi - v) <= 1e-6 * max(1.0, abs(v)))
    return Fit(family, field.mean, field.n, field.p, sill, tau, a, nu, ll,
               tuple(float(b) for b in beta), bool(best.success), at_bound,
               str(best.message), count[0])


def fit_all(field: Field, families: Sequence[str] = FAMILIES) -> list[Fit]:
    """D5: every family fitted to one field and mean model, lowest AIC first."""
    return sorted((fit(field, f) for f in families), key=lambda f: f.aic)


def simulate(rng: np.random.Generator, n: int, extent: tuple[float, float], family: str,
             sill: float, eff_range: float, tau: float, nu: Optional[float] = None,
             mean: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """n points uniform over [0, width] x [0, height] (m) and one draw of the
    Gaussian field on them: the simulation rule R2 runs."""
    xy = rng.uniform((0.0, 0.0), extent, size=(n, 2))
    L = linalg.cholesky(covariance(xy, family, sill, eff_range, tau, nu), lower=True)
    return xy, mean + L @ rng.standard_normal(n)
