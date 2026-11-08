"""Stage 4, D8: predictive distributions and their scores.

Every method's prediction at a held-out location is a predictive
distribution of one of four kinds:

    MixtureT     Bayesian kriging (D3): a posterior mixture of Student-t
                 distributions sharing one number of degrees of freedom
    Normal       plug-in kriging
    Weighted     an empirical distribution with weights: QRF (D7) and the
                 baseline
    Transformed  a distribution on a transformed scale, carried back (D6)

Each gives its distribution function, its quantiles at the 39 levels of D8,
its mean and its CRPS at an observed value. The CRPS of a mixture is
E|X - y| - 1/2 E|X - X'|: the first term in closed form for every
component (Jordan et al. 2019), the second, the integral of F(1 - F), by
Gauss-Legendre quadrature on panels across the mixture's support. A weighted
empirical distribution's CRPS is exact, by the energy form; S1 sets a second,
piecewise integral beside it.

Inside the mixture's integrals and its quantile search, the Student-t
distribution function is a cubic Hermite interpolant of SciPy's stdtr on
steps of 0.005 (within 2e-12 of it; test_scores checks), five times faster
than stdtr itself; cdf() and every check use stdtr.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import special

LEVELS = np.round(np.arange(1, 40) * 0.025, 3)          # 0.025, 0.05, ..., 0.975
CENTRAL = np.round(np.arange(1, 20) * 0.05, 2)           # PICP's intervals, 5 % ... 95 %
PRUNE = 1e-15                   # mixture components lighter than this are dropped
TAIL = 1e-13                    # each component's support is cut at these quantiles
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
PANEL = 2.0                     # a panel's width, in units of the narrowest component's scale
TABLE_STEP = 0.005              # the interpolated t distribution function's step ...
TABLE_EDGE = 40.0               # ... over [-40, 40]; stdtr itself outside
MAX_PANELS = 4000
NEWTON_TOL = 1e-13              # |F(q) - level| at which a quantile is accepted
SQRT_PI = math.sqrt(math.pi)


def level_index(level: float) -> int:
    """The column of LEVELS that holds a level."""
    return int(round(level / 0.025)) - 1


def central_bounds(p: float) -> tuple[int, int]:
    """Columns of LEVELS holding the central p-interval's ends."""
    return level_index((1.0 - p) / 2.0), level_index((1.0 + p) / 2.0)


# --- Student-t and normal pieces --------------------------------------------------------

def t_cdf(z, df: float):
    return special.stdtr(df, z)


def t_pdf(z, df: float):
    z = np.asarray(z, dtype=float)
    c = special.gammaln((df + 1.0) / 2.0) - special.gammaln(df / 2.0) - 0.5 * math.log(df * math.pi)
    return np.exp(c - (df + 1.0) / 2.0 * np.log1p(z * z / df))


def t_ppf(p, df: float):
    return special.stdtrit(df, p)


def t_half_spread(df: float) -> float:
    """1/2 E|T - T'| for two independent standard Student-t variables."""
    return 2.0 * math.sqrt(df) * math.exp(special.betaln(0.5, df - 0.5)
                                          - 2.0 * special.betaln(0.5, df / 2.0)) / (df - 1.0)


def t_abs_moment(z, df: float):
    """E|T - z| for a standard Student-t T."""
    z = np.asarray(z, dtype=float)
    return z * (2.0 * t_cdf(z, df) - 1.0) + 2.0 * t_pdf(z, df) * (df + z * z) / (df - 1.0)


def crps_t(y, loc, scale, df: float):
    """CRPS of a location-scale Student-t (Jordan et al. 2019)."""
    z = (np.asarray(y, dtype=float) - loc) / scale
    return scale * (t_abs_moment(z, df) - t_half_spread(df))


def crps_normal(y, loc, scale):
    """CRPS of a normal (Gneiting & Raftery 2007)."""
    z = (np.asarray(y, dtype=float) - loc) / scale
    return scale * (z * special.erf(z / math.sqrt(2.0))
                    + 2.0 * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) - 1.0 / SQRT_PI)


def integrate(f: Callable, a: float, b: float, panels: int) -> float:
    """Gauss-Legendre with 16 nodes on each of `panels` equal panels of [a, b]."""
    if b <= a:
        return 0.0
    x, w = nodes(a, b, panels)
    return float(np.dot(w, f(x)))


def nodes(a: float, b: float, panels: int) -> tuple[np.ndarray, np.ndarray]:
    """integrate()'s nodes and weights, in increasing order."""
    edges = np.linspace(a, b, panels + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    return ((mid[:, None] + half[:, None] * GL_NODES[None, :]).ravel(),
            (half[:, None] * GL_WEIGHTS[None, :]).ravel())


class TTable:
    """The standard Student-t distribution function with df degrees of freedom
    as a cubic Hermite interpolant of stdtr, matching its values and slopes
    every TABLE_STEP on [-TABLE_EDGE, TABLE_EDGE]; stdtr itself outside."""

    def __init__(self, df: float):
        self.df = float(df)
        self.n = int(round(2.0 * TABLE_EDGE / TABLE_STEP))
        t = np.linspace(-TABLE_EDGE, TABLE_EDGE, self.n + 1)
        F, f = t_cdf(t, self.df), t_pdf(t, self.df) * TABLE_STEP
        F0, F1, f0, f1 = F[:-1], F[1:], f[:-1], f[1:]
        self.c = np.stack([F0, f0, 3.0 * (F1 - F0) - 2.0 * f0 - f1, 2.0 * (F0 - F1) + f0 + f1])

    def cdf(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        u = (z + TABLE_EDGE) * (1.0 / TABLE_STEP)
        i = np.clip(np.floor(u), 0, self.n - 1).astype(np.intp)
        s = u - i
        c0, c1, c2, c3 = self.c[:, i]
        out = c0 + s * (c1 + s * (c2 + s * c3))
        outside = np.abs(z) > TABLE_EDGE
        if outside.any():
            out[outside] = t_cdf(z[outside], self.df)
        return out


@lru_cache(maxsize=64)
def t_table(df: float) -> TTable:
    return TTable(df)


# --- the four kinds of predictive distribution ------------------------------------------------

class MixtureT:
    """sum_k w_k t_df((x - loc_k) / scale_k) / scale_k."""

    def __init__(self, weights, locs, scales, df: float):
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        keep = w >= PRUNE
        self.w = w[keep] / w[keep].sum()
        self.loc = np.asarray(locs, dtype=float)[keep]
        self.scale = np.asarray(scales, dtype=float)[keep]
        self.df = float(df)
        if not (np.all(self.scale > 0) and self.df > 1.0):
            raise ValueError("every scale must be positive and df above 1")
        self._spread = self._grid = None

    def _z(self, x: np.ndarray) -> np.ndarray:
        return (x[None, :] - self.loc[:, None]) / self.scale[:, None]

    def cdf(self, x):
        """The distribution function, by stdtr."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self.w @ t_cdf(self._z(x), self.df)

    def fast_cdf(self, x):
        """The same through the interpolated t distribution function."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self.w @ t_table(self.df).cdf(self._z(x))

    def pdf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return (self.w / self.scale) @ t_pdf(self._z(x), self.df)

    def mean(self) -> float:
        return float(np.dot(self.w, self.loc))

    def var(self) -> float:
        m = self.mean()
        return float(np.dot(self.w, self.scale ** 2 * self.df / (self.df - 2.0)
                            + (self.loc - m) ** 2)) if self.df > 2.0 else math.inf

    def support(self) -> tuple[float, float]:
        q = t_ppf(TAIL, self.df)
        return float((self.loc + self.scale * q).min()), float((self.loc - self.scale * q).max())

    def panels(self, a: float, b: float) -> int:
        return int(min(MAX_PANELS, max(8, math.ceil((b - a) / (PANEL * self.scale.min())))))

    def _on_nodes(self):
        """The quadrature's nodes and weights across the support, and F there."""
        if self._grid is None:
            a, b = self.support()
            x, w = nodes(a, b, self.panels(a, b))
            self._grid = (x, w, self.fast_cdf(x))
        return self._grid

    def quantiles(self, levels=LEVELS) -> np.ndarray:
        """Each level's quantile by Newton's method, started from F on the
        quadrature's nodes, kept inside the bracket the components' own
        quantiles give and bisected where a step leaves it."""
        levels = np.asarray(levels, dtype=float)
        cq = self.loc[:, None] + self.scale[:, None] * t_ppf(levels, self.df)[None, :]
        lo, hi = cq.min(axis=0), cq.max(axis=0)
        xn, _, Fn = self._on_nodes()
        x = np.clip(np.interp(levels, np.maximum.accumulate(Fn), xn), lo, hi)
        for _ in range(200):
            f = self.fast_cdf(x) - levels
            if np.all(np.abs(f) <= NEWTON_TOL):
                break
            lo = np.where(f < 0, x, lo)
            hi = np.where(f > 0, x, hi)
            d = self.pdf(x)
            step = np.where(d > 0, x - f / np.where(d > 0, d, 1.0), np.nan)
            ok = np.isfinite(step) & (step > lo) & (step < hi)
            x = np.where(np.abs(f) <= NEWTON_TOL, x, np.where(ok, step, 0.5 * (lo + hi)))
        return x

    def spread(self) -> float:
        """1/2 E|X - X'|, the integral of F(1 - F)."""
        if self._spread is None:
            _, w, F = self._on_nodes()
            self._spread = float(np.dot(w, F * (1.0 - F)))
        return self._spread

    def crps(self, y: float) -> float:
        z = (float(y) - self.loc) / self.scale
        return float(np.dot(self.w, self.scale * t_abs_moment(z, self.df))) - self.spread()


class Normal:
    def __init__(self, loc: float, scale: float):
        self.loc, self.scale = float(loc), float(scale)

    def cdf(self, x):
        return special.ndtr((np.atleast_1d(np.asarray(x, dtype=float)) - self.loc) / self.scale)

    def mean(self) -> float:
        return self.loc

    def quantiles(self, levels=LEVELS) -> np.ndarray:
        return self.loc + self.scale * special.ndtri(np.asarray(levels, dtype=float))

    def crps(self, y: float) -> float:
        return float(crps_normal(y, self.loc, self.scale))


class Weighted:
    """sum_i w_i [x_i <= x]: a weighted empirical distribution."""

    def __init__(self, values, weights=None):
        values = np.asarray(values, dtype=float)
        w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        order = np.argsort(values, kind="stable")
        self.x = values[order]
        self.w = w[order] / w.sum()
        self.cw = np.cumsum(self.w)

    def cdf(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        i = np.searchsorted(self.x, t, side="right")
        return np.where(i > 0, self.cw[np.maximum(i - 1, 0)], 0.0)

    def mean(self) -> float:
        return float(np.dot(self.w, self.x))

    def quantiles(self, levels=LEVELS) -> np.ndarray:
        """inf{x : F(x) >= level} (Meinshausen 2006, eq. 1), with 1e-12 of
        slack for the rounding of the cumulative weights."""
        i = np.searchsorted(self.cw, np.asarray(levels, dtype=float) - 1e-12, side="left")
        return self.x[np.minimum(i, len(self.x) - 1)]

    def crps(self, y: float) -> float:
        """The energy form: sum w|x - y| - sum_{i<j} w_i w_j (x_j - x_i)."""
        c = np.concatenate(([0.0], self.cw[:-1]))
        d = np.concatenate(([0.0], np.cumsum(self.w * self.x)[:-1]))
        pairs = float(np.dot(self.w, self.x * c - d))
        return float(np.dot(self.w, np.abs(self.x - float(y)))) - pairs

    def crps_integral(self, y: float) -> float:
        """The same CRPS as the integral of (F - [t >= y])^2, piece by piece."""
        pts = np.unique(np.concatenate((self.x, [float(y)])))
        F = self.cdf(pts[:-1])
        H = (pts[:-1] >= float(y)).astype(float)
        return float(np.dot((F - H) ** 2, np.diff(pts)))


class Transformed:
    """Y = g(Z), g increasing on its domain, for Z with distribution `base` (a
    MixtureT) conditioned on the domain: where Z's mixture of t reaches past
    g's domain, as a Box-Cox transform's at lam > 0 bounds it below, the part
    beyond is cut off and the rest renormalised, so that Y has the power-t
    analogue of the power-normal distribution, the inverse Box-Cox of a
    truncated normal (Freeman & Modarres 2006). Its mean and CRPS are
    integrals on Y's own scale, where F_Y(t) is smooth, panelled by the
    components' scales carried through g'."""

    def __init__(self, base: MixtureT, g: Callable, g_inv: Callable, g_prime: Callable,
                 domain: tuple = (-math.inf, math.inf)):
        self.base, self.g, self.g_inv, self.g_prime = base, g, g_inv, g_prime
        self.domain = domain
        lo, hi = domain
        self.p_below = float(base.cdf(lo)[0]) if math.isfinite(lo) else 0.0
        self.p_above = float(1.0 - base.cdf(hi)[0]) if math.isfinite(hi) else 0.0
        self._fast = ((float(base.fast_cdf(lo)[0]) if math.isfinite(lo) else 0.0),
                      (float(1.0 - base.fast_cdf(hi)[0]) if math.isfinite(hi) else 0.0))

    @property
    def p_outside(self) -> float:
        """The share of the base distribution beyond g's domain, cut off."""
        return self.p_below + self.p_above

    def cdf(self, y):
        """The distribution function, by stdtr."""
        F = self.base.cdf(self.g_inv(np.atleast_1d(np.asarray(y, dtype=float))))
        return np.clip((F - self.p_below) / (1.0 - self.p_outside), 0.0, 1.0)

    def fast_cdf(self, y):
        below, above = self._fast
        F = self.base.fast_cdf(self.g_inv(np.atleast_1d(np.asarray(y, dtype=float))))
        return np.clip((F - below) / (1.0 - below - above), 0.0, 1.0)

    def quantiles(self, levels=LEVELS) -> np.ndarray:
        levels = np.asarray(levels, dtype=float)
        return self.g(self.base.quantiles(self.p_below + levels * (1.0 - self.p_outside)))

    def _range(self) -> tuple[float, float, int]:
        """The support on Y's scale and the number of panels across it."""
        a, b = self.base.support()
        a, b = max(a, self.domain[0]), min(b, self.domain[1])
        ya, yb = (float(v) for v in self.g(np.array([a, b])))
        local = self.base.scale * self.g_prime(self.base.loc)
        local = local[local > 0]
        width = PANEL * local.min() if len(local) else (yb - ya) / MAX_PANELS
        return ya, yb, int(min(MAX_PANELS, max(8, math.ceil((yb - ya) / width))))

    def mean(self) -> float:
        """g(a) + the integral of 1 - F_Y over the support [g(a), g(b)]."""
        ya, yb, n = self._range()
        return ya + integrate(lambda t: 1.0 - self.fast_cdf(t), ya, yb, n)

    def crps(self, y: float) -> float:
        """The integral of (F_Y(t) - [t >= y])^2, split at y."""
        ya, yb, n = self._range()
        y = float(y)
        left = integrate(lambda t: self.fast_cdf(t) ** 2, min(ya, y), y, n)
        right = integrate(lambda t: (1.0 - self.fast_cdf(t)) ** 2, y, max(yb, y), n)
        return left + right


def boxcox_maps(lam: float):
    """g, g^-1, g' and g's domain for the Box-Cox transform at lam >= 0 (the
    logarithm at 0): at lam > 0 the domain is lam z + 1 > 0, positive stocks."""
    if lam < 0:
        raise ValueError("D6 transforms only at lam >= 0")
    if abs(lam) < 1e-12:
        return np.exp, lambda y: np.log(np.maximum(y, 1e-300)), np.exp, (-math.inf, math.inf)

    def g(z):
        return np.maximum(lam * np.asarray(z, dtype=float) + 1.0, 0.0) ** (1.0 / lam)

    def g_inv(y):
        return (np.maximum(np.asarray(y, dtype=float), 0.0) ** lam - 1.0) / lam

    def g_prime(z):
        b = np.maximum(lam * np.asarray(z, dtype=float) + 1.0, 0.0)
        return np.where(b > 0, np.where(b > 0, b, 1.0) ** (1.0 / lam - 1.0), 0.0)
    return g, g_inv, g_prime, (-1.0 / lam, math.inf)


# --- scores over many locations -------------------------------------------------------------

def coverage(quantiles: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QCP at the 39 levels and PICP for the 19 central intervals, from each
    location's 39 quantiles (one row per location)."""
    q, y = np.asarray(quantiles, dtype=float), np.asarray(y, dtype=float)
    below = y[:, None] <= q
    qcp = below.mean(axis=0)
    picp = np.array([np.mean((q[:, lo] <= y) & (y <= q[:, hi]))
                     for lo, hi in (central_bounds(p) for p in CENTRAL)])
    return qcp, picp


def point_scores(prediction, y) -> dict:
    """ME (prediction minus observation), RMSE and MEC."""
    p, y = np.asarray(prediction, dtype=float), np.asarray(y, dtype=float)
    e = p - y
    return {"me": float(e.mean()), "rmse": float(np.sqrt(np.mean(e * e))),
            "mec": float(1.0 - np.sum(e * e) / np.sum((y - y.mean()) ** 2))}
