"""Review of v1.0, M3: the field-mean method against held-out samples
(RUNBOOK, Stage 5, Addendum on M3, H1-H5; rules fixed 15.11.2026 before this
was written).

At each grid field, 20 random splits of the locations into halves A and B,
each used both ways: Bayesian kriging fitted to A alone predicts the mean of
B's stocks (the nugget included), beside A's sample mean with its t
predictive; both are scored against B's observed mean.

    python -m src.transform.halves [--budget SECONDS]
        writes curated/potash_halves.parquet and prints H4 and H5. Every
        split is cached, so a run stopped by --budget (exit 4) continues.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
import time

import numpy as np
import pandas as pd

try:                        # run as a module from the repository root
    from src.transform import field_means as FM, kriging as K, reml as M, scores as SC
except ImportError:         # imported with src/transform on sys.path (pytest)
    import field_means as FM
    import kriging as K
    import reml as M
    import scores as SC

OUT = K.CURATED / "potash_halves.parquet"
CACHE = K.CURATED / "halves_cache"
SITES = ("IL-DG", "IL-PT", "IL-RD", "IL-RS", "IL-RT")      # H1
SPLITS = 20                                                 # H2
SEED = 20261115                                             # H2
KEEP = FM.KEEP
COVER = 0.90                                                # H5
ORCHESTRATION = ("main", "run")
_CODE = K.code_hash(__file__, skip=ORCHESTRATION)


def splits(n: int, count: int = SPLITS, seed: int = SEED) -> list[np.ndarray]:
    """H2: `count` random halves of range(n) (the first n // 2 of a permutation)."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.permutation(n)[: n // 2]) for _ in range(count)]


def point_set_mean(model: K.Model, xyA, zA, covA, xyB, covB, drift: tuple, lp: np.ndarray) -> dict:
    """H3: at every kept grid point, the universal-kriging predictor of B's
    mean from A and its variance per unit sill, the nugget included
    ((1 - tau) mean R_BB + tau / n_B), with S^2 / (n_A - p)."""
    X, X0 = K.design(model.mean, xyA, xyB, covA, covB, drift)
    n, p = X.shape
    nB = len(xyB)
    xB = X0.mean(axis=0)
    D = K.distances(xyA, xyA, model.angle, model.ratio)
    DBA = K.distances(xyB, xyA, model.angle, model.ratio)
    DBB = K.distances(xyB, xyB, model.angle, model.ratio)
    keep = lp >= lp.max() + np.log(KEEP)
    out = {k: [] for k in ("g", "pred", "v", "s2")}
    g = 0
    for a, nu, taus in model.grid().points():
        if not keep[g:g + len(taus)].any():
            g += len(taus)
            continue
        R = M.correlation(model.family, D, a, nu)
        lam, U = np.linalg.eigh(R)
        rb = M.correlation(model.family, DBA, a, nu).mean(axis=0)
        rbb = float(M.correlation(model.family, DBB, a, nu).mean())
        Uz, UX, Ub = U.T @ zA, U.T @ X, U.T @ rb
        for tau in taus:
            if keep[g]:
                dinv = 1.0 / ((1.0 - tau) * lam + tau)
                XCX = UX.T @ (dinv[:, None] * UX)
                beta = np.linalg.solve(XCX, UX.T @ (dinv * Uz))
                r = Uz - UX @ beta
                kw = (1.0 - tau) * Ub
                gv = xB - UX.T @ (dinv * kw)
                v = (1.0 - tau) * rbb + tau / nB - float(kw @ (dinv * kw)) + float(gv @ np.linalg.solve(XCX, gv))
                out["g"].append(g)
                out["pred"].append(float(xB @ beta + kw @ (dinv * r)))
                out["v"].append(max(v, 0.0))
                out["s2"].append(float(r @ (dinv * r)) / (n - p))
            g += 1
    out["n"], out["p"] = n, p
    return out


def kriging_predictive(res: dict, lp: np.ndarray) -> SC.MixtureT:
    g = np.array(res["g"])
    w = np.exp(lp[g] - lp[g].max())
    w /= w.sum()
    return SC.MixtureT(w, np.array(res["pred"]), np.sqrt(np.array(res["s2"]) * np.array(res["v"])),
                       float(res["n"] - res["p"]))


def sample_mean_predictive(zA: np.ndarray, nB: int) -> tuple[float, float, float]:
    """H3: (loc, scale, df) of A's mean as a predictive of B's."""
    nA = len(zA)
    return float(zA.mean()), float(zA.std(ddof=1) * math.sqrt(1 / nA + 1 / nB)), float(nA - 1)


def t_scores(y: float, loc: float, scale: float, df: float) -> dict:
    q = {lv: loc + scale * SC.t_ppf(lv, df) for lv in (0.025, 0.1, 0.9, 0.975)}
    return {"crps": float(SC.crps_t(y, loc, scale, df)), "in95": bool(q[0.025] <= y <= q[0.975]),
            "in80": bool(q[0.1] <= y <= q[0.9]), "width95": float(q[0.975] - q[0.025])}


def mixture_scores(y: float, mix: SC.MixtureT) -> dict:
    q = mix.quantiles([0.025, 0.1, 0.9, 0.975])
    return {"crps": float(mix.crps(y)), "in95": bool(q[0] <= y <= q[3]), "in80": bool(q[1] <= y <= q[2]),
            "width95": float(q[3] - q[0])}


def one(fd: K.FieldData, model: K.Model, A: np.ndarray, B: np.ndarray) -> dict:
    """One prediction of B's mean from A, by both methods."""
    xyA, zA, covA = fd.xy[A], fd.z[A], fd.cov[A]
    drift = (K.drift_candidates()[K.choose(K.ml_table(model.grid(), xyA, zA, covA))]
             if model.mean == "drift" else ())
    X = K.design(model.mean, xyA, xyA[:1], covA, covA[:1], drift)[0]
    lp = K.evaluate(model.grid(), xyA, zA, X, angle=model.angle, ratio=model.ratio).logpost
    res = point_set_mean(model, xyA, zA, covA, fd.xy[B], fd.cov[B], drift, lp)
    y = float(fd.z[B].mean())
    k = mixture_scores(y, kriging_predictive(res, lp))
    s = t_scores(y, *sample_mean_predictive(zA, len(B)))
    return {"y": y, "drift": [int(c) for c in drift], **{f"k_{key}": val for key, val in k.items()},
            **{f"s_{key}": val for key, val in s.items()}}


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """H4 per field and pooled."""
    cols = ["k_in95", "s_in95", "k_in80", "s_in80", "k_crps", "s_crps", "k_width95", "s_width95"]
    per = rows.groupby("site")[cols].mean()
    per.loc["pooled"] = rows[cols].mean()
    per["width_ratio"] = per["k_width95"] / per["s_width95"]
    return per


def verdict(per: pd.DataFrame) -> tuple[bool, str]:
    """H5 on the pooled row."""
    p = per.loc["pooled"]
    ok = bool(p["k_in95"] >= COVER and p["k_crps"] <= p["s_crps"])
    return ok, (f"pooled: kriging's 95 % intervals hold {p['k_in95']:.3f} of B's means (rule: >= {COVER}), "
                f"CRPS {p['k_crps']:.3f} against the sample mean's {p['s_crps']:.3f}: "
                + ("SUPPORTED by held-out samples (H5)" if ok else "NOT SUPPORTED by held-out samples (H5)"))


def run(fields: dict, models: dict, selected, cache: K.Cache, log=lambda m: None) -> pd.DataFrame:
    rows = []
    for site in SITES:
        t0 = time.monotonic()
        fd = fields[site]
        model = next(m for m in models[site] if m.name == selected[site])
        for i, half in enumerate(splits(len(fd.z))):
            other = np.setdiff1d(np.arange(len(fd.z)), half)
            for way, (A, B) in enumerate(((half, other), (other, half))):
                r = cache.get(["halves", _CODE, site, model.key(), i, way, K.data_key(fd.xy, fd.z, fd.cov)],
                              lambda: one(fd, model, A, B))
                rows.append({"site": site, "split": i, "way": way, "nA": len(A), "nB": len(B), **r})
        log(f"{site} ({model.name}): {time.monotonic() - t0:.0f} s")
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    budget = None
    if argv[:1] == ["--budget"] and len(argv) == 2:
        budget = float(argv[1])
    elif argv:
        print(__doc__)
        return 2
    fields, models, selected = FM.load_inputs()
    cache = K.Cache(CACHE, None if budget is None else time.monotonic() + budget)
    try:
        rows = run(fields, models, selected, cache, log=print)
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget; run again to continue from the cache.")
        return 4
    rows.to_parquet(OUT, index=False)
    per = summarise(rows)
    pd.set_option("display.width", 220)
    print(per.round(3).to_string())
    print(verdict(per)[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
