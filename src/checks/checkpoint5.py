"""Checkpoint 5: each field's mean stock (RUNBOOK, Stage 5).

The rules were fixed in RUNBOOK.md on 10.11.2026, before any field mean was
computed: B1 block algebra, B2 block predictive, B3 calibration by
simulation, E1 effort, R1 record. G1, gstat in R, runs on its own
(python -m src.checks.gstat), where R is installed.

    python -m src.checks.checkpoint5 [--budget SECONDS] [--only b1,b3,e1]
        exit 0: every rule passed; 1: a rule failed; 2: an input is missing;
        4: stopped at the budget, run again to continue from the cache

The recomputations are cached in curated/checkpoint5_cache/.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import datetime as dt
import math
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

try:                        # run as a module from the repository root
    from src.checks import checkpoint1 as C1
    from src.transform import field_means as FM, folds as FO, kriging as K, maps as MP, reml as M, scores as SC
except ImportError:         # imported with src/* on sys.path (pytest)
    import checkpoint1 as C1
    import field_means as FM
    import folds as FO
    import kriging as K
    import maps as MP
    import reml as M
    import scores as SC

Rule = C1.Rule
CACHE = K.CURATED / "checkpoint5_cache"
CODE = K.code_hash(__file__, FM.__file__, MP.__file__)
TOL_B1_PRED, TOL_B1_VAR, TOL_B1_BIN = 1e-9, 1e-6, 1e-6
TOL_B2 = 1e-9
B3_FIELDS, B3_STEP, B3_SITE, B3_SEED = 400, 25, "IL-MC", FO.SEED + 5
TOL_B3 = {0.5: 0.07, 0.9: 0.04, 0.95: 0.03}
TOL_E1 = 1e-6
E1_N = 100
PARTS = ("b1", "b3", "e1")


class NotThere(Exception):
    """A curated table Checkpoint 5 reads is missing."""


def load() -> dict:
    for p in (FM.OUT, FM.OUT_EFFORT, MP.OUT, K.OUT_SELECTION):
        if not p.exists():
            raise NotThere(p.relative_to(K.ROOT).as_posix())
    fields, models, selected = FM.load_inputs()
    return {"fields": fields, "models": models, "selected": selected,
            "means": pd.read_parquet(FM.OUT), "effort": pd.read_parquet(FM.OUT_EFFORT)}


def _key(*parts) -> list:
    return ["checkpoint5", CODE] + list(parts)


def _model(d: dict, site: str) -> K.Model:
    return next(m for m in d["models"][site] if m.name == d["selected"][site])


def _theta(model: K.Model, g: int) -> tuple:
    a, t, nu = model.grid().arrays()
    return float(a[g]), (None if np.isnan(nu[g]) else float(nu[g])), float(t[g])


# --- B1 -------------------------------------------------------------------------------------

def _b1_site(d: dict, site: str, cache: K.Cache) -> dict:
    model = _model(d, site)
    f, res = FM.field_mean(d["fields"], site, model, K.Cache(FM.CACHE))
    fd, cells, cov_c = f["fd"], f["cells_xy"], f["cells_cov"]
    g = np.array(res["g"])
    w = f["lp"][g]
    order = np.argsort(-w)
    picks = [int(order[0])] + [int(i) for i in np.linspace(0, len(g) - 1, 6)[1:-1].astype(int)]

    def compute():
        X, X0 = K.design(model.mean, fd.xy, cells, fd.cov, cov_c, f["drift"])
        n, p = X.shape
        Dss = K.distances(fd.xy, fd.xy, model.angle, model.ratio)
        Dsc = K.distances(cells, fd.xy, model.angle, model.ratio)
        out = {"pred": 0.0, "var": 0.0, "bin": 0.0}
        for j, i in enumerate(picks):
            a, nu, tau = _theta(model, int(g[i]))
            exact = FM.exact_mean_correlation(cells, model.family, a, nu, model.angle, model.ratio)
            binned = FM.mean_correlation(f["hist"], len(cells), model.family, a, nu)
            out["bin"] = max(out["bin"], abs(binned - exact) / exact)
            if j:
                continue
            C = (1 - tau) * M.correlation(model.family, Dss, a, nu) + tau * np.eye(n)
            kb = (1 - tau) * M.correlation(model.family, Dsc, a, nu).mean(axis=0)
            xb = X0.mean(axis=0)
            A = np.block([[C, X], [X.T, np.zeros((p, p))]])
            sol = np.linalg.solve(A, np.r_[kb, xb])
            pred = float(sol[:n] @ fd.z)
            var = (1 - tau) * exact - float(sol[:n] @ kb) - float(sol[n:] @ xb)
            out["pred"] = abs(res["pred"][i] - pred) / abs(pred)
            out["var"] = abs(res["v"][i] - var) / var
        return out
    return cache.get(_key("b1", site, model.key()) + f["data"], compute)


def b1_block(d: dict, cache: K.Cache) -> tuple[Rule, pd.DataFrame]:
    t = pd.DataFrame([{"site": s, **_b1_site(d, s, cache)} for s in sorted(d["fields"])])
    ok = (t["pred"] <= TOL_B1_PRED).all() and (t["var"] <= TOL_B1_VAR).all() and (t["bin"] <= TOL_B1_BIN).all()
    detail = (f"at every field's posterior mode, the block predictor within {t['pred'].max():.1e} and its variance "
              f"within {t['var'].max():.1e} (relative) of the bordered system with every pair of cells; the binned "
              f"mean correlation within {t['bin'].max():.1e} of the unbinned at five grid points per field")
    return Rule("B1 block algebra", bool(ok), detail), t


# --- B2 -------------------------------------------------------------------------------------

def b2_predictive(d: dict) -> Rule:
    worst_cdf = worst_mean = 0.0
    for site in sorted(d["fields"]):
        model = _model(d, site)
        f, res = FM.field_mean(d["fields"], site, model, K.Cache(FM.CACHE))
        g = np.array(res["g"])
        w = np.exp(f["lp"][g] - f["lp"][g].max())
        w /= w.sum()
        locs = np.array(res["pred"])
        mix = SC.MixtureT(w, locs, np.sqrt(np.array(res["s2"]) * np.array(res["v"])), res["n"] - res["p"])
        q = mix.quantiles(list(FM.LEVELS))
        worst_cdf = max(worst_cdf, float(np.max(np.abs(mix.cdf(q) - np.array(FM.LEVELS)))))
        worst_mean = max(worst_mean, abs(mix.mean() - float(w @ locs)) / abs(float(w @ locs)))
        rec = d["means"].set_index("site").loc[site]
        worst_mean = max(worst_mean, abs(rec["mean"] - mix.mean()) / abs(mix.mean()))
    ok = worst_cdf <= TOL_B2 and worst_mean <= TOL_B2
    return Rule("B2 block predictive", ok,
                f"in all eight fields the distribution function returns 2.5, 50 and 97.5 % at the quantiles within "
                f"{worst_cdf:.1e}, and the mean is the posterior-weighted mean of the block predictors, as the record "
                f"holds it, within {worst_mean:.1e} (relative)")


# --- B3 -------------------------------------------------------------------------------------

def _b3_step(d: dict, start: int) -> list:
    fd0 = d["fields"][B3_SITE]
    own = d["models"][B3_SITE][0]
    model = K.Model("sim", "constant", "exponential", own.lo, own.hi)
    grid = model.grid()
    a_arr, t_arr, _ = grid.arrays()
    cells = MP.hull_cells(fd0.xy, MP.load_grid(B3_SITE))[2]
    hist = FM.pair_histogram(cells)
    allp = np.r_[fd0.xy, cells]
    D = K.distances(allp, allp)
    n = len(fd0.z)
    out = []
    for k in range(start, min(start + B3_STEP, B3_FIELDS)):
        rng = np.random.default_rng([B3_SEED, k])
        g = int(rng.integers(grid.size))
        a, tau = float(a_arr[g]), float(t_arr[g])
        L = np.linalg.cholesky((1 - tau) * M.correlation("exponential", D, a, None) + 1e-10 * np.eye(len(allp)))
        W = L @ rng.normal(size=len(allp))
        z = W[:n] + math.sqrt(tau) * rng.normal(size=n)
        target = float(W[n:].mean())
        fd = K.FieldData("sim", fd0.ids, fd0.xy, z, fd0.cov)
        X = K.design("constant", fd.xy, fd.xy)[0]
        lp = K.evaluate(grid, fd.xy, z, X).logpost
        res = FM.block_grid(fd, model, cells, fd0.cov[:1].repeat(len(cells), axis=0), (), lp, hist)
        gg = np.array(res["g"])
        w = np.exp(lp[gg] - lp[gg].max())
        w /= w.sum()
        s2, v, locs = np.array(res["s2"]), np.array(res["v"]), np.array(res["pred"])
        mix = SC.MixtureT(w, locs, np.sqrt(s2 * v), n - 1)
        top = int(np.argmax(w))
        plug = SC.Normal(locs[top], math.sqrt(s2[top] * v[top]))
        out.append([float(mix.cdf(target)[0]), float(plug.cdf(target)[0])])
    return out


def b3_calibration(d: dict, cache: K.Cache) -> tuple[Rule, pd.DataFrame]:
    pit = []
    for start in range(0, B3_FIELDS, B3_STEP):
        pit += cache.get(_key("b3", start, B3_FIELDS, B3_STEP), lambda: _b3_step(d, start))
    pit = np.array(pit)
    rows, ok = [], True
    for p, tol in TOL_B3.items():
        bay = float(np.mean(np.abs(pit[:, 0] - 0.5) <= p / 2))
        plug = float(np.mean(np.abs(pit[:, 1] - 0.5) <= p / 2))
        ok &= abs(bay - p) <= tol
        rows.append({"interval": p, "bayesian": bay, "plug_in": plug, "tolerance": tol})
    t = pd.DataFrame(rows)
    detail = (f"{B3_FIELDS} fields at {B3_SITE}'s locations and cells, parameters from the prior: the Bayesian block "
              f"predictive holds the field mean at 50, 90 and 95 % in "
              + ", ".join(f"{r.bayesian:.3f}" for r in t.itertuples())
              + " of them; plug-in kriging " + ", ".join(f"{r.plug_in:.3f}" for r in t.itertuples()))
    return Rule("B3 calibration by simulation", bool(ok), detail), t


# --- E1 -------------------------------------------------------------------------------------

def _e1_site(d: dict, site: str, cache: K.Cache) -> dict:
    model = _model(d, site)
    f, res = FM.field_mean(d["fields"], site, model, K.Cache(FM.CACHE))
    thetas, we = FM.heavy_points(model, res, f["lp"])
    pts = FM.planning_grid(f["cells_xy"], f["fd"].xy, E1_N, f["cell"])
    pts_cov = f["cells_cov"][FM.nearest_cells(pts, f["cells_xy"])]
    starts = [i for i in range(len(thetas)) if i == 0 or thetas[i][:2] != thetas[i - 1][:2]]
    bounds = starts[::10] + [len(thetas)]
    dist = None
    v, rbb = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        def compute():
            nonlocal dist
            if dist is None:
                dist = pdist(FM.transformed(f["cells_xy"] - f["cells_xy"].mean(axis=0), model.angle, model.ratio))
            N = len(f["cells_xy"])
            exact = {}
            for a, nu, _ in thetas[lo:hi]:
                if (a, nu) not in exact:
                    exact[(a, nu)] = float((N + 2.0 * M.correlation(model.family, dist, a, nu).sum()) / N ** 2)
            X, X0 = K.design(model.mean, pts, f["cells_xy"], pts_cov, f["cells_cov"], f["drift"])
            npts, p = X.shape
            xb = X0.mean(axis=0)
            Dpp = K.distances(pts, pts, model.angle, model.ratio)
            Dpc = K.distances(f["cells_xy"], pts, model.angle, model.ratio)
            vv, corr = [], {}
            for a, nu, tau in thetas[lo:hi]:
                if (a, nu) not in corr:
                    corr = {(a, nu): (M.correlation(model.family, Dpp, a, nu),
                                      M.correlation(model.family, Dpc, a, nu).mean(axis=0))}
                R, rb = corr[(a, nu)]
                k = (1 - tau) * rb
                A = np.block([[(1 - tau) * R + tau * np.eye(npts), X], [X.T, np.zeros((p, p))]])
                sol = np.linalg.solve(A, np.r_[k, xb])
                vv.append((1 - tau) * exact[(a, nu)] - float(sol[:npts] @ k) - float(sol[npts:] @ xb))
            return {"v": vv, "rbb": [exact[(a, nu)] * (1 - tau) for a, nu, tau in thetas[lo:hi]]}
        part = cache.get(_key("e1", site, model.key(), lo, hi) + f["data"], compute)
        v += part["v"]
        rbb += part["rbb"]
    V = float(we @ np.array(v))
    rec = d["effort"]
    rv = float(rec[(rec["site"] == site) & (rec["n_target"] == E1_N)]["V"].iloc[0])
    return {"site": site, "n": len(pts), "V": abs(V - rv) / rv}


def e1_effort(d: dict, cache: K.Cache) -> tuple[Rule, pd.DataFrame]:
    t = pd.DataFrame([_e1_site(d, s, cache) for s in sorted(d["fields"])])
    e, m = d["effort"], d["means"]
    finite = bool(np.isfinite(e["V"]).all() and (e["V"] > 0).all() and (m["d2"] > 0).all())
    ok = finite and (t["V"] <= TOL_E1).all()
    return Rule("E1 effort", bool(ok),
                f"at the grid of about {E1_N} samples in every field, V(n) with every pair of cells unbinned and a "
                f"direct solve within {t['V'].max():.1e} (relative); every V(n) and D² positive and finite"), t


# --- R1 -------------------------------------------------------------------------------------

def r1_record(d: dict) -> Rule:
    m, e = d["means"], d["effort"]
    cols = ["mean", "sd", "q025", "q50", "q975", "plugin_mean", "plugin_sd", "sample_mean", "sample_se",
            "sample_lo", "sample_hi", "sample_se_model", "mdc", "d2"]
    problems = []
    if sorted(m["site"]) != sorted(d["fields"]):
        problems.append("not one row per field")
    if not np.isfinite(m[cols].to_numpy(float)).all():
        problems.append("a value not finite")
    if not ((m["q025"] < m["q50"]) & (m["q50"] < m["q975"]) & (m["sample_lo"] < m["sample_hi"])).all():
        problems.append("an interval out of order")
    if not np.isfinite(e[["V", "se_grid", "se_srs", "mdc_grid", "mdc_srs"]].to_numpy(float)).all():
        problems.append("an effort value not finite")
    ok = not problems
    detail = (f"{len(m)} field means, {len(e)} effort rows ({e['n_target'].nunique()} grids per field at most), every "
              f"value finite and every interval ordered") if ok else "; ".join(problems)
    return Rule("R1 record", ok, detail)


# --- the run --------------------------------------------------------------------------------

def run(budget: Optional[float] = None, only=None) -> tuple[list, dict]:
    d = load()
    cache = K.Cache(CACHE, None if budget is None else time.monotonic() + budget)
    steps = {"b1": lambda: b1_block(d, cache), "b3": lambda: b3_calibration(d, cache), "e1": lambda: e1_effort(d, cache)}
    if only:
        for part in only:
            steps[part]()
        return [], {}
    b1, t1 = b1_block(d, cache)
    b3, t3 = b3_calibration(d, cache)
    e1, te = e1_effort(d, cache)
    rules = [b1, b2_predictive(d), b3, e1, r1_record(d)]
    tables = {"B1 per field (relative errors):": t1.assign(**{c: t1[c].map(lambda v: f"{v:.1e}") for c in ("pred", "var", "bin")}),
              "B3: the share of simulated field means inside the intervals:": t3.round(3),
              "E1 per field:": te.assign(V=te["V"].map(lambda v: f"{v:.1e}")),
              "The field means (Mg C/ha) and what they mean for monitoring:": d["means"].round(3)}
    return rules, tables


def report(rules: list, tables: Optional[dict], when: Optional[dt.datetime] = None) -> str:
    when = when or dt.datetime.now(dt.timezone.utc)
    lines = [f"Checkpoint 5, run {when:%Y-%m-%dT%H:%M:%SZ}", ""]
    lines += [f"{'PASS' if r.passed else 'FAIL'}  {r.rule}: {r.detail}" for r in rules]
    if tables:
        lines += ["", "Reported, not judged:"]
        for title, table in tables.items():
            lines += ["", title, table.to_string(index=False)]
    passed = sum(r.passed for r in rules)
    lines += ["", f"Checkpoint 5 {'PASSED' if passed == len(rules) else 'FAILED'}: {passed}/{len(rules)} rules"]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """python -m src.checks.checkpoint5 [--budget SECONDS] [--only b1,b3,e1]"""
    budget, only = None, None
    args = list(argv)
    while args:
        if len(args) >= 2 and args[0] == "--budget":
            budget = float(args[1])
        elif len(args) >= 2 and args[0] == "--only" and set(args[1].split(",")) <= set(PARTS):
            only = args[1].split(",")
        else:
            print(main.__doc__)
            return 2
        args = args[2:]
    try:
        rules, tables = run(budget, only)
    except NotThere as e:
        print(f"NOT THERE: {e}")
        return 2
    except K.OutOfTime:
        print(f"Stopped at the {budget:g} s budget; run again to continue from the cache.")
        return 4
    if only:
        print(f"{', '.join(only)}: in the cache")
        return 0
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(report(rules, tables))
    return 0 if all(r.passed for r in rules) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
