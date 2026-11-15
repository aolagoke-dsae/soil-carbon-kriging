"""Rules L1 and L2: this side on the fixture; the R side where R and nlme are installed."""
import json
import shutil
import subprocess

import numpy as np
import pytest

import nlme as N


@pytest.fixture(scope="module")
def fixture():
    return json.loads(N.FIXTURE.read_text(encoding="utf-8"))


def _nlme_available():
    if shutil.which("Rscript") is None:
        return False
    try:
        r = subprocess.run(["Rscript", "-e", "library(nlme); library(jsonlite)"], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def test_the_fixture_and_this_side(fixture):
    assert sum(len(f["z"]) for f in fixture["fields"]) == 1163 and "CC BY 4.0" in fixture["source"]
    assert {f["site"] for f in fixture["fields"] if "ours" in f} == set(N.LARGE)
    mine = N.ours(fixture)
    assert len(mine) == 16 and all(len(m["loglik"]) == 9 and np.all(np.isfinite(m["loglik"])) for m in mine)


def test_compare_ignores_a_constant_offset():
    mine = [{"site": "A", "mean": "constant", "loglik": [1.0, 2.0, 3.0]}]
    theirs = {"l1": [{"site": "A", "mean": "constant", "loglik": [6.0, 7.0, 8.0 + 1e-3]}]}
    worst, offsets = N.compare(mine, theirs)
    assert worst == pytest.approx(1e-3) and offsets[0] == pytest.approx(5.0, abs=1e-3)


@pytest.mark.skipif(not _nlme_available(), reason="R with nlme and jsonlite is not installed")
def test_l1_against_nlme(fixture):
    mine = N.ours(fixture)
    worst, _ = N.compare(mine, N.nlme(fixture, mine))
    assert worst <= N.TOL
