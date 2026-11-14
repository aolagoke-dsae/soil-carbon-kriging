"""Rule G1's Python side on its fixture; the R side where R and gstat are installed."""
import json
import shutil
import subprocess

import numpy as np
import pytest

import gstat as G


@pytest.fixture(scope="module")
def fixture():
    return json.loads(G.FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_and_this_repositorys_side(fixture):
    assert len(fixture["z"]) == 245 and len(fixture["cx"]) == 2903 and "CC BY 4.0" in fixture["source"]
    o = G.ours(fixture)
    assert np.all(np.isfinite(o["pred"])) and np.all(o["var"] > 0) and o["block_var"] > 0
    assert min(fixture["z"]) < o["block_pred"] < max(fixture["z"])


def _gstat_available():
    """R alone is not enough: GitHub's Windows runners have R without gstat."""
    if shutil.which("Rscript") is None:
        return False
    try:
        r = subprocess.run(["Rscript", "-e", "library(gstat); library(sp); library(jsonlite)"],
                           capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


@pytest.mark.skipif(not _gstat_available(), reason="R with gstat, sp and jsonlite is not installed")
def test_g1_against_gstat(fixture):
    ok, detail = G.check(fixture)
    assert ok, detail
