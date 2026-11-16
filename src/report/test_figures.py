"""The report's figures redrawn from the tables committed beside them
(docs/figures/*.csv), so that a change to figures.py or to the environment
that breaks a figure fails CI. The map and the variograms need the curated
tables, which are not committed, and are drawn by `python -m src.report.figures`
only. Added 15.11.2026 after the review of v1.0 (docs/REVIEW_V1.md, section 3)."""
import pandas as pd
import pytest

from src.report import figures as FG

TABLES = FG.FIGURES


@pytest.fixture(scope="module")
def means():
    return pd.read_csv(TABLES / "field_means.csv")


@pytest.fixture(scope="module")
def selected(means):
    return means.set_index("site")["model"].sort_index()


def _drawn(path):
    assert path.exists() and path.stat().st_size > 20_000
    return True


def test_calibration(tmp_path, selected):
    out = tmp_path / "calibration.png"
    FG.draw_calibration(pd.read_csv(TABLES / "calibration.csv"), selected, out)
    assert _drawn(out)


def test_crps_by_field(tmp_path, selected):
    table = pd.read_csv(TABLES / "crps_by_field.csv")
    assert set(table["site"]) == set(selected.index)
    out = tmp_path / "crps_by_field.png"
    FG.draw_crps(table, selected, out)
    assert _drawn(out)


def test_field_means(tmp_path, means):
    out = tmp_path / "field_means.png"
    FG.draw_field_means(means, out)
    assert _drawn(out)


def test_detectable_change(tmp_path, means):
    out = tmp_path / "detectable_change.png"
    FG.draw_detectable_change(pd.read_csv(TABLES / "detectable_change.csv"), means, out)
    assert _drawn(out)


def test_bze_map(tmp_path):
    table = pd.read_csv(TABLES / "bze_map.csv.gz")
    assert set(table["land_use"]) == {"A", "G"} and (table["q0.05"] <= table["q0.95"]).all()
    out = tmp_path / "bze_map.png"
    FG.draw_bze_map(table, out)
    assert _drawn(out)
