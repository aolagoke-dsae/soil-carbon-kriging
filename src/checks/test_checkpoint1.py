"""Checkpoint 1 on fixtures: a clean pair of datasets passes every rule, and
each rule fails on the defect it exists to catch. The fixtures are made up;
only their shape follows the two sources. No network is used."""
import datetime as dt
import hashlib
import json

import pytest

import checkpoint1 as C

UTC = dt.timezone.utc
WHEN = dt.datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)
LATER = WHEN + dt.timedelta(hours=1)
SITES = [f"S{i}" for i in range(1, 9)]


def _loc_rows():
    return [(s, str(j), -88.2 + 0.0005 * j, 40.1 + 0.01 * k)
            for k, s in enumerate(SITES) for j in range(1, 4)]


def _meas_rows():
    return [(s, str(j), a, b, 1.5, 1.3)
            for s in SITES for j in range(1, 4) for a, b in ((0, 15), (15, 30))]


def _locations(rows=None):
    rows = _loc_rows() if rows is None else rows
    return ("site,location_id,X,Y\n"
            + "".join(f"{s},{j},{x},{y}\n" for s, j, x, y in rows)).encode()


def _measurements(rows=None):
    rows = _meas_rows() if rows is None else rows
    return ("site,location_id,sample_depth_min,sample_depth_max,SOCc,BD\n"
            + "".join(f"{s},{j},{a},{b},{c},{d}\n" for s, j, a, b, c, d in rows)).encode()


def _feature(pid, x=None, y=None, kv=50.0, kv2="40", lu="A", gx=None):
    x = 400000.0 + 1000 * pid if x is None else x
    y = 5500000.0 + 1000 * pid if y is None else y
    return {"type": "Feature", "id": f"bze_lw_standorte_verschleiert.{pid}",
            "geometry": {"type": "Point", "coordinates": [x if gx is None else gx, y]},
            "properties": {"pointid": pid, "xcoord": x, "ycoord": y, "kv_0_30": kv,
                           "kv_30_100": kv2, "landnutzung": lu, "bze_moor": 0,
                           "probenahme_jahr": 2015}}


def _page(features, matched=5):
    return {"type": "FeatureCollection", "features": features, "numberMatched": matched,
            "numberReturned": len(features),
            "crs": {"type": "name", "properties": {"name": C.BZE_CRS}}}


def _pages(first=None, second=None, matched=5):
    return [_page(first or [_feature(i) for i in (1, 2, 3)], matched),
            _page(second or [_feature(i) for i in (4, 5)], matched)]


@pytest.fixture
def data(monkeypatch, tmp_path):
    """Point the checkpoint at an empty raw/, and return a function that lands
    a pair of datasets in it: clean unless told otherwise."""
    root = tmp_path / "raw"
    monkeypatch.setattr(C.F, "RAW", root)
    monkeypatch.setattr(C.W, "RAW", root)
    monkeypatch.setattr(C, "BZE_SITES", 5)

    def land(loc=None, meas=None, pages=None):
        files = {"locations.csv": _locations() if loc is None else loc,
                 "measurements.csv": _measurements() if meas is None else meas}
        doc = {"id": C.F.ARTICLE_ID, "version": C.F.VERSION, "license": {"name": "CC BY 4.0"},
               "files": [{"name": n, "size": len(b), "computed_md5": hashlib.md5(b).hexdigest(),
                          "download_url": f"https://ndownloader.figshare.com/files/{i}"}
                         for i, (n, b) in enumerate(files.items(), 1)]}
        C.raw_store.store(C.F.article_item(), json.dumps(doc).encode(), True, WHEN)
        for n, b in files.items():
            C.raw_store.store(C.F.file_item(n), b, True, WHEN)
        start = 0
        for p in (_pages() if pages is None else pages):
            C.raw_store.store(C.W.page_item(start), json.dumps(p).encode(), True, WHEN)
            start += len(p["features"])
    return land


def _rules():
    rules, diagnostics = C.run()
    return {r.rule.split()[0]: r for r in rules}, diagnostics


# --- the whole checkpoint -------------------------------------------------------

def test_a_clean_pair_of_datasets_passes_every_rule(data):
    data()
    rules, diag = _rules()
    assert [k for k, r in rules.items() if not r.passed] == []
    assert list(rules) == ["P1", "P2", "P3", "P4", "P5", "P6", "B1", "B2", "B3"]
    assert list(diag["potash"]["locations"]) == [3] * 8
    assert diag["bze"]["sites"] == 5


def test_the_report_states_when_it_ran_and_the_verdict(data):
    data()
    rules, diag = C.run()
    lines = C.report(rules, diag, when=WHEN).splitlines()
    assert lines[0] == "Checkpoint 1, run 2026-11-05T12:00:00Z"
    assert lines[-1] == "Checkpoint 1 PASSED: 9/9 rules"


def test_nothing_pulled_exits_2_and_says_what_to_run(data, capsys):
    assert C.main([]) == 2
    assert "python -m src.ingest.figshare_client" in capsys.readouterr().out


# --- Potash et al. (2023) -----------------------------------------------------

def test_a_file_that_differs_from_the_article_fails_p1(data):
    """Prevents: analysing bytes that figshare did not publish."""
    data()
    C.raw_store.store(C.F.file_item("locations.csv"), _locations() + b"S1,9,-88.2,40.1\n", True, LATER)
    rules, _ = _rules()
    assert not rules["P1"].passed and "locations.csv" in rules["P1"].detail


def test_a_missing_column_fails_p2_and_stops_the_potash_rules(data):
    data(loc=b"site,location_id,X\nS1,1,-88.2\n")
    rules, diag = _rules()
    assert not rules["P2"].passed and "'Y'" in rules["P2"].detail
    assert "P3" not in rules and diag["potash"] is None


def test_a_site_without_measurements_fails_p3(data):
    data(meas=_measurements([r for r in _meas_rows() if r[0] != "S8"]))
    rules, _ = _rules()
    assert not rules["P3"].passed and "S8" in rules["P3"].detail


def test_a_measurement_without_its_location_fails_p4(data):
    """Prevents: a sample with no position silently leaving a variogram."""
    data(meas=_measurements(_meas_rows() + [("S1", "99", 0, 15, 1.5, 1.3)]))
    rules, _ = _rules()
    assert not rules["P4"].passed
    assert "1 measurement rows without a location" in rules["P4"].detail


def test_the_same_location_id_at_two_sites_is_not_a_duplicate(data):
    """location_id is unique only within a site (figshare description); every
    fixture site uses ids 1-3."""
    data()
    rules, _ = _rules()
    assert rules["P4"].passed


def test_a_location_outside_the_envelope_fails_p5(data):
    rows = _loc_rows()
    rows[0] = (rows[0][0], rows[0][1], 88.2, 40.1)      # east of Greenwich, not west
    data(loc=_locations(rows))
    rules, _ = _rules()
    assert not rules["P5"].passed and "outside the envelope" in rules["P5"].detail


def test_an_inverted_segment_or_impossible_bulk_density_fails_p6(data):
    rows = _meas_rows()
    rows[0] = ("S1", "1", 15, 0, 1.5, 1.3)
    rows[1] = ("S1", "1", 15, 30, 1.5, 3.1)
    data(meas=_measurements(rows))
    rules, _ = _rules()
    assert not rules["P6"].passed
    assert "inverted" in rules["P6"].detail and "BD outside" in rules["P6"].detail


def test_an_empty_bulk_density_is_counted_not_failed(data):
    rows = _meas_rows()
    rows[0] = ("S1", "1", 0, 15, 1.5, "")
    data(meas=_measurements(rows))
    rules, _ = _rules()
    assert rules["P6"].passed and "1 BD empty" in rules["P6"].detail


def test_a_gap_above_30_cm_or_no_segment_is_not_a_complete_0_30_cm(data):
    """Prevents: a location whose deepest segment reaches 30 cm counting as
    covered when a segment above it is missing (met on 05.11.2026: IL-BR)."""
    rows = [r for r in _meas_rows() if not (r[0] == "S1" and r[1] == "1" and r[2] == 0)]
    rows = [r for r in rows if not (r[0] == "S2" and r[1] == "1")]
    data(meas=_measurements(rows))
    rules, diag = _rules()
    assert rules["P4"].passed
    per_site = diag["potash"].set_index("site")["complete 0-30 cm"]
    assert per_site["S1"] == "2/3" and per_site["S2"] == "2/3" and per_site["S3"] == "3/3"


def test_coverage_needs_contiguous_segments_from_the_surface():
    import pandas as pd
    seg = lambda *pairs: pd.DataFrame(pairs, columns=["sample_depth_min", "sample_depth_max"])
    assert C.covers(seg((0, 15), (15, 30)))
    assert C.covers(seg((15, 30), (0, 15), (30, 60)))        # order does not matter
    assert not C.covers(seg((0, 10), (20, 30)))               # a gap
    assert not C.covers(seg((5, 15), (15, 30)))               # not from the surface
    assert not C.covers(seg((0, 15), (10, 30)))               # an overlap
    assert not C.covers(seg((0, 27)))                         # too shallow
    assert C.scheme(seg((15, 30), (0, 15), (30, 60))) == "0-15 15-30"


# --- BZE-LW ---------------------------------------------------------------------

def test_a_site_repeated_across_pages_fails_b1(data):
    """Prevents: unordered paging counting one site twice and missing another."""
    data(pages=_pages(second=[_feature(3), _feature(5)]))
    rules, _ = _rules()
    assert not rules["B1"].passed and "repeated pointid" in rules["B1"].detail


def test_pages_short_of_the_stated_total_fail_b1(data):
    data(pages=_pages(matched=6))
    rules, _ = _rules()
    assert not rules["B1"].passed and "the server states 6" in rules["B1"].detail


def test_a_geometry_away_from_its_coordinates_fails_b2(data):
    data(pages=_pages(first=[_feature(1, gx=401500.0), _feature(2), _feature(3)]))
    rules, _ = _rules()
    assert not rules["B2"].passed and "differ from xcoord" in rules["B2"].detail


def test_a_point_outside_the_published_extent_fails_b2(data):
    data(pages=_pages(first=[_feature(1, x=100.0, y=100.0), _feature(2), _feature(3)]))
    rules, _ = _rules()
    assert not rules["B2"].passed and "outside the published extent" in rules["B2"].detail


def test_a_negative_stock_or_unknown_land_use_fails_b3(data):
    data(pages=_pages(first=[_feature(1, kv=-3.0), _feature(2, lu="X"), _feature(3)]))
    rules, _ = _rules()
    assert not rules["B3"].passed
    assert "outside [0, 1000]" in rules["B3"].detail and "'X'" in rules["B3"].detail


def test_text_in_kv_30_100_is_reported_not_judged(data):
    data(pages=_pages(first=[_feature(1, kv2="n.b."), _feature(2, kv2=None), _feature(3)]))
    rules, diag = _rules()
    assert all(r.passed for r in rules.values())
    assert diag["bze"]["kv_30_100 text values"] == {"n.b.": 1}
    assert diag["bze"]["kv_30_100 numeric / text / empty"] == "3 / 1 / 1"


# --- config ----------------------------------------------------------------------

def test_the_variogram_readings_follow_webster_and_oliver():
    assert C.reading(49).startswith("fewer than 50")
    assert C.reading(50).startswith("50-99")
    assert C.reading(149).startswith("100-149")
    assert C.reading(150).startswith("150-224")
    assert C.reading(225).startswith("225 or more")
    assert C.reading(10_000).startswith("225 or more")


def test_every_config_row_carries_its_source():
    """Provenance is a column, not a comment (portfolio standard, honesty rule 3)."""
    for name, column in (("sources.csv", "attribution"), ("extents.csv", "basis"),
                         ("variogram_sample_sizes.csv", "citation")):
        rows = C._config(name, column)
        assert rows and all(r[column].strip() for r in rows)
