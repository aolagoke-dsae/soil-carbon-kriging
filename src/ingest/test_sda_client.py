"""The SDA client: what counts as SDA's table, the queries it sends, and what
reaches raw/ and the manifest. No network: the session is replaced."""
import csv
import json

import pandas as pd
import pytest

import sda_client as S


def _table(columns, rows):
    return json.dumps({"Table": [columns] + rows}).encode()


POLY = _table(S.POLYGON_COLUMNS, [["1", "111", "IL001", "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"]])
HZ = _table(S.HORIZON_COLUMNS, [["111", "9", "Alpha", "90", "Yes", "0", "20", "1", "2", "3", "1.3"]])
CAT = _table(S.CATALOG_COLUMNS, [["IL001", "9/3/2025 12:00:00 AM"]])


def test_only_sdas_table_with_the_columns_asked_for_is_accepted():
    """Prevents: an empty answer ({}), an error page or another query's table
    stored as the survey."""
    assert S.table(POLY, S.POLYGON_COLUMNS) is not None
    assert S.table(b"{}", S.POLYGON_COLUMNS) is None
    assert S.table(b"<html>error</html>", S.POLYGON_COLUMNS) is None
    assert S.table(_table(S.POLYGON_COLUMNS, []), S.POLYGON_COLUMNS) is None
    assert S.table(POLY, S.CATALOG_COLUMNS) is None


def test_keys_are_refused_rather_than_quoted():
    """Prevents: a key that is not a key reaching the SQL."""
    assert S._keys(["12", "3", "12"]) == "'12', '3'"
    with pytest.raises(ValueError):
        S._keys(["1'; DROP TABLE x; --"])


def test_the_box_is_widened_and_closed():
    b = S.box(pd.Series([-90.2, -90.1]), pd.Series([39.0, 39.1]))
    assert b == pytest.approx((-90.201, 38.999, -90.099, 39.101))
    w = S.box_wkt(b)
    assert w.startswith("POLYGON((") and w.split("(")[-1].split(",")[0] == w.split(", ")[-1].rstrip("))")


class Resp:
    def __init__(self, status, content):
        self.status_code, self.content, self.headers = status, content, {}


class Server:
    """Answers each POST by what its query selects from."""
    def __init__(self, polygons=POLY):
        self.polygons, self.queries = polygons, []

    def post(self, url, json=None, timeout=None):
        q = json["query"]
        self.queries.append(q)
        assert url == S.URL and json["format"] == "JSON+COLUMNNAME"
        if "FROM mupolygon" in q:
            return Resp(200, self.polygons)
        if "FROM component" in q:
            return Resp(200, HZ)
        return Resp(200, CAT)


@pytest.fixture
def raw(monkeypatch, tmp_path):
    root = tmp_path / "raw"
    monkeypatch.setattr(S, "RAW", root)
    monkeypatch.setattr(S, "MANIFEST", root / "manifest.csv")
    stocks = tmp_path / "stocks.parquet"
    pd.DataFrame({"site": ["A", "A", "B"], "lon": [0.2, 0.4, 0.5], "lat": [0.2, 0.3, 0.5]}).to_parquet(stocks)
    return root, stocks


def _serve(monkeypatch, server):
    monkeypatch.setattr(S.http_get, "session", lambda: server)
    return server


def test_pull_stores_each_item_once_and_records_it(monkeypatch, raw):
    root, stocks = raw
    server = _serve(monkeypatch, Server())
    rows = S.pull(stocks=stocks)
    assert len(rows) == 4                       # two fields, the horizons, the catalog
    assert "'111'" in server.queries[2] and "'IL001'" in server.queries[3]
    with (root / "manifest.csv").open(newline="") as fh:
        manifest = list(csv.DictReader(fh))
    assert [r["rejected"] for r in manifest] == ["0"] * 4
    assert all(r["revision"].startswith("q") and len(r["revision"]) == 13 for r in manifest)
    assert S.stored(S.polygons_item("A"), S.POLYGON_COLUMNS) is not None
    assert S.pull(stocks=stocks) == []          # nothing fetched twice


def test_an_empty_answer_is_rejected_and_stops_the_pull(monkeypatch, raw):
    root, stocks = raw
    _serve(monkeypatch, Server(polygons=b"{}"))
    with pytest.raises(RuntimeError, match="rejected"):
        S.pull(stocks=stocks)
    assert S.raw_store.current_version(S.polygons_item("A")) is None
    assert list((root / "SDA" / "polygons" / "A" / "_failed").iterdir())
