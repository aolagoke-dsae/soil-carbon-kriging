"""The WFS client: pages in pointid order, paging that stops at the stated
total, error pages kept aside, no page fetched twice. No network is used: a
fake server slices a fixed layer by startIndex and count."""
import csv
import json

import pytest

import thuenen_wfs_client as W

CRS = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::25832"}}
EXCEPTION = (b'<?xml version="1.0" ?><ows:ExceptionReport version="2.0.0">'
             b"<ows:Exception><ows:ExceptionText>Illegal property name</ows:ExceptionText>"
             b"</ows:Exception></ows:ExceptionReport>")


def _features(n):
    return [{"type": "Feature", "id": f"layer.{i}",
             "geometry": {"type": "Point", "coordinates": [400000.0 + i, 5500000.0 + i]},
             "properties": {"pointid": i}} for i in range(1, n + 1)]


class Resp:
    def __init__(self, status, content, url):
        self.status_code, self.content, self.url, self.headers = status, content, url, {}


class Layer:
    """GetFeature over n features; `answer` can replace the response for a
    given startIndex."""
    def __init__(self, n, answer=None):
        self.features, self.answer, self.starts = _features(n), answer or {}, []

    def get(self, url, params=None, timeout=None):
        start, count = params["startIndex"], params["count"]
        self.starts.append(start)
        if start in self.answer:
            status, body = self.answer[start]
        else:
            page = self.features[start:start + count]
            status, body = 200, json.dumps({
                "type": "FeatureCollection", "features": page, "crs": CRS,
                "numberMatched": len(self.features), "numberReturned": len(page)}).encode()
        return Resp(status, body, f"{url}?startIndex={start}")


@pytest.fixture
def raw(monkeypatch, tmp_path):
    root = tmp_path / "raw"
    monkeypatch.setattr(W, "RAW", root)
    monkeypatch.setattr(W, "MANIFEST", root / "manifest.csv")
    monkeypatch.setattr(W, "PAGE", 2)
    return root


def _serve(monkeypatch, layer):
    monkeypatch.setattr(W.http_get, "session", lambda: layer)
    return layer


def test_pages_are_asked_for_as_geojson_in_pointid_order():
    """Prevents: pages from an unordered result skipping or repeating sites."""
    p = W.params(3000)
    assert (p["sortBy"], p["outputFormat"], p["startIndex"], p["count"]) == (
        "pointid", "application/json", 3000, W.PAGE)
    assert p["typeNames"] == "geonode:bze_lw_standorte_verschleiert"


def test_only_a_featurecollection_stating_its_total_is_data():
    ok = json.dumps({"type": "FeatureCollection", "features": [], "numberMatched": 0}).encode()
    assert W.usable(200, ok)
    assert not W.usable(500, ok)
    assert not W.usable(200, EXCEPTION)          # GeoServer may send it with HTTP 200
    assert not W.usable(200, json.dumps({"type": "FeatureCollection", "features": []}).encode())
    assert not W.usable(200, b"<html>Bad Gateway</html>")


def test_paging_stops_when_the_pages_reach_the_stated_total(raw, monkeypatch):
    layer = _serve(monkeypatch, Layer(5))
    rows = W.pull()
    assert layer.starts == [0, 2, 4]
    assert [r["rejected"] for r in rows] == [0, 0, 0]
    stored = sorted(p.name for p in W.root().iterdir())
    assert stored == ["start_00000", "start_00002", "start_00004"]


def test_every_page_is_in_the_manifest_with_its_url(raw, monkeypatch):
    _serve(monkeypatch, Layer(3))
    W.pull()
    with (raw / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["url"].rsplit("=", 1)[1] for r in rows] == ["0", "2"]
    assert all(len(r["sha256"]) == 64 and r["http_status"] == "200" for r in rows)


def test_a_rerun_fetches_no_page_already_stored(raw, monkeypatch):
    _serve(monkeypatch, Layer(5))
    W.pull()
    again = _serve(monkeypatch, Layer(5))
    assert W.pull() == [] and again.starts == []


def test_a_rejected_page_is_kept_aside_and_the_pull_fails(raw, monkeypatch):
    """Prevents: an exception report standing in for a page of sites."""
    _serve(monkeypatch, Layer(5, answer={2: (200, EXCEPTION)}))
    with pytest.raises(RuntimeError, match="startIndex 2 rejected"):
        W.pull()
    assert W.raw_store.current_version(W.page_item(2)) is None
    assert len(list(W.root().rglob("_failed/*.json"))) == 1


def test_an_empty_page_before_the_total_fails_instead_of_looping(raw, monkeypatch):
    empty = json.dumps({"type": "FeatureCollection", "features": [],
                        "numberMatched": 5}).encode()
    _serve(monkeypatch, Layer(5, answer={2: (200, empty)}))
    with pytest.raises(RuntimeError, match="empty"):
        W.pull()
