"""The National Map client: the newest version is taken and then pinned, only
the header and the covering blocks are read, a server that ignores the range
is refused before its body is read, and nothing is fetched twice. No network
is used: the session is replaced by a server that answers from a small
synthetic tile."""
import csv
import hashlib
import json
import re

import pytest

import patches as P
import tnm_client as T

BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/historical/n40w089/"
OLD, NEW, NEWER = "20221205", "20240104", "20260902"


def _url(version):
    return f"{BASE}USGS_13_n40w089_{version}.tif"


def _item(version, size, published=None):
    return {"title": f"USGS 1/3 Arc Second n40w089 {version}", "format": "GeoTIFF",
            "publicationDate": published or f"{version[:4]}-{version[4:6]}-{version[6:]}",
            "sizeInBytes": size, "downloadURL": _url(version),
            "vendorMetaUrl": f"https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/elevation/"
                             f"1-3_arc-second/IL_X/USGS_13_n40w089_{version}.xml",
            "boundingBox": {"minX": -89.0006, "maxX": -87.9994, "minY": 39.9994, "maxY": 41.0006}}


def _listing(items):
    odd = {"title": "USGS 13 arc-second n40w089 1 x 1 degree", "format": "GeoTIFF",
           "publicationDate": "2030-01-01", "sizeInBytes": 431690,
           "downloadURL": BASE + "USGS_13_n40w089_20180305.tif"}
    items = items + [odd]
    return json.dumps({"total": len(items), "items": items}).encode()


class Unread:
    """A body that must not be read."""
    def __get__(self, obj, owner):
        raise AssertionError("the body of a refused range response was read")


class Resp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self._content, self.headers = status, content, headers or {}
        self.closed = False

    @property
    def content(self):
        return self._content

    def close(self):
        self.closed = True


class RefusedResp(Resp):
    content = Unread()


class Server:
    """Listings by prefix; tiles answer byte ranges as S3 does."""
    def __init__(self, tile, listing_versions=(OLD, NEW), ignore_range=False, etags=None):
        self.tile, self.ignore_range = tile, ignore_range
        self.listing_versions = list(listing_versions)
        self.etags = etags or {}
        self.calls = []

    def get(self, url, params=None, headers=None, stream=False, timeout=None):
        self.calls.append((url, (headers or {}).get("Range")))
        if url.startswith(T.API):
            return Resp(200, _listing([_item(v, len(self.tile)) for v in self.listing_versions]))
        if url.endswith(".xml"):
            return Resp(200, b"<metadata><title>IL_X lidar</title></metadata>")
        m = re.match(r"bytes=(\d+)-(\d+)", (headers or {}).get("Range", ""))
        if self.ignore_range or not m:
            return RefusedResp(200, headers={"Content-Length": str(len(self.tile))})
        a, b = int(m.group(1)), min(int(m.group(2)), len(self.tile) - 1)
        version = url.rsplit("_", 1)[1][:8]
        etag = self.etags.get(version, f'"etag-{version}"')
        if callable(etag):
            etag = etag()
        return Resp(206, self.tile[a:b + 1],
                    {"Content-Range": f"bytes {a}-{b}/{len(self.tile)}", "ETag": etag})


@pytest.fixture
def setup(monkeypatch, tmp_path, synthetic_tile):
    data, _, _, loc = synthetic_tile
    root = tmp_path / "raw"
    stocks = tmp_path / "stocks.parquet"
    loc.to_parquet(stocks)
    monkeypatch.setattr(T, "RAW", root)
    monkeypatch.setattr(T, "MANIFEST", root / "manifest.csv")
    monkeypatch.setattr(T, "STOCKS", stocks)
    return data, root


def _serve(monkeypatch, server):
    monkeypatch.setattr(T.http_get, "session", lambda: server)
    return server


def _manifest(root):
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_only_dated_one_third_arc_second_geotiffs_are_versions():
    """Prevents: the older '13 arc-second ... 1 x 1 degree' product, dated
    later here, being taken for the newest version."""
    by = T.versions(json.loads(_listing([_item(OLD, 10), _item(NEW, 10)]))["items"])
    assert list(by) == ["n40w089"] and len(by["n40w089"]) == 2
    assert T.newest(by["n40w089"])["downloadURL"] == _url(NEW)


def test_the_latest_publication_date_decides_not_the_date_in_the_name():
    a, b = _item("20220105", 10, "2022-01-12"), _item("20220110", 10, "2022-01-11")
    assert T.newest([a, b])["downloadURL"] == _url("20220105")


def test_a_pull_reads_the_header_and_the_covering_blocks_only(setup, monkeypatch):
    data, root = setup
    server = _serve(monkeypatch, Server(data))
    rows = T.pull()
    tile_rows = [r for r in rows if r["url"] == _url(NEW)]
    assert tile_rows and all(r["http_status"] == 206 for r in tile_rows)
    assert all(r["url"] != _url(OLD) for r in rows)              # the newest version only
    read = sum(r["bytes"] for r in tile_rows)
    assert read < 0.6 * len(data)
    rev = T.parse_revision(tile_rows[0]["revision"])
    assert rev["first"] == 0 and rev["etag"] == f'"etag-{NEW}"' and rev["size"] == len(data)
    kinds = {r["path"].split("/TNM/")[1].split("/")[0] for r in rows}
    assert kinds == {"listing", "tiles", "metadata"}
    for r in _manifest(root):                                    # stored bytes match their rows
        body = open(r["path"], "rb").read()
        assert hashlib.sha256(body).hexdigest() == r["sha256"] and len(body) == int(r["bytes"])
    assert all(rng is None or rng.startswith("bytes=") for _, rng in server.calls)


def test_a_second_pull_fetches_nothing(setup, monkeypatch):
    data, _ = setup
    _serve(monkeypatch, Server(data))
    T.pull()
    server = _serve(monkeypatch, Server(data))
    assert T.pull() == [] and server.calls == []


def test_a_pinned_version_is_read_again_when_a_newer_one_appears(setup, monkeypatch):
    data, root = setup
    _serve(monkeypatch, Server(data))
    T.pull()
    server = _serve(monkeypatch, Server(data, listing_versions=(OLD, NEW, NEWER)))
    T.pull(refresh=True)
    assert not any(NEWER in url for url, _ in server.calls if url.endswith(".tif"))
    assert T.pinned(root / "manifest.csv") == {"n40w089": _url(NEW)}


def test_a_server_that_ignores_the_range_is_refused_before_its_body_is_read(setup, monkeypatch):
    """Prevents: a 0.4 GB tile being downloaded whole."""
    data, root = setup
    _serve(monkeypatch, Server(data, ignore_range=True))
    with pytest.raises(RuntimeError, match="HTTP 200"):
        T.pull()
    refused = [r for r in _manifest(root) if r["rejected"] == "1"]
    assert len(refused) == 1 and "/_failed/" in refused[0]["path"]
    assert b"not read" in open(refused[0]["path"], "rb").read()


def test_a_file_that_changes_between_requests_is_refused(setup, monkeypatch):
    data, _ = setup
    tags = iter([f'"e{i}"' for i in range(1000)])
    _serve(monkeypatch, Server(data, etags={NEW: lambda: next(tags)}))
    with pytest.raises(RuntimeError, match="changed between requests"):
        T.pull()


def test_the_cap_stops_the_pull_before_any_block(setup, monkeypatch):
    data, _ = setup
    server = _serve(monkeypatch, Server(data))
    with pytest.raises(RuntimeError, match="more than"):
        T.pull(max_mb=1e-6)
    ranges = [rng for url, rng in server.calls if url.endswith(".tif")]
    assert ranges and all(rng.startswith("bytes=0-") for rng in ranges)   # the header only


def test_the_metadata_record_is_the_xml_or_else_the_sciencebase_json():
    """Prevents: a page of the bucket's browser being stored as the record."""
    xml = {"vendorMetaUrl": "https://thor-f5.er.usgs.gov/a/USGS_13_n40w089_20240104.xml",
           "metaUrl": "https://www.sciencebase.gov/catalog/item/abc"}
    page = {"vendorMetaUrl": "https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/x",
            "metaUrl": "https://www.sciencebase.gov/catalog/item/abc"}
    assert T.metadata_url(xml) == xml["vendorMetaUrl"]
    assert T.metadata_url(page) == "https://www.sciencebase.gov/catalog/item/abc?format=json"
    assert T.metadata_item("n40w089", "20240104", T.metadata_url(page)).suffix == ".json"


def test_tile_urls_carry_their_block_and_date():
    assert T.tile_version(_url(NEW)) == ("n40w089", NEW)
    with pytest.raises(ValueError):
        T.tile_version(BASE.replace("historical", "current") + "USGS_13_n40w089.tif")
