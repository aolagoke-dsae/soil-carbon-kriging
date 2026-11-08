"""Checkpoint 3 end to end on a synthetic tile: a pull from a server that
answers byte ranges as S3 does, the transform, then E1-E6. And each rule
fails when what it guards is broken."""
import json
import re

import numpy as np
import pytest

import checkpoint3 as C

T, TC = C.T, C.TC          # the module objects the checkpoint itself uses

URL = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/historical/"
       "n40w089/USGS_13_n40w089_{v}.tif")


class Resp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    def close(self):
        pass


class Server:
    def __init__(self, tile, bbox):
        self.tile, self.bbox = tile, bbox

    def get(self, url, params=None, headers=None, stream=False, timeout=None):
        if url.startswith(TC.API):
            items = [{"title": f"USGS 1/3 Arc Second n40w089 {v}", "format": "GeoTIFF",
                      "publicationDate": f"{v[:4]}-{v[4:6]}-{v[6:]}", "sizeInBytes": len(self.tile),
                      "downloadURL": URL.format(v=v), "boundingBox": self.bbox,
                      "vendorMetaUrl": f"https://thor-f5.er.usgs.gov/x/USGS_13_n40w089_{v}.xml"}
                     for v in ("20221205", "20240104")]
            return Resp(200, json.dumps({"total": 2, "items": items}).encode())
        if url.endswith(".xml"):
            return Resp(200, b"<metadata><idinfo><timeperd><timeinfo><rngdates>"   # as USGS's,
                             b"<begdate>20120217</begdate><enddate>20220628</enddate>"  # no lineage
                             b"</rngdates></timeinfo></timeperd></idinfo></metadata>")
        a, b = map(int, re.match(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        b = min(b, len(self.tile) - 1)
        return Resp(206, self.tile[a:b + 1], {"Content-Range": f"bytes {a}-{b}/{len(self.tile)}",
                                              "ETag": '"abc"'})


@pytest.fixture
def stage(monkeypatch, tmp_path, synthetic_tile):
    data, transform, _, loc = synthetic_tile
    bbox = {"minX": transform.c, "maxX": transform.c + 600 * transform.a,
            "minY": transform.f + 600 * transform.e, "maxY": transform.f}
    stocks = tmp_path / "curated" / "stocks.parquet"
    stocks.parent.mkdir()
    loc.to_parquet(stocks)
    raw = tmp_path / "raw"
    monkeypatch.setattr(TC, "RAW", raw)
    monkeypatch.setattr(TC, "MANIFEST", raw / "manifest.csv")
    monkeypatch.setattr(TC, "STOCKS", stocks)
    monkeypatch.setattr(TC.http_get, "session", lambda: Server(data, bbox))
    TC.pull()
    monkeypatch.setattr(T, "STOCKS", stocks)
    monkeypatch.setattr(T, "MANIFEST", raw / "manifest.csv")
    monkeypatch.setattr(T, "OUT", tmp_path / "curated" / "terrain.parquet")
    monkeypatch.setattr(T, "OUT_GRIDS", tmp_path / "curated" / "terrain")
    table, fields = T.run_stage(T.load_locations(), T.stored_tiles())
    T.write(table, fields)
    monkeypatch.setattr(C, "N_LOCATIONS", len(loc))
    return C.load()


def test_every_rule_passes_on_a_clean_stage(stage):
    rules = [C.e1_provenance(stage), C.e2_grid(stage), C.e3_registration(stage),
             C.e4_local(stage), C.e5_flow(stage), C.e6_table(stage)]
    for r in rules:
        assert r.passed, f"{r.rule}: {r.detail}"
    text = C.report(rules, {"x": C.tile_table(stage)})
    assert "Checkpoint 3 PASSED: 6/6 rules" in text
    tiles = C.tile_table(stage)
    assert "20240104" in tiles.to_string()                         # the newest version
    assert tiles["sources named"].tolist() == ["0"]
    assert tiles["content dated"].tolist() == ["2012-02-17 to 2022-06-28"]


def test_e1_fails_when_a_stored_range_is_altered(stage):
    r = stage["rows"]
    first = r[r["revision"].str.startswith("bytes=")].iloc[0]
    p = T._resolve(first["path"])
    p.write_bytes(b"\0" + p.read_bytes()[1:])
    assert not C.e1_provenance(stage).passed


def test_e1_fails_when_an_older_version_was_taken(stage):
    stage["tiles"]["n40w089"].url = URL.format(v="20221205")
    assert not C.e1_provenance(stage).passed


def test_e6_fails_on_a_missing_value_or_a_stock_column(stage):
    t = stage["table"].copy()
    t.loc[0, "twi"] = np.nan
    assert not C.e6_table({**stage, "table": t}).passed
    t = stage["table"].copy()
    t["stock"] = 1.0
    assert not C.e6_table({**stage, "table": t}).passed


def test_e4_fails_when_a_kernel_is_wrong(stage, monkeypatch):
    g = stage["fields"]["T1"]["grids"]
    g["slope_90"] = g["slope_90"] * (1 + 1e-6)
    assert not C.e4_local(stage).passed


def test_e5_fails_when_the_accumulation_is_wrong(stage):
    g = stage["fields"]["T1"]["grids"]
    g["area"] = g["area"] * 1.001
    assert not C.e5_flow(stage).passed


def test_e2_fails_when_a_window_leaves_no_native_cell_to_spare(stage):
    p = stage["fields"]["T1"]["parts"][0]
    r0, r1, c0, c1 = p.window
    assert r0 > 0
    p.window, p.values = (r0 + 2, r1, c0, c1), p.values[2:]
    assert not C.e2_grid(stage).passed


def test_e3_fails_when_the_grid_is_shifted_half_a_cell(stage):
    cog = C.cog
    for p in stage["fields"]["T1"]["parts"]:
        g = p.geo
        p.geo = cog.Geo(g.crs, g.west + 0.5 * g.dx, g.north, g.dx, g.dy, g.width, g.height,
                        g.tilewidth, g.tilelength, g.nodata)
    assert not C.e3_registration(stage).passed
