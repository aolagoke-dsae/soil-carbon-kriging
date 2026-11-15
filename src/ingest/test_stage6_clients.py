"""Stage 6's three clients on canned responses: what each accepts, what it
chooses, and what reaches raw/ and the manifest. No network is used."""
import csv
import gzip
import hashlib
import io
import zipfile

import pytest

import bkg_client as B
import dwd_client as D
import organic_client as O


class Resp:
    def __init__(self, status, content, url="u"):
        self.status_code, self.content, self.url, self.headers = status, content, url, {}


class Server:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None):
        key = url if params is None else (url, params.get("request"), params.get("typeNames"))
        self.calls.append(key)
        status, body = self.routes[key]
        return Resp(status, body, url)


def _raw(monkeypatch, tmp_path, *mods):
    root = tmp_path / "raw"
    for m in mods:
        monkeypatch.setattr(m, "RAW", root)
        monkeypatch.setattr(m, "MANIFEST", root / "manifest.csv")
    return root


def _manifest(root):
    with (root / "manifest.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


# --- DWD -----------------------------------------------------------------------------------------

def _listing(*names):
    return ("<html><body>" + "".join(f'<a href="{n}">{n[:30]}..&gt;</a>' for n in names) + "</body></html>").encode()


GRID = gzip.compress(b"NCOLS 2\nNROWS 2\nXLLCORNER 0\nYLLCORNER 0\nCELLSIZE 1000\nNODATA_VALUE -999\n1 2\n3 4\n")
T = ["grids_germany_multi_annual_air_temp_mean_1961-1990_17.asc.gz",
     "grids_germany_multi_annual_air_temp_mean_1991_2020_17.asc.gz",
     "grids_germany_multi_annual_air_temp_mean_1991_2020_07.asc.gz"]
P = ["grids_germany_multi_annual_precipitation_1961-1990_17.asc.gz",
     "grids_germany_multi_annual_precipitation_1981-2010_17.asc.gz"]


def test_dwd_takes_the_latest_period_both_variables_publish():
    """Prevents: a temperature of 1991-2020 beside a precipitation of 1961-1990."""
    listings = {"t": _listing(*T).decode(), "p": _listing(*P).decode()}
    assert D.latest_common(listings) == (1961, 1990)
    P2 = P + ["grids_germany_multi_annual_precipitation_1991_2020_17.asc.gz"]
    assert D.latest_common({"t": listings["t"], "p": _listing(*P2).decode()}) == (1991, 2020)
    assert D.annual_files(_listing(*T).decode())[(1991, 2020)] == T[1]       # not the July file


def test_dwd_reads_the_short_period_codes_too():
    """DWD writes some periods as 6190 or 9120."""
    files = D.annual_files(_listing("grids_germany_multi_annual_precipitation_6190_17.asc.gz",
                                    "grids_germany_multi_annual_precipitation_9120_17.asc.gz",
                                    "grids_germany_multi_annual_precipitation_9120_07.asc.gz").decode())
    assert set(files) == {(1961, 1990), (1991, 2020)} and files[(1991, 2020)].endswith("_9120_17.asc.gz")


def test_dwd_accepts_only_a_gzipped_ascii_grid():
    assert D.usable_grid(200, GRID)
    assert not D.usable_grid(200, b"<html>not found</html>")
    assert not D.usable_grid(404, GRID)


def test_dwd_pull_stores_listings_and_the_two_annual_grids(monkeypatch, tmp_path):
    root = _raw(monkeypatch, tmp_path, D)
    P2 = P + ["grids_germany_multi_annual_precipitation_1991_2020_17.asc.gz"]
    routes = {D.BASE + "air_temperature_mean/": (200, _listing(*T)), D.BASE + "precipitation/": (200, _listing(*P2)),
              D.BASE + "air_temperature_mean/" + T[1]: (200, GRID), D.BASE + "precipitation/" + P2[-1]: (200, GRID)}
    server = Server(routes)
    monkeypatch.setattr(D.http_get, "session", lambda: server)
    rows = D.pull()
    assert len(rows) == 4 and [r["rejected"] for r in _manifest(root)] == ["0"] * 4
    assert rows[-1]["revision"] == "1991-2020"
    assert D.pull() == []


# --- BKG -----------------------------------------------------------------------------------------

def _zip(name="dgm200.tif"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, b"II*\x00fake")
    return buf.getvalue()


def test_bkg_needs_the_published_md5_and_a_geotiff():
    body = _zip()
    md5 = hashlib.md5(body).hexdigest()
    assert B.checksum(f"{md5}  dgm200.utm32s.tiff.zip\n".encode()) == md5
    assert B.usable_zip(200, body, md5)
    assert not B.usable_zip(200, body, "0" * 32)
    other = _zip("readme.txt")
    assert not B.usable_zip(200, other, hashlib.md5(other).hexdigest())


def test_bkg_pull_rejects_a_corrupted_download(monkeypatch, tmp_path):
    root = _raw(monkeypatch, tmp_path, B)
    body = _zip()
    server = Server({B.URL + ".md5": (200, (hashlib.md5(body).hexdigest() + "\n").encode()), B.URL: (200, body[:-3])})
    monkeypatch.setattr(B.http_get, "session", lambda: server)
    with pytest.raises(RuntimeError, match="rejected"):
        B.pull()
    assert B.raw_store.current_version(B.zip_item()) is None
    assert [r["rejected"] for r in _manifest(root)] == ["0", "1"]


# --- Thünen organic soils ------------------------------------------------------------------------

CAPS = b"""<?xml version="1.0"?><wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
<wfs:FeatureTypeList>
<wfs:FeatureType><wfs:Name>geonode:bze_lw_standorte_verschleiert</wfs:Name><wfs:Title>BZE-LW</wfs:Title></wfs:FeatureType>
<wfs:FeatureType><wfs:Name>geonode:kulisse_org_boeden</wfs:Name><wfs:Title>Aktualisierte Kulisse organischer Boeden</wfs:Title></wfs:FeatureType>
</wfs:FeatureTypeList></wfs:WFS_Capabilities>"""


def test_organic_candidates_from_the_capabilities():
    types = O.feature_types(CAPS)
    assert len(types) == 2
    assert O.candidates(types) == [("geonode:kulisse_org_boeden", "Aktualisierte Kulisse organischer Boeden")]
    assert O.number_matched(b'<wfs:FeatureCollection numberMatched="12345" numberReturned="0"/>') == 12345
