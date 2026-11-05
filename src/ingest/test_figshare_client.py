"""The figshare client: the pinned version, the MD5 check, what reaches the
manifest, and no file fetched twice. No network is used: the session is
replaced by a table of canned responses."""
import csv
import datetime as dt
import hashlib
import json

import pytest

import figshare_client as F

LOC = b"site,location_id,X,Y\nA,1,-88.2,40.1\n"
MEAS = b"site,location_id,sample_depth_min,sample_depth_max,SOCc,BD\nA,1,0,15,1.9,1.31\n"
SIGNED = "https://storage.example/file?X-Amz-Credential=TEMP&X-Amz-Signature=SECRET"


def _meta(name, body, file_id):
    return {"id": file_id, "name": name, "size": len(body),
            "download_url": f"https://ndownloader.figshare.com/files/{file_id}",
            "computed_md5": hashlib.md5(body).hexdigest()}


def _article(**over):
    doc = {"id": F.ARTICLE_ID, "version": F.VERSION,
           "license": {"name": "CC BY 4.0"},
           "files": [_meta("locations.csv", LOC, 1), _meta("measurements.csv", MEAS, 2)]}
    doc.update(over)
    return doc


class Resp:
    def __init__(self, status, content, url):
        self.status_code, self.content, self.url, self.headers = status, content, url, {}


class Server:
    """url -> (status, body); file downloads answer from a signed address,
    as figshare's redirect does."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        status, body = self.routes[url]
        return Resp(status, body, SIGNED if "ndownloader" in url else url)


@pytest.fixture
def raw(monkeypatch, tmp_path):
    root = tmp_path / "raw"
    monkeypatch.setattr(F, "RAW", root)
    monkeypatch.setattr(F, "MANIFEST", root / "manifest.csv")
    return root


def _serve(monkeypatch, article=None, loc=LOC, meas=MEAS):
    doc = article or _article()
    server = Server({F.API: (200, json.dumps(doc).encode()),
                     doc["files"][0]["download_url"]: (200, loc),
                     doc["files"][1]["download_url"]: (200, meas)})
    monkeypatch.setattr(F.http_get, "session", lambda: server)
    return server


def _manifest(raw):
    with (raw / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_article_is_read_at_its_pinned_version_never_latest():
    assert F.API == "https://api.figshare.com/v2/articles/23669304/versions/2"


def test_only_the_pinned_record_with_sizes_and_md5s_is_usable():
    """Prevents: another version, an error page or a record without
    checksums being taken as the article."""
    ok = json.dumps(_article()).encode()
    assert F.usable_article(200, ok)
    assert not F.usable_article(404, ok)
    assert not F.usable_article(200, json.dumps(_article(version=3)).encode())
    assert not F.usable_article(200, json.dumps(_article(license={})).encode())
    no_md5 = _article()
    del no_md5["files"][0]["computed_md5"]
    assert not F.usable_article(200, json.dumps(no_md5).encode())
    assert not F.usable_article(200, b"<html>Service unavailable</html>")


def test_a_pull_stores_the_record_and_each_file_once(raw, monkeypatch):
    _serve(monkeypatch)
    rows = F.pull()
    assert [r["rejected"] for r in rows] == [0, 0, 0]
    assert F.stored_article()["version"] == F.VERSION
    assert F.raw_store.current_version(F.file_item("locations.csv")).read_bytes() == LOC
    assert F.raw_store.current_version(F.file_item("measurements.csv")).read_bytes() == MEAS


def test_the_manifest_keeps_the_listed_url_not_the_signed_redirect(raw, monkeypatch):
    """Prevents: a temporary storage credential being committed with the manifest."""
    _serve(monkeypatch)
    F.pull()
    text = (raw / "manifest.csv").read_text(encoding="utf-8")
    assert "SECRET" not in text and "X-Amz" not in text
    urls = [r["url"] for r in _manifest(raw)]
    assert urls[1:] == ["https://ndownloader.figshare.com/files/1",
                        "https://ndownloader.figshare.com/files/2"]
    assert all(r["revision"] == "v2" for r in _manifest(raw))


def test_a_file_whose_md5_differs_is_kept_aside_and_the_pull_fails(raw, monkeypatch):
    """Prevents: a truncated or altered download being analysed."""
    _serve(monkeypatch, loc=LOC.replace(b"40.1", b"41.1"))
    with pytest.raises(RuntimeError, match="MD5"):
        F.pull()
    assert F.raw_store.current_version(F.file_item("locations.csv")) is None
    failed = list((raw / "FIGSHARE").rglob("_failed/*.csv"))
    assert len(failed) == 1
    assert _manifest(raw)[-1]["rejected"] == "1"


def test_an_error_page_for_the_record_is_kept_but_never_used(raw, monkeypatch):
    server = Server({F.API: (200, b"<html>maintenance</html>")})
    monkeypatch.setattr(F.http_get, "session", lambda: server)
    with pytest.raises(RuntimeError, match="article record rejected"):
        F.pull()
    assert F.stored_article() is None
    assert server.calls == [F.API]


def test_a_rerun_fetches_nothing_already_stored(raw, monkeypatch):
    """Prevents: every rerun downloading the data again into raw/."""
    _serve(monkeypatch)
    F.pull()
    again = _serve(monkeypatch)
    assert F.pull() == [] and again.calls == []


def test_refresh_fetches_everything_again_beside_the_old(raw, monkeypatch):
    _serve(monkeypatch)
    F.pull()
    later = F._now() + dt.timedelta(hours=1)
    monkeypatch.setattr(F, "_now", lambda: later)
    _serve(monkeypatch)
    assert len(F.pull(refresh=True)) == 3
    assert len(list((raw / "FIGSHARE").rglob("files/locations/*.csv"))) == 2
