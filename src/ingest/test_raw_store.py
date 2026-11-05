"""raw/ is append-only: the rules of src/ingest/raw_store.py, copied from
grid-carbon-intensity, tested here on their own. No network is used."""
import datetime as dt

import pytest

import raw_store as R

UTC = dt.timezone.utc
T1 = dt.datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)
T2 = dt.datetime(2026, 11, 5, 12, 0, 1, tzinfo=UTC)


def test_a_new_retrieval_is_added_and_the_old_one_kept(tmp_path):
    item = tmp_path / "raw" / "X" / "a.json"
    first = R.store(item, b'{"v": 1}', True, T1)
    second = R.store(item, b'{"v": 2}', True, T2)
    assert first.exists() and second.exists() and first != second
    assert R.current_version(item) == second


def test_a_rejected_response_is_kept_but_never_becomes_current(tmp_path):
    """Prevents: an error page being read as the data."""
    item = tmp_path / "raw" / "X" / "a.json"
    good = R.store(item, b'{"v": 1}', True, T1)
    bad = R.store(item, b"<html>503</html>", False, T2)
    assert bad.parent.name == R.FAILED
    assert R.current_version(item) == good


def test_store_refuses_to_overwrite_different_bytes(tmp_path):
    item = tmp_path / "raw" / "X" / "a.json"
    path = R.store(item, b"one", True, T1)
    path.write_bytes(b"tampered")
    with pytest.raises(FileExistsError):
        R.store(item, b"one", True, T1)


def test_store_needs_a_utc_time(tmp_path):
    cet = dt.timezone(dt.timedelta(hours=1))
    with pytest.raises(ValueError):
        R.store(tmp_path / "a.json", b"x", True, T1.astimezone(cet))


def test_a_manifest_with_other_columns_is_refused(tmp_path):
    """Prevents: appending rows under the wrong headers."""
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("retrieved_at_utc,path\n", encoding="utf-8")
    with pytest.raises(ValueError, match="migrate"):
        R.append_manifest(manifest, {"path": "x"})
