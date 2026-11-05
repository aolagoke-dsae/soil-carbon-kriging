"""Write-once raw storage, shared by every ingestion client.

Copied from grid-carbon-intensity (src/ingest/raw_store.py); only this
paragraph is new. The portfolio standard sets its rules for every project,
and the examples below are that project's.

raw/ is append-only. A logical item -- one ENTSO-E document type, zone and
year; one SMARD index or week -- is a DIRECTORY, and every retrieval of it is
a new file inside, named <UTC time>_<first 8 hex digits of its sha256>:

    raw/A44/10Y1001A1001A82H/2025/20261006T183150Z_9d48b590.xml
    raw/A44/10Y1001A1001A82H/2025/_failed/20261010T091500Z_3f2a1c0d.xml

Nothing is ever overwritten. A response that is not data goes to _failed/,
kept for diagnosis and never read as the current version. Downstream takes the
newest accepted file in the directory (current_version).

Why it exists: until 10.10.2026 each client wrote straight onto <year>.xml.
The first A44 pull of 06.10 was overwritten by the second, so the payload
behind the first manifest row no longer exists, and an expired token's 401
page would have replaced a year of data while smoke.py printed ACCEPTED.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path

MANIFEST_FIELDS = [
    "retrieved_at_utc", "path", "url", "http_status", "bytes",
    "sha256", "rejected", "attempt", "revision",
]
FAILED = "_failed"


def version_dir(logical: Path) -> Path:
    """raw/A44/<eic>/2025.xml -> raw/A44/<eic>/2025/"""
    return logical.with_suffix("")


def store(logical: Path, body: bytes, accepted: bool, when: dt.datetime) -> Path:
    """Write body once, under the logical item's directory. Never overwrites."""
    if when.utcoffset() != dt.timedelta(0):
        raise ValueError("store() needs a UTC timestamp")
    sha = hashlib.sha256(body).hexdigest()
    folder = version_dir(logical) if accepted else version_dir(logical) / FAILED
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{when:%Y%m%dT%H%M%SZ}_{sha[:8]}{logical.suffix}"
    try:
        with path.open("xb") as fh:         # exclusive: fails if the file exists
            fh.write(body)
    except FileExistsError:
        # Same second, same digest: these bytes are already stored. Different
        # bytes under that name would mean overwriting, which is the one thing
        # this module exists to refuse.
        if path.read_bytes() != body:
            raise
    return path


def current_version(logical: Path) -> Path | None:
    """The newest accepted retrieval of a logical item, or None."""
    folder = version_dir(logical)
    if not folder.is_dir():
        return None
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix == logical.suffix
    )
    return files[-1] if files else None


def check_manifest(manifest: Path) -> None:
    """Refuse a manifest whose columns differ, rather than misalign its rows."""
    if not manifest.exists() or manifest.stat().st_size == 0:
        return
    with manifest.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), None)
    if header != MANIFEST_FIELDS:
        raise ValueError(
            f"{manifest} has columns {header}, expected {MANIFEST_FIELDS}. "
            "Appending would misalign every row: migrate it first."
        )


def append_manifest(manifest: Path, row: dict) -> None:
    """Append one row to the shared provenance record."""
    check_manifest(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    new = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
