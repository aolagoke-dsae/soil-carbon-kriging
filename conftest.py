"""Make the package layout importable from a plain `pytest` at the repo root."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
for _p in ("src/ingest", "src/checks", "src/transform"):
    sys.path.insert(0, str(ROOT / _p))
