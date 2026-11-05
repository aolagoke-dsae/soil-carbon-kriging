# Architecture: what differs from the portfolio standard

Dr. Adewole Olagoke · 5 November 2026

This repository follows the portfolio standard in
`grid-carbon-intensity/docs/ARCHITECTURE.md`: the governing rule (free at the
point of use, no account that can lapse), the repository shape, write-once
`raw/` with a committed manifest, rules fixed before the data, one commit per
passing checkpoint, and the honesty rules. Only what differs is recorded here.

| Layer | Standard | Here | Why |
|---|---|---|---|
| Ingestion | `requests` clients | The same, against two open APIs without a token: figshare REST API v2 (version pinned, size and MD5 per file) and OGC WFS 2.0 (paged, sorted by `pointid`) | Both sources publish through an API, so no browser download is needed |
| Warehouse | DuckDB + dbt | None at Stage 1 | Two tables of a few thousand rows. Revisit at Stage 5, if covariates are joined at scale |
| Analysis | NumPy | GeoPandas and SciPy now; scikit-gstat or PyKrige from Stage 2 | Geostatistics. A library enters `requirements*.txt` when a stage uses it, not before |
| Publication | Quarto → GitHub Pages | Not yet | After Stage 3, when there is a result to publish |

Notebooks are percent-cell Python (`notebooks/*.py`), never `.ipynb`, and
exploration only: nothing in one is load-bearing. The rules and their tests
live in `src/`.

## Why the clients use requests, not pd.read_csv(url)

`pd.read_csv(url)` downloads and parses in one line, and is the right tool for
a first look. It is not ingestion:

- the bytes are never stored, so nothing can be hashed, compared with the size
  and MD5 figshare states, or recorded in the manifest
- an HTML error page is parsed, or fails obscurely, instead of being rejected
- there is no retry, backoff or timeout policy
- it reads CSV only, and the Thünen layer comes as paged GeoJSON

So the clients fetch with `requests` and land the bytes in `raw/`, and pandas
reads them from there (`src/checks/checkpoint1.py`).

## Why the figshare manifest records the listed URL

Each file's `download_url` redirects to a signed storage address whose query
string carries a temporary credential. The manifest records the URL the
article lists, never the address it redirected to, so that the committed
manifest cannot carry a credential (`src/ingest/test_figshare_client.py`).
