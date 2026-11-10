# Architecture: what differs from the portfolio standard

Dr. Adewole Olagoke · 5 November 2026

This repository follows the portfolio standard in
`grid-carbon-intensity/docs/ARCHITECTURE.md`: the governing rule (free at the
point of use, no account that can lapse), the repository shape, write-once
`raw/` with a committed manifest, rules fixed before the data, one commit per
passing checkpoint, and the honesty rules. Only what differs is recorded here.

| Layer | Standard | Here | Why |
|---|---|---|---|
| Ingestion | `requests` clients | The same, against open APIs without a token: figshare REST API v2 (version pinned, size and MD5 per file) and OGC WFS 2.0 (paged, sorted by `pointid`); from Stage 3, the TNM Access API and HTTP range requests on The National Map's DEM tiles, reading only the blocks that cover the fields | Every source publishes through an API or plain HTTPS, so no browser download is needed |
| Warehouse | DuckDB + dbt | None. From Stage 2 the transforms write Parquet to `curated/`, which is rebuilt from `raw/` and not committed | Two tables of a few thousand rows. Revisit at Stage 6, if national covariates are joined at scale |
| Analysis | NumPy | GeoPandas and SciPy now. From Stage 2: REML and the gstat-style baseline in NumPy and SciPy, with scikit-learn and scikit-gstat for the independent checks (rules R1, V2, V3); from Stage 3, terrain analysis in NumPy and SciPy, the tiles read with tifffile, and rasterio (GDAL) and scikit-image for the independent checks (rules E3, E5); from Stage 4, Bayesian kriging in NumPy and SciPy, and scikit-learn's random forests, the quantile regression forest built on them, for the benchmark (RUNBOOK, Stage 4, D3 and D7); the kriging checked against GSTools (rule K6), and the figures in `docs/figures/` drawn with matplotlib (Stage 4's addendum) | Geostatistics by REML, as current practice recommends (`docs/REVIEW_STAGE2.md`). A library enters `requirements*.txt` when a stage uses it, not before |
| Earth Engine | not in the standard | Not used. The review of Stage 2 proposed it for Stage 3 (ML2); Stage 3 reads the newest USGS DEM tiles from The National Map instead (RUNBOOK, Stage 3, D8) | Earth Engine's copy of the DEM ends in May 2022, and The National Map needs no account, so the governing rule holds with no exception |
| Publication | Quarto → GitHub Pages | Figures by script in `docs/figures/` (Stage 4's addendum, D11); no report yet | The Quarto report waits for Stage 5's field means, the result a report leads with |

Notebooks are percent-cell Python (`notebooks/*.py`), never `.ipynb`, and
exploration only: nothing in one is load-bearing. The rules and their tests
live in `src/`.

## Why REML, and why the moment baseline stays

The models Stage 4 krigs with are fitted by REML (RUNBOOK, Stage 2, D5):
method-of-moments variograms of residuals from a fitted trend are biased,
and REML copes better with the four small fields and the clustered ones. No
Python geostatistics library used here fits by REML, so it is written with
NumPy and SciPy and checked twice: its likelihood against scikit-learn's, and
its estimates against known parameters on simulated fields (rules R1, R2).
The gstat-style moment fit stays as a baseline (D8), because it is what the
author ran in R and the bridge between the two.

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
