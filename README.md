# Soil Carbon Kriging

**Soil organic carbon mapped with its uncertainty, in Python.** Ordinary
kriging and the kriging variance on field-scale carbon stocks from eight U.S.
Midwest fields, then on Germany's agricultural soil inventory, whose published
site locations are offset by up to 4 km for data protection.

**Author:** Dr. Adewole Olagoke · [aolagoke-dsae](https://github.com/aolagoke-dsae)

> **Status: Stage 1 written (05.11.2026), not run.** The two ingestion
> clients, the Checkpoint 1 rules and their tests exist, and the tests pass on
> fixtures. No data has been pulled and nothing has been kriged. The rules
> were fixed in [`RUNBOOK.md`](RUNBOOK.md) by the first commit, before any
> data was pulled.

## Why

The author's kriging so far was done in R, for a vegetation study (*Journal of
Vegetation Science*, 2023, https://doi.org/10.1111/jvs.13196). This repeats the
method in Python on open soil-carbon data, and reports what soil-carbon work
needs from it: not a smooth map, but the uncertainty of each prediction and of
each field's mean.

## Data

| Source | What | Licence | How it is pulled |
|---|---|---|---|
| Potash (2023), [figshare 23669304 v2](https://doi.org/10.6084/m9.figshare.23669304.v2) | Soil organic carbon concentration and bulk density by depth segment, at georeferenced locations in eight commercial U.S. Midwest fields. Methods: Potash et al. (2023), *Geoderma* 438, 116587, https://doi.org/10.1016/j.geoderma.2023.116587 | CC BY 4.0 | figshare REST API, version pinned; each file checked against the size and MD5 the article states |
| Poeplau et al. (2020), [BZE-LW core dataset](https://doi.org/10.3220/DATA20200203151139) | 3,104 agricultural sites in Germany with the organic carbon stock in 0–30 cm, land use and soil description. **The coordinates are randomised within 4 km of each planned point, for data protection** | CC BY 4.0 | OGC WFS 2.0 from the Thünen Atlas, paged in `pointid` order; completeness checked against the total the server states |

`raw/` is not committed: the clients fetch it again. `raw/manifest.csv` is,
and records every response's URL, status, size, SHA-256 and time. How to run
the pulls and the checkpoint is in [`RUNBOOK.md`](RUNBOOK.md), Stage 1.

## What is real

- **Real:** two ingestion clients built on `requests` (retries, timeouts,
  write-once storage, a manifest row per response); the Checkpoint 1 rules;
  45 unit tests that replace every request; a CI workflow for Python 3.11
  and 3.14, which runs once the repository is pushed.
- **Not yet:** the pulls, Checkpoint 1's result, and everything from Stage 2
  on: variograms, kriging, the kriging variance, the comparison with the
  design-based estimate.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Data: CC BY 4.0, from the sources above,
attributed as their publishers ask in [`config/sources.csv`](config/sources.csv).
