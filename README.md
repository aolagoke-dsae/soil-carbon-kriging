# Soil Carbon Kriging

**Soil organic carbon mapped with its uncertainty, in Python.** Ordinary
kriging and the kriging variance on field-scale carbon stocks from eight U.S.
Midwest fields, then on Germany's agricultural soil inventory, whose published
site locations are offset by up to 4 km for data protection.

**Author:** Dr. Adewole Olagoke · [aolagoke-dsae](https://github.com/aolagoke-dsae)

> **Status: Stage 2 run (07.11.2026); Stage 3's rules fixed (08.11.2026).**
> Both datasets are pulled and pass Checkpoint 1, 9 rules of 9, and Stage 2
> passes Checkpoint 2, 9 of 9: the stocks, the REML covariance models with
> their intervals and the pooling test, and the gstat-style baseline beside
> them ([`RUNBOOK.md`](RUNBOOK.md), Stage 2). Each stage's rules are fixed
> by a commit before its data is touched. Nothing has been kriged yet.

## Why

The author's spatial statistics go back to point-pattern analyses of tree
interactions in a regenerating mangrove forest (*Acta Oecologica*, 2013,
https://doi.org/10.1016/j.actao.2013.04.001), and the author's kriging so far was done
in R, with gstat, for a grassland study (*Journal of Vegetation Science*, 2023,
https://doi.org/10.1111/jvs.13196). This repeats that kriging in Python on open
soil-carbon data, with the variogram fitted by REML as current practice
recommends, and reports what soil-carbon work needs from it: not a smooth map,
but the uncertainty of each prediction and of each field's mean. A
machine-learning benchmark tests whether a forest does better.

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
  write-once storage, a manifest row per response); both datasets pulled and
  checked against their sources (Checkpoint 1, 9/9 rules); the 0–30 cm stock
  at 1,163 locations, checked against a second computation; REML for three
  covariance families and two mean models, written in NumPy and SciPy, its
  likelihood checked against scikit-learn's and its estimates on 50
  simulated fields; the REML fits to all eight fields under both mean
  models, with 95 % profile-likelihood intervals, the pooling test for the
  small fields, a Box-Cox profile per field and a geometric-anisotropy check
  for the large fields; the gstat-style moment baseline beside them, its
  semivariances and fits checked against scikit-gstat's (Checkpoint 2, 9/9
  rules); 110 unit tests, none of which makes a request; a CI workflow for
  Python 3.11 and 3.14.
- **Not yet:** everything from Stage 3 on: the terrain covariates, kriging
  and the kriging variance, the machine-learning benchmark, and the
  comparison with the design-based estimate. Stage 3's rules, for terrain
  covariates from the newest USGS 3DEP tiles, were fixed on 08.11.2026; no
  elevation has been fetched.
  Stage 2's rules were fixed on 05.11.2026, after a review against current
  practice in [`docs/REVIEW_STAGE2.md`](docs/REVIEW_STAGE2.md). After them:
  no kriging or kriging variance, no machine-learning benchmark, no
  comparison with the design-based estimate.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Data: CC BY 4.0, from the sources above,
attributed as their publishers ask in [`config/sources.csv`](config/sources.csv).
