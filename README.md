# Soil Carbon Kriging

**Soil organic carbon mapped with its uncertainty, in Python.** Ordinary
kriging and the kriging variance on field-scale carbon stocks from eight U.S.
Midwest fields, then on Germany's agricultural soil inventory, whose published
site locations are offset by up to 4 km for data protection.

**Author:** Dr. Adewole Olagoke · [aolagoke-dsae](https://github.com/aolagoke-dsae)

![IL-RT, one of the eight fields: the predicted soil organic carbon stock at 2,903 cells of 10 m beside its predictive standard deviation, with the 245 sampled locations](docs/figures/map_IL-RT.png)

*IL-RT, one of the eight fields: the stock in 0–30 cm predicted at each 10 m cell by Bayesian kriging of 245 samples (dots), beside its uncertainty, lowest at the samples. In cross-validation the field's kriging has a modelling efficiency of 0.57, and its 90 % intervals hold 89.5 % of the held-out stocks.*

> **Status: Stage 5 run (14.11.2026): each field's mean stock, with what it means for monitoring.**
> The report: https://aolagoke-dsae.github.io/soil-carbon-kriging/
> Both datasets are pulled and pass Checkpoint 1, 9 rules of 9; Stage 2
> passes Checkpoint 2, 9 of 9: the stocks, the REML covariance models with
> their intervals and the pooling test, and the gstat-style baseline beside
> them; Stage 3 passes Checkpoint 3, 6 of 6: eleven terrain covariates at
> every location, from the newest USGS 3DEP tiles; and Stage 4 passes
> Checkpoint 4, 13 of 13: Bayesian kriging cross-validated at all 1,163
> locations under three mean models, against two forests, its kriging
> checked against GSTools, and maps of all eight fields; Stage 5 passes
> Checkpoint 5, 5 of 5: each field's mean by Bayesian block kriging, the
> samples needed and the detectable change, with R's gstat as a check
> ([`RUNBOOK.md`](RUNBOOK.md), Stages 2 to 5). Each stage's rules are fixed
> by a commit before its data is touched; Stage 2's followed a review
> against current practice ([`docs/REVIEW_STAGE2.md`](docs/REVIEW_STAGE2.md)).

## Kurzfassung

Dieses Repository kartiert den organischen Kohlenstoffvorrat im Boden
(0–30 cm) mitsamt seiner Unsicherheit, in Python. Grundlage sind 1.163
georeferenzierte Proben aus acht landwirtschaftlich genutzten Feldern im
Mittleren Westen der USA (Potash et al., 2023); die Kovarianzmodelle werden
per REML geschätzt, Geländeparameter aus USGS-3DEP-Höhendaten dienen als
Kovariaten. In dreimal wiederholter zehnfacher Kreuzvalidierung halten die
90-%-Intervalle des bayesschen Krigings an jedem Feld 88 bis 91 % der
zurückgehaltenen Vorräte, und im CRPS schneidet es an sieben von acht
Feldern besser ab als ein Quantile Regression Forest. Der mittlere Vorrat je
Feld, per bayesschem Block-Kriging geschätzt, liegt zwischen 21,8 und
65,7 Mg C/ha; um eine Änderung von 10 % des Mittels zwischen zwei
Beprobungen nachzuweisen, genügen auf einem Raster 13 bis 52 Proben statt 25
bis 103 bei einfacher Zufallsstichprobe. Die Regeln jeder Stufe werden per
Commit festgelegt, bevor ihre Daten verwendet werden; die Übertragung auf die
Bodenzustandserhebung Landwirtschaft (BZE-LW) des Thünen-Instituts folgt als
Stufe 6.

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

## Quick start

```bash
pip install -r requirements-lock.txt       # the pinned versions CI tests; requirements.txt for the newest
python -m pytest -q
```

The pulls and every stage's commands are in [`RUNBOOK.md`](RUNBOOK.md).

## Data

| Source | What | Licence | How it is pulled |
|---|---|---|---|
| Potash (2023), [figshare 23669304 v2](https://doi.org/10.6084/m9.figshare.23669304.v2) | Soil organic carbon concentration and bulk density by depth segment, at georeferenced locations in eight commercial U.S. Midwest fields. Methods: Potash et al. (2023), *Geoderma* 438, 116587, https://doi.org/10.1016/j.geoderma.2023.116587 | CC BY 4.0 | figshare REST API, version pinned; each file checked against the size and MD5 the article states |
| Poeplau et al. (2020), [BZE-LW core dataset](https://doi.org/10.3220/DATA20200203151139) | 3,104 agricultural sites in Germany with the organic carbon stock in 0–30 cm, land use and soil description. **The coordinates are randomised within 4 km of each planned point, for data protection** | CC BY 4.0 | OGC WFS 2.0 from the Thünen Atlas, paged in `pointid` order; completeness checked against the total the server states |
| U.S. Geological Survey, [3DEP 1/3 arc-second DEM](https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5), through The National Map | Elevation in metres (NAVD 88) on a 1/3 arc-second grid, about 10 m, in tiles of 1 × 1 degree: for each tile under a Potash field, its newest version | Public domain (U.S. Government work) | TNM Access API for the versions; HTTP range requests for each tile's header and the blocks under the fields, 15.8 MB of 2.2 GB; the versions pinned in the manifest |

`raw/` is not committed: the clients fetch it again. `raw/manifest.csv` is,
and records every response's URL, status, size, SHA-256 and time. How to run
the pulls and the checkpoints is in [`RUNBOOK.md`](RUNBOOK.md), Stages 1 and 3.

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
  semivariances and fits checked against GSTools' (Checkpoint 2, 9/9
  rules); a third client, which reads only the parts of five USGS tiles
  under the fields by HTTP range requests, and from them slope and mean
  curvature at three scales, TPI at three and TWI at all 1,163 locations,
  checked against GDAL, direct least-squares fits, scikit-image and a
  sparse linear solve (Checkpoint 3, 6/6 rules); Bayesian kriging in NumPy
  and SciPy, its covariance parameters integrated over a grid inside Stage
  2's bounds, under a constant mean, a quadratic trend and terrain as an
  external drift chosen by AIC inside each training fold, with plug-in
  kriging beside it; a quantile regression forest and a random forest with
  kriged residuals, each trained across the eight fields; all of them
  scored at every location by three repeats of ten-fold cross-validation on
  the CRPS, PICP and QCP, and checked against the bordered kriging system,
  explicit inverses, Stage 2's REML likelihood, a calibration simulation and
  the trees' own leaves, and against GSTools, an independent geostatistics
  library, which reproduces its kriging within 10⁻¹⁴; maps of the
  predictive mean and standard deviation at every 10 m cell of the eight
  fields, checked against the cross-validation's own predictive
  (Checkpoint 4, 13/13 rules); each field's mean stock by Bayesian block
  kriging, checked against a direct solve, a calibration simulation and R's
  gstat (Checkpoint 5, 5/5 rules); 232 unit tests, none of which makes a
  request; a CI workflow for Python 3.11 and 3.14 on Linux, 3.14 on Windows from
  the lock file, and the gstat check in R.
- **What Stage 4 found:** the selected kriging's 90 % intervals hold 88 to
  91 % of the held-out stocks at every field; carrying the parameters'
  uncertainty widens the predictive variance by 2 to 12 % against plug-in
  kriging; terrain as an external drift is selected at four fields of eight;
  the selected kriging scores better than the quantile regression forest at
  seven fields and better than the forest with kriged residuals at five;
  its mean squared deviation ratio (MSDR), 1.06 over the eight fields
  against plug-in kriging's 1.11, shows its variance about right where the
  plug-in's is too small ([`RUNBOOK.md`](RUNBOOK.md), Stage 4, *Result* and
  *Addendum: result*).

  ![Calibration: the share of held-out stocks below each predictive quantile, for Bayesian kriging, plug-in kriging and the quantile regression forest, the eight fields pooled and each field's departure from the 1:1 line](docs/figures/calibration.png)
- **What Stage 5 found:** each field's mean stock in 0–30 cm, from 21.8 Mg
  C/ha at IL-BR to 65.7 at IL-RD, agrees with its sample mean at every
  field, with an interval up to 46 % narrower; to detect a change of 10 % of
  the mean between two campaigns, 13 to 52 samples on a grid suffice under
  the fitted models, against 25 to 103 by simple random sampling
  ([`RUNBOOK.md`](RUNBOOK.md), Stage 5).

  ![The smallest detectable change in each field's mean stock against the number of samples per campaign](docs/figures/detectable_change.png)
- **Not yet:** the national-scale model for BZE-LW (Stage 6), which has no
  rules yet.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Data: CC BY 4.0, from the sources above,
attributed as their publishers ask in [`config/sources.csv`](config/sources.csv).

To cite this repository: [`CITATION.cff`](CITATION.cff), shown by GitHub as
*Cite this repository*; the release is v1.0 (Stages 1 to 5).
