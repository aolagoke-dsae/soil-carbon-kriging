# Runbook: Soil Carbon Kriging

Dr. Adewole Olagoke · written 5 November 2026

What to build next here, and how to know it is right. The standard this
repository follows (stack, shape, commit rule, honesty rules) is the
portfolio's, in `grid-carbon-intensity/docs/ARCHITECTURE.md`; what differs is
in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Dates belong to the Master
Plan, not to this file.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Pull both datasets through their clients; Checkpoint 1 | **run** 05.11.2026: Checkpoint 1 passed, 9/9 rules |
| 2 | Potash: stocks in 0–30 cm; covariance models by REML, with the pooling of small fields tested; moment variograms as the gstat baseline | **run** 07.11.2026: Checkpoint 2 passed, 9/9 rules |
| 3 | Terrain covariates for the eight fields: slope, curvature, topographic position and wetness from the newest USGS 3DEP 1/3 arc-second tiles, read from The National Map | **run** 08.11.2026: Checkpoint 3 passed, 6/6 rules |
| 4 | Kriging per field with the Stage 2 models: (a) against (b), the parameter uncertainty propagated, cross-validation and interval coverage (PICP, QCP); and a machine-learning benchmark against it: a quantile regression forest, and a random forest with kriged residuals, on coordinates and terrain | not yet specified |
| 5 | Each field's mean stock: block kriging against the stratified, design-based estimate the paper evaluates | not yet specified |
| 6 | BZE-LW: a quantile regression forest with kriged residuals at national scale, its covariates averaged within 4 km of each published point | not yet specified |

Stages 4–6 are a direction, not instructions. Each gets its rules, written
before its data is touched, once the stage before it has passed. The order
of Stages 3–6 and the machine-learning benchmark follow the review of the
Stage 2 draft ([`docs/REVIEW_STAGE2.md`](docs/REVIEW_STAGE2.md)).

---

## Stage 1: pull both datasets

From the repository root, in the project's environment:

```bash
python -m src.ingest.figshare_client      # Potash et al. (2023), figshare 23669304 v2
python -m src.ingest.thuenen_wfs_client   # BZE-LW site layer, Thünen Atlas WFS
python -m src.checks.checkpoint1          # exit 0 = passed, 1 = a rule failed, 2 = not pulled
python -m pytest -q
```

The clients write every response once to `raw/` and append a row to
`raw/manifest.csv` (`src/ingest/raw_store.py`). A rerun fetches only what is
missing; `--refresh` fetches everything again, as new files beside the old.
The checkpoint reads `raw/` and `config/` only and makes no request.

### Checkpoint 1: rules fixed 5 November 2026, before any pull

Checkpoint 1 passes when every rule below passes. Each rule is one function in
`src/checks/checkpoint1.py`, and `src/checks/test_checkpoint1.py` shows each
one failing on the defect it is there to catch.

**Potash et al. (2023)**, figshare article 23669304, version 2:

| Rule | Passes when |
|---|---|
| P1 provenance | The stored record is version 2 under CC BY 4.0, and every file it lists is stored with the size and MD5 the record states |
| P2 schema | `locations.csv` has `site, location_id, X, Y` and `measurements.csv` has `site, location_id, sample_depth_min, sample_depth_max, SOCc, BD`: the columns the figshare description names |
| P3 sites | Eight sites ("eight commercial U.S. Midwest fields"), the same eight in both files |
| P4 keys | `(site, location_id)` is filled and unique in `locations.csv` (`location_id` alone is not unique across sites), and every measurement row has its location |
| P5 coordinates | Every location has numeric `X` and `Y` inside the conterminous-US envelope in `config/extents.csv` |
| P6 values | Every segment's top is at or below the surface and above its bottom; where given, `SOCc` is in [0, 100) % and `BD` in (0, 2.65] g/cm³, the particle density of quartz, which no bulk density exceeds. Empty values are counted, not failed |

**BZE-LW site layer**, Thünen Atlas WFS, `geonode:bze_lw_standorte_verschleiert`:

| Rule | Passes when |
|---|---|
| B1 completeness | The pages add up to the total every page states (`numberMatched`), which is the inventory's 3,104 sites; each `pointid` appears once; every attribute the checks read is present |
| B2 CRS and geometry | EPSG:25832 points whose geometry equals `xcoord`/`ycoord` within 1 mm, all inside the layer extent published in the Atlas record (`config/extents.csv`) |
| B3 values | Where given, `kv_0_30` is numeric and within [0, 1,000] Mg/ha (a gross-error bound, not a plausibility claim), and land use is `A`, `G` or `SO` (the attribute description). Empty values are counted, not failed |

**Reported, not judged.** Recorded with the result because Stage 2 needs
them; no value of them fails the checkpoint:

- per Potash field: locations against Webster and Oliver (1992), from
  `config/variogram_sample_sizes.csv`; segments; extent; median spacing; the
  usual segments above 30 cm; how many locations cover 0–30 cm without a gap
  (changed after the first run: see *If it fails*); the deepest segment
- BZE-LW: `kv_0_30` quantiles and medians by land use, sites by land use, the
  peat flag `bze_moor`, what `kv_30_100` (stored as text) holds, the sampling
  years, and the nearest-neighbour distances of the published points

**Known, and not testable:** the BZE-LW coordinates are not the sampling
sites. For data protection each was generated at random within 4 km of the
originally planned point (the `xcoord`/`ycoord` description in the Atlas
record; `config/sources.csv`). Every later stage carries that offset.

### Result

**Passed on 5 November 2026: 9/9 rules.** The author pulled both datasets on
Windows (Python 3.14) at 19:35–19:36 UTC: seven responses, every one HTTP 200
at the first attempt and none rejected (`raw/manifest.csv`). The checkpoint
gave the same rules and diagnostics there (19:36:44Z) and on Linux (Python
3.10.12). This is the Linux run after the diagnostic change recorded under
*If it fails*; 47 tests pass.

```text
Checkpoint 1, run 2026-11-05T19:41:59Z

PASS  P1 provenance: version 2, CC BY 4.0; locations.csv, measurements.csv match the article's size and MD5
PASS  P2 schema: locations.csv 1170 rows, measurements.csv 3977 rows; every described column present
PASS  P3 sites: 8 sites in both files: IL-BR, IL-DG, IL-MC, IL-PT, IL-RD, IL-RS, IL-RT, NE
PASS  P4 keys: 1170 unique locations; all 3977 measurement rows have one
PASS  P5 coordinates: all 1170 inside lon -125..-66, lat 24..50; observed lon -96.4547..-88.1512, lat 39.0542..41.1524
PASS  P6 values: 3977 segments ordered; SOCc and BD within bounds (0 SOCc and 0 BD empty)
PASS  B1 completeness: 3104 features in 4 pages = numberMatched = 3104; pointid unique
PASS  B2 CRS and geometry: EPSG:25832 points; geometry = xcoord/ycoord within 0.001 m; all inside the published extent
PASS  B3 values: kv_0_30 within [0, 1000] Mg/ha (0 empty); land use codes in ['A', 'G', 'SO']

Potash et al. (2023), per site (reported, not judged):
 site  locations  segments variogram (Webster & Oliver 1992) extent E x N (m)  median spacing (m) usual segments above 30 cm complete 0-30 cm  deepest (cm)
IL-BR         63       310       50-99: below the 100 needed        623 x 360                21.6           0-10 10-20 20-30            60/63          90.0
IL-DG         97       559       50-99: below the 100 needed        351 x 517                40.8                 0-15 15-30            97/97          90.0
IL-MC         52       257       50-99: below the 100 needed        667 x 331                37.3                 0-15 15-30            52/52          75.0
IL-PT         93       554       50-99: below the 100 needed        577 x 366                44.5                 0-15 15-30            92/93          90.0
IL-RD        249       497     225 or more: usually reliable        742 x 406                36.3                       0-30          248/249          60.0
IL-RS        225       450     225 or more: usually reliable        697 x 400                34.2                       0-30          225/225          60.0
IL-RT        247       486     225 or more: usually reliable        432 x 673                36.6                       0-30          245/247          60.0
   NE        144       864       100-149: the minimum needed        238 x 543                11.6             0-5 5-15 15-30          144/144         120.0

BZE-LW site layer (reported, not judged):
  sites: 3104
  kv_0_30 quantiles 5/25/50/75/95 (Mg/ha): [33.8, 48.6, 62.2, 83.7, 161.0]
  sites by land use: {'A': 2234, 'G': 820, 'SO': 50}
  kv_0_30 median by land use (Mg/ha): {'A': 55.7, 'G': 88.2, 'SO': 59.4}
  bze_moor values: {'0': 2958, '1': 146}
  kv_30_100 numeric / text / empty: 3088 / 16 / 0
  kv_30_100 text values: {'NA': 16}
  sampling years: 2011-2018
  nearest published neighbour 5/25/50/75/95 (km): [4.11, 5.66, 6.83, 7.99, 10.26]

Checkpoint 1 PASSED: 9/9 rules
```

What Stage 2 takes from it (facts, not yet rules):

- **Four fields are below the 100 locations a variogram needs** (Webster and
  Oliver 1992): IL-BR 63, IL-DG 97, IL-MC 52, IL-PT 93. NE has 144; IL-RD,
  IL-RS and IL-RT have 225–249. By that reading only NE and the three large
  grids have enough locations for a variogram of their own.
- **The designs differ.** IL-RD, IL-RS and IL-RT are regular grids at about
  35 m, IL-DG and IL-PT offset grids at 41–45 m, IL-BR, IL-MC and NE irregular
  or clustered. At the five grids no two locations are closer than 20 m (the
  closest pair: 20.3 m at IL-RT, 42.8 m at IL-PT), so a nugget there will be
  extrapolated, not observed. NE has 111 pairs closer than 20 m (closest
  5.6 m), IL-BR 5 and IL-MC 2.
- **Every field's usual segments break at 30 cm**, so a 0–30 cm stock is a sum of
  whole segments. 1,163 of 1,170 locations cover 0–30 cm without a gap; the
  seven that do not are three at IL-BR, one at IL-PT and one at IL-RD, each
  missing a segment above 30 cm, and two at IL-RT (RT-B12 has no segment, RT-M6
  stops at 27 cm).
- **BZE-LW:** the published points' nearest neighbours lie 4.1–10.3 km away
  (5th–95th percentile, median 6.8 km), so they are not on the 8 km grid.
  146 sites carry `bze_moor` = 1. Grassland's median 0–30 cm stock (88.2 Mg/ha)
  is well above cropland's (55.7). `kv_30_100` is `NA` at 16 sites. The layer's
  sampling years run 2011–2018, inside the 2009–2018 project period its
  abstract gives.

### If it fails

**05.11.2026: HTTP 403 from the build environment's proxy.** The environment
the repository was written in reaches neither `api.figshare.com` nor
`atlas.thuenen.de`: its egress proxy refuses both hosts (`CONNECT tunnel
failed, response 403`). That is the environment, not the sources; both answer
from an ordinary connection. Run the two pulls where they are reachable.

**05.11.2026: a diagnostic that overstated depth coverage.** The first run
reported, per field, the locations whose deepest segment reaches 30 cm, and
gave IL-BR 63/63. Three IL-BR locations nevertheless miss a segment above
30 cm (516-1: 20–30; 520-3: 0–10; 521-3: 10–20), so no 0–30 cm stock can be
summed for them without filling one in. The diagnostic now counts a location
only when its segments run from the surface to 30 cm without a gap or an
overlap (`covers()`, tested), out of all the site's locations, with segments
or not. It is reported, not judged, so the verdict did not change.

---

## Stage 2: stocks in 0–30 cm and covariance models (Potash fields)

> **Rules fixed on 5 November 2026** by the commit that adds them, before any
> variogram was computed. Before fixing, the draft was reviewed against
> current practice ([`docs/REVIEW_STAGE2.md`](docs/REVIEW_STAGE2.md)) and changed
> as the review proposed. The only data seen so far are Checkpoint 1's counts,
> spacings and segments.

```bash
python -m src.transform.stocks       # curated/potash_stocks_0_30.parquet
python -m src.transform.reml         # curated/potash_reml.parquet: the models Stage 4 krigs with
python -m src.transform.variograms   # curated/potash_variograms.parquet: the gstat-style baseline
python -m src.checks.checkpoint2
```

Stage 2 adds `pyproj` (UTM coordinates), `scikit-gstat` (rules V2 and V3),
`scikit-learn` (rule R1) and `pyarrow` (Parquet) to `requirements.txt`.

### Decisions, fixed before any variogram

| | Decision | Why |
|---|---|---|
| D1 stock | Fixed depth, 0–30 cm: stock (Mg C/ha) = Σ SOCc (%) × BD (g/cm³) × thickness (cm), over the segments above 30 cm. No coarse-fragment correction: the files give none | Every field's usual segments end at 30 cm, so a stock is a sum of whole segments. Fixed depth suits a single survey; for change over time, equivalent soil mass is the standard (von Haden et al. 2020), and a resampling stage would use it |
| D2 incomplete locations | The seven locations without contiguous segments from 0 to 30 cm are left out, not filled in: 1,163 stocks | A filled-in segment would put a model inside the data |
| D3 scale | The original scale, Mg C/ha. Each field's skewness with its standard error, and a REML profile of the Box-Cox λ, are reported; a log-scale model becomes a Stage 4 sensitivity only for a field whose 95 % interval for λ excludes 1 | The field mean, the quantity soil-carbon accounting reports, is a linear functional of the stocks and is kriged on this scale without back-transform bias (Cressie 2006). A skewness threshold would flip on noise: its standard error is about ±0.34 at 52 points |
| D4 two mean models per field | (a) a constant mean, for ordinary kriging; (b) a second-order polynomial in the coordinates, for second-order polynomial kriging | The two methods of the author's R workflow (Olagoke et al. 2023). REML likelihoods are not comparable between different mean models, so Stage 4 compares (a) with (b) by cross-validation |
| D5 REML | For each field and each of (a) and (b), the covariance is estimated by REML: exponential, spherical and Matérn (ν estimated within 0.2–3), each with a nugget, coordinates in the field's UTM zone; the lowest AIC wins. These are the models Stage 4 krigs with | REML is consistent and less biased than moment estimates, above all with a trend (Lark et al. 2006), and copes better with small samples (Alves et al. 2009). AIC for choosing variogram models (Webster & McBratney 1989); the Matérn as a general soil model (Minasny & McBratney 2005) |
| D6 parameter uncertainty | A 95 % profile-likelihood interval for each chosen model's effective range and nugget-to-sill ratio | Stage 4 propagates it instead of treating fitted parameters as known (Marchant & Lark 2004). At the five grid fields no pair is closer than 20 m, and the nugget's interval shows what the data cannot see |
| D7 pooling, tested | The four large fields (NE, IL-RD, IL-RS, IL-RT) are fitted jointly by REML with shared correlation parameters (range, nugget ratio, ν) and field-specific means and variances, and the joint fit's AIC is compared with the sum of their own fits' AICs. If sharing is worse by at most 2 AIC units, the four small fields (IL-BR, IL-DG, IL-MC, IL-PT) take the shared parameters, re-estimated from all eight fields; otherwise each keeps its own REML fit | The large fields test the assumption before the small fields rely on it. Borrowing strength pays below about 60 points (Cui & Stein 1995), and the small fields are below the 100 a variogram needs (Webster & Oliver 1992) |
| D8 moment baseline | Matheron's estimator; bins centred on multiples of the field's median nearest-neighbour spacing w (edges 0, w/2, 3w/2, …); lags up to half the largest pairwise distance; bins of fewer than 30 pairs not fitted; spherical and exponential fitted by weighted least squares with gstat's default weights N_j/h_j² (`fit.method = 7`), nugget ≥ 0, partial sill > 0, 0 < range ≤ 2 × the largest lag. Reported beside REML and drawn in notebook 02; Stage 4 does not krig with it | The gstat workflow as the author ran it in R (Pebesma 2004), so that a Python fit can be set beside an R one. The 30 pairs are a convention chosen here, not a cited result |
| D9 scope | No kriging, no cross-validation and no map in Stage 2 | Stage 4 gets its own rules once Stages 2 and 3 have passed |

### Checkpoint 2: rules

| Rule | Passes when |
|---|---|
| S1 stocks | 1,163 stocks, one per (site, location_id); none of the seven incomplete locations; every stock finite and above 0 |
| S2 arithmetic | Every stock equals Σ SOCc × BD × thickness recomputed by a second, loop-based path, within 1e-9 Mg/ha, and a hand-worked example reproduces in the tests |
| R1 likelihood | The Gaussian log-likelihood the code computes for given parameters equals scikit-learn's `GaussianProcessRegressor.log_marginal_likelihood` for the same Matérn-plus-white-noise kernel on the same centred values, within 1e-8 relative; and the REML criterion equals a direct, unoptimised implementation of its formula within 1e-8 relative |
| R2 recovery | On 50 simulated fields of 400 points (exponential covariance, effective range 150 m, nugget-to-sill ratio 0.3, sill 100, field 600 × 400 m), with the family given, the median REML estimates of the effective range, the nugget-to-sill ratio and the sill each lie within 10 % of the truth |
| R3 fits | Every REML fit converged with its parameters inside their bounds, and every interval of D6 contains its estimate |
| R4 record | `curated/potash_reml.parquet` holds one row per field and mean model (a, b), plus the joint fits of D7: family, ν, nugget, partial sill, effective range, AIC, the D6 intervals and the number of stocks, with no empty cell except ν for a non-Matérn family; and it states which parameters each small field takes, by the comparison D7 defines |
| V1 bins | Every baseline variogram has the edges of D8, fits only bins of 30 pairs or more, and has no lag longer than half its field's largest pairwise distance |
| V2 independent estimate | scikit-gstat, given the same coordinates, values and bin edges, returns the same semivariances (relative difference ≤ 1e-9) |
| V3 baseline fits | Every baseline fit converged inside the D8 bounds, and scikit-gstat fitting the same bins and model with the same weights (`fit_sigma` = h/√N) reaches a weighted sum of squares within 1 % of ours |

**Reported, not judged:**

- per field: the number of stocks, mean, standard deviation, coefficient of
  variation, skewness with its standard error, the Box-Cox λ with its 95 %
  interval, and the R² of the second-order trend
- per REML model: family, ν, nugget, partial sill, effective range, AIC, the
  D6 intervals, the nugget-to-sill ratio with its class after Cambardella et
  al. (1994) (below 25 % strong, 25–75 % moderate, above 75 % weak spatial
  dependence), and any parameter that reached its bound
- the pooling comparison of D7, and for each large field the ratio of the
  shared to its own effective range
- the moment baseline beside REML: effective range and nugget-to-sill ratio by both
- anisotropy: directional moment variograms at 0, 45, 90 and 135°, and per
  large field a REML fit with geometric anisotropy (ratio, angle, change in
  AIC). Stage 4 adopts anisotropy only where it lowers the AIC by more than 2
- robustness: the Cressie–Hawkins estimator beside Matheron's (Lark 2000)
- notebook 02: each field's moment variogram with the baseline fit and the
  REML model drawn over it

**Known limits, accepted:**

- IL-BR, IL-MC and NE are clustered or irregular, and method-of-moments
  variograms from clustered data can be erratic (Marchant et al. 2013). That is
  one reason Stage 4 krigs with the REML models; the baseline is for comparison
- laboratory error sits inside the nugget, and modelling it gives more
  realistic uncertainty (Takoutsing, Heuvelink & Stoorvogel 2022). The files
  give no replicate precision; if the Geoderma paper reports one, it bounds
  the nugget from below in Stage 4, and otherwise this stays a limit

### Result

**Passed on 7 November 2026: 9/9 rules.** The five commands, in order:

```bash
python -m src.transform.stocks       # curated/potash_stocks_0_30.parquet
python -m src.transform.reml         # curated/potash_reml.parquet and three tables beside it
python -m src.transform.anisotropy   # curated/potash_anisotropy.parquet
python -m src.transform.variograms   # curated/potash_variograms.parquet and the baseline fits
python -m src.checks.checkpoint2     # exit 0 passed, 1 a rule failed, 2 an input missing
```

The REML run takes about two minutes, the anisotropy run under half a
minute. Both cache each step (`curated/reml_cache/`,
`curated/anisotropy_cache/`), keyed by the module's source and the step's
data, so `--budget SECONDS` can stop either early (exit 4) and a second run
continues where it stopped; a changed input or a changed module starts
afresh. Beside the REML record go every family's fit
(`potash_reml_families.parquet`), the pooling comparison
(`potash_pooling.parquet`) and the Box-Cox profiles
(`potash_boxcox.parquet`); beside the semivariances, the baseline's fits
(`potash_variogram_fits.parquet`). Notebook 02 draws them.

`requirements.txt` now has pyproj, pyarrow, scikit-learn and scikit-gstat.
`curated/` is not committed: the transforms rebuild it from `raw/`, and this
file records what they produced.

Choices made while building, inside the decisions above. None of them
depends on the stocks:

- **Effective range.** Each family is written in terms of the distance at
  which its correlation falls to 0.05, or, for the spherical, which reaches
  zero, its range. A Matérn with ν = 0.5 is then the exponential with the
  same effective range, and the bounds mean the same in every family.
- **The REML criterion** is the density of the n − p orthonormal error
  contrasts (Harville 1974), the form with −log|X′X|. It does not depend on
  how the trend's coordinates are scaled or centred, and R1's direct version
  centres them differently to test that.
- **The sill is profiled out** in closed form, so the search runs over the
  effective range, the nugget-to-sill ratio and, for the Matérn, ν.
- **Bounds**, against which R3 judges the fits: the effective range from
  half the smallest separation to three times the largest; the
  nugget-to-sill ratio from 10⁻⁶ to 1, so that a smaller nugget counts as
  none and the correlation matrix stays invertible; ν from 0.2 to 3 (D5).
- **Search:** 72 starting points on a grid (360 for the Matérn), the best two
  refined by L-BFGS-B, and a bounded Nelder–Mead as a second try where
  L-BFGS-B stops without converging.
- **Coordinates:** each field in the UTM zone of its mean longitude. IL-BR
  lies just west of 90° W and falls in zone 15, the other Illinois fields in
  16, NE in 14.
- **R1's parameter sets** come from each field's geometry and sample
  variance, never from a fit: effective ranges of 0.1 and 0.3 times the
  largest separation, nugget-to-sill ratios of 0.1 and 0.5, and ν of 0.3,
  0.5, 1.5 and 2.2 (two of scikit-learn's closed forms and its
  Bessel-function path). For the Matérn, R1's direct REML takes its
  covariance from scikit-learn's kernel and so shares no code with
  `reml.py`. For the exponential and the spherical it uses
  `reml.covariance`, whose correlations the unit tests check against closed
  forms.
- **Joint fits (D7).** The fields are independent, so a joint restricted
  likelihood is the sum of the fields' own, each with its own mean and
  maximised over its own sill; k counts the shared parameters and one sill
  per field. The effective range may run over the union of the fields'
  bounds. The large fields' joint family is the one with the lowest joint
  AIC, and the refit on all eight fields keeps it.
- **Profile intervals (D6).** From the estimate outward, in steps of 0.1 in
  log a or 0.03 in the nugget-to-sill ratio, each step 1.6 times the last,
  the other parameters re-estimated at every step; then Brent's method finds
  where the profile falls χ²₁(0.95)/2 below the maximum (to 10⁻⁴ in log a,
  10⁻⁵ in the ratio). An end the profile does not reach inside the bounds
  is the bound, and the record says so (`interval_open`). A profile rising
  above its fit by more than 10⁻⁴ would mean the fit missed the maximum, and
  R3 fails on it.
- **The Box-Cox profile (D3)** is taken under mean model (a) and the family
  (a) chose, with the Jacobian term (λ − 1) Σ log z: its maximum on a grid
  from −1 to 3, refined between neighbours (out to −2 or 4 at the grid's
  ends), and its interval's ends sought between −2 and 4.
- **The record** has one row per field and mean model for its own fit, and
  one row per member field for each joint fit, with that field's own sill.
  `takes` says whether a field's own fit or the shared parameters apply, and
  `for_stage4` marks the one model per field and mean model that Stage 4
  krigs with.
- **The baseline (D8)** takes the stocks under (a) and their OLS residuals
  from the second-order trend under (b), as gstat's `variogram()` does for
  ordinary and universal kriging. A bin is [lo, hi) and its lag the mean
  distance of its pairs. The range in D8's bound is gstat's range parameter
  a: the spherical's range, and the exponential's distance parameter, a
  third of its practical range; the baseline's effective range is reported
  as REML defines it (a, or a ln 20). For a given a the model is linear in
  the nugget and the partial sill, so they are solved exactly by
  non-negative least squares and only a is searched: 200 points on a log
  grid, then Brent's method. "c > 0" is c ≥ 10⁻⁹ times the largest
  semivariance, "0 < a" is a ≥ 0.001 times the largest lag. Of the two
  families, the lower weighted sum of squares is the baseline's.
- **Cressie–Hawkins** carries the 0.045/N² term, as scikit-gstat has it.
  **Directions** are 0, 45, 90 and 135° clockwise from north, ±22.5°, as
  gstat's `alpha` and `tol.hor`, so the four sectors partition the pairs.
- **V2 and V3's independent side** is scikit-gstat 1.0: for V2 its
  `Variogram`, given the same coordinates, values and upper bin edges (its
  bins close on the right, which differs from [lo, hi) only for a pair at
  an edge's exact distance; the pair counts agree); for V3 its model
  functions, fitted by `scipy.optimize.curve_fit`, the routine scikit-gstat
  fits with, from three starts, with sigma = h/√N, the same bins and D8's
  bounds in its parameters (its exponential range r is 3a).
- **Anisotropy**, reported: each large field's D5 family refitted with the
  distance across an axis at azimuth θ stretched by 1/ratio (ratio from 0.1
  to 1), started from the isotropic estimate at four angles. At ratio 1 it
  is the isotropic model, so the refit can only match or better it; R3
  checks that it does, and that every refit converged inside its bounds.

Run on Linux (Python 3.10.12) in 59 seconds; 110 tests pass, 47 of them from
Stage 1. The author's Windows run (Python 3.14.8, pandas 3), at 21:18:44Z
before V1–V3 were built, gave the same S1–R4; its REML fits came from this
run's cache. On Windows, `python -m pip install -r requirements.txt` adds
scikit-gstat for V2 and V3.

```text
Checkpoint 2, run 2026-11-07T21:39:44Z

PASS  S1 stocks: 1163 stocks, one per (site, location_id), none of the seven incomplete locations; all finite, 13.2 to 117.1 Mg C/ha
PASS  S2 arithmetic: largest difference from the loop-based recomputation 1.4e-14 Mg C/ha (rule: 1e-09); the hand-worked example is in src/transform/test_stocks.py
PASS  R1 likelihood: Gaussian log-likelihood against scikit-learn's GaussianProcessRegressor, largest relative difference 6.2e-13 over 128 parameter sets; REML criterion against its formula with explicit inverses, 6.5e-13 over 384, both mean models and all three families (rule: 1e-08)
PASS  R2 recovery: medians over 50 fields of 400 points: effective range 151.3 m (+0.8%; quartiles 126 to 178), nugget-to-sill 0.296 (-1.3%; 0.255 to 0.346), sill 99.4 (-0.6%; 91.5 to 108); 50/50 fits converged (rule: each median within 10%)
PASS  R3 fits: all 55 fits converged inside their bounds (48 of one field, 7 joint), and so did the 8 anisotropic refits, none below its isotropic fit; the D6 intervals of all 19 chosen models contain their estimates, and no profile rose above its fit by more than 0.0001
PASS  R4 record: 16 rows, one per field and mean model, and 16 rows for D7's joint fits; no empty cell but nu outside the Matern; the numbers of stocks as in the stock table; constant mean, sharing +6.9 AIC: the small fields take own parameters; quadratic mean, sharing -0.7 AIC: the small fields take shared parameters; one model per field and mean model marked for Stage 4
PASS  V1 bins: 16 baseline variograms, every field and mean model, with D8's edges; only and all bins of 30 pairs or more fitted (172 bins); no lag beyond half its field's largest separation
PASS  V2 independent estimate: scikit-gstat 1.0.24, given the same coordinates, values and edges: the same pair counts, and semivariances within 1.7e-14 relative over 176 bins (rule: 1e-09)
PASS  V3 baseline fits: all 32 baseline fits converged inside D8's bounds; scikit-gstat's models, fitted to the same bins with sigma = h/sqrt(N), reach weighted sums of squares within 0.00% of ours (rule: 1%)

Reported, not judged:

Stocks in 0-30 cm per field, Mg C/ha:
 site  stocks mean   sd cv (%) skewness ± se trend R2
IL-BR      60 21.8  3.8   17.2  +0.19 ± 0.31     0.20
IL-DG      97 61.0  8.2   13.5  +0.26 ± 0.24     0.09
IL-MC      52 50.9  6.6   13.1  +0.24 ± 0.33     0.31
IL-PT      92 61.5  7.4   12.0  +0.21 ± 0.25     0.17
IL-RD     248 65.4 14.2   21.7  +0.20 ± 0.15     0.08
IL-RS     225 64.0 14.7   23.0  -0.18 ± 0.16     0.14
IL-RT     245 61.7 17.4   28.2  +0.18 ± 0.16     0.44
   NE     144 64.4  7.4   11.5  +0.33 ± 0.20     0.31

REML models, (a) constant and (b) second-order mean; 95 % profile intervals, * where an end is the bound:
field mean         family next family nugget partial sill effective range (m)                 nugget/sill      at bound  takes
IL-BR  (a)      spherical    +0.0 AIC    0.0         14.0      34 [7*, 1981*]  0.00 [0.00*, 1.00*] strong           tau    own
IL-DG  (a)      spherical    +0.1 AIC   63.7         18.2   1820 [19*, 1820*]     0.78 [0.28, 1.00*] weak     eff_range    own
IL-MC  (a)      spherical    +0.2 AIC   30.4         59.8    2064 [65, 2064*] 0.34 [0.00*, 0.92] moderate     eff_range    own
IL-PT  (a)      spherical    +0.5 AIC   45.9         11.0    401 [128, 1981*]      0.81 [0.19, 0.98] weak             -    own
IL-RD  (a)      spherical    +0.3 AIC  135.5         88.4       237 [86, 289]  0.61 [0.34, 0.82] moderate             -    own
IL-RS  (a)      spherical    +0.5 AIC  108.8        115.9      164 [120, 209]  0.48 [0.27, 0.69] moderate             -    own
IL-RT  (a) matern nu 0.23    +1.1 AIC    0.0        626.5   2401 [212, 2401*]   0.00 [0.00*, 0.25] strong eff_range,tau    own
   NE  (a)    exponential    +1.2 AIC    0.0         63.2       129 [79, 350]   0.00 [0.00*, 0.14] strong           tau    own
IL-BR  (b)      spherical    +0.6 AIC    0.0         12.3      27 [7*, 1981*]  0.00 [0.00*, 1.00*] strong           tau shared
IL-DG  (b)    exponential    +0.0 AIC   64.7          0.0     19 [19*, 1820*]    1.00 [0.00*, 1.00*] weak eff_range,tau shared
IL-MC  (b)      spherical    +0.8 AIC    0.0         35.0      64 [8*, 2064*]  0.00 [0.00*, 1.00*] strong           tau shared
IL-PT  (b)    exponential    +0.0 AIC   47.5          0.0     21 [21*, 1981*]    1.00 [0.00*, 1.00*] weak eff_range,tau shared
IL-RD  (b)      spherical    +2.1 AIC  126.1        125.3       245 [88, 296]  0.50 [0.06, 0.77] moderate             -    own
IL-RS  (b)      spherical    +0.7 AIC  111.3        108.0       160 [97, 305]  0.51 [0.03, 0.74] moderate             -    own
IL-RT  (b)      spherical    +0.3 AIC    0.0        177.5         73 [64, 92]   0.00 [0.00*, 0.28] strong           tau    own
   NE  (b)      spherical    +1.5 AIC    2.6         45.6         68 [51, 84]   0.05 [0.00*, 0.26] strong             -    own

D7, the large fields' correlation parameters shared:
mean         shared AIC jointly AIC own, summed difference small fields take shared range (m) shared nugget/sill                            shared/own range
 (a) matern nu 0.24      6620.2          6613.3       +6.9               own 268 [144, 2491*] 0.00 [0.00*, 0.20] IL-RD 1.13, IL-RS 1.64, IL-RT 0.11, NE 2.09
 (b)    exponential      6404.8          6405.5       -0.7            shared     93 [76, 133] 0.00 [0.00*, 0.20] IL-RD 0.38, IL-RS 0.58, IL-RT 1.26, NE 1.37

D7, the shared parameters re-estimated from all eight fields:
mean      family    range (m)        nugget/sill                            small fields' sills
 (b) exponential 79 [66, 106] 0.00 [0.00*, 0.22] IL-BR 16.6, IL-DG 76.6, IL-MC 36.1, IL-PT 58.2

D3, the Box-Cox lambda under mean model (a), 95 % profile interval:
field      family             lambda interval excludes 1
IL-BR   spherical  1.87 [0.65, 3.16]                  no
IL-DG   spherical  1.61 [0.79, 2.47]                  no
IL-MC   spherical 3.63 [1.88, 4.00*]                 yes
IL-PT   spherical 2.76 [1.18, 4.00*]                 yes
IL-RD   spherical  1.06 [0.67, 1.46]                  no
IL-RS   spherical  1.48 [1.10, 1.89]                 yes
IL-RT      matern  0.62 [0.30, 0.95]                 yes
   NE exponential  1.87 [0.79, 2.97]                  no

D8, the moment baseline beside REML (effective ranges as REML defines them):
field mean    baseline range (m) nugget/sill     at bound           REML    REML range (m)    REML nugget/sill
IL-BR  (a)   spherical       159        0.79            -      spherical    34 [7*, 1981*] 0.00 [0.00*, 1.00*]
IL-DG  (a)   spherical         0        1.00 partial_sill      spherical 1820 [19*, 1820*]  0.78 [0.28, 1.00*]
IL-MC  (a)   spherical       171        0.61            -      spherical  2064 [65, 2064*]  0.34 [0.00*, 0.92]
IL-PT  (a)   spherical       528        0.79            a      spherical  401 [128, 1981*]   0.81 [0.19, 0.98]
IL-RD  (a) exponential       220        0.52            -      spherical     237 [86, 289]   0.61 [0.34, 0.82]
IL-RS  (a) exponential       291        0.45            -      spherical    164 [120, 209]   0.48 [0.27, 0.69]
IL-RT  (a) exponential      1487        0.15            - matern nu 0.23 2401 [212, 2401*]  0.00 [0.00*, 0.25]
   NE  (a) exponential       139        0.00       nugget    exponential     129 [79, 350]  0.00 [0.00*, 0.14]
IL-BR  (b)   spherical        30        0.99            -      spherical    27 [7*, 1981*] 0.00 [0.00*, 1.00*]
IL-DG  (b)   spherical         0        1.00 partial_sill    exponential   19 [19*, 1820*] 1.00 [0.00*, 1.00*]
IL-MC  (b)   spherical        72        0.37            -      spherical    64 [8*, 2064*] 0.00 [0.00*, 1.00*]
IL-PT  (b)   spherical         0        1.00 partial_sill    exponential   21 [21*, 1981*] 1.00 [0.00*, 1.00*]
IL-RD  (b) exponential       139        0.52            -      spherical     245 [88, 296]   0.50 [0.06, 0.77]
IL-RS  (b)   spherical       137        0.61            -      spherical     160 [97, 305]   0.51 [0.03, 0.74]
IL-RT  (b)   spherical       120        0.41            -      spherical       73 [64, 92]  0.00 [0.00*, 0.28]
   NE  (b)   spherical        62        0.03            -      spherical       68 [51, 84]  0.05 [0.00*, 0.26]

Cressie-Hawkins beside Matheron over the fitted bins:
field mean  bins Cressie-Hawkins / Matheron, median smallest largest
IL-BR  (a)    13                               1.00     0.81    1.16
IL-DG  (a)     6                               0.80     0.68    0.82
IL-MC  (a)     8                               1.01     0.92    1.10
IL-PT  (a)     6                               1.02     0.97    1.11
IL-RD  (a)    10                               0.98     0.87    1.00
IL-RS  (a)    10                               0.97     0.94    1.01
IL-RT  (a)    10                               1.01     0.90    1.05
   NE  (a)    23                               0.99     0.88    1.09
IL-BR  (b)    13                               1.01     0.84    1.15
IL-DG  (b)     6                               0.79     0.71    0.80
IL-MC  (b)     8                               1.02     0.94    1.18
IL-PT  (b)     6                               1.04     0.95    1.07
IL-RD  (b)    10                               0.98     0.87    1.01
IL-RS  (b)    10                               0.96     0.93    1.00
IL-RT  (b)    10                               1.01     0.90    1.03
   NE  (b)    23                               0.96     0.80    1.04

Geometric anisotropy by REML, the large fields (azimuth of the major axis, clockwise from north; Stage 4 adopts it only where the AIC falls by more than 2; directional moment variograms are in curated/potash_variograms.parquet and notebook 02):
field mean         family major axis ratio  ranges (m) change in AIC Stage 4 adopts      at bound
IL-RD  (a)      spherical     80 deg  0.30    306 / 92          -8.6            yes             -
IL-RS  (a)      spherical     90 deg  0.75   185 / 139          +2.2             no             -
IL-RT  (a) matern nu 0.21    154 deg  0.47 2401 / 1119          -4.2            yes eff_range,tau
   NE  (a)    exponential    177 deg  0.73   157 / 114          +2.1             no           tau
IL-RD  (b)      spherical     82 deg  0.27    306 / 81          -7.9            yes             -
IL-RS  (b)      spherical     90 deg  0.74   186 / 137          +2.1             no             -
IL-RT  (b)      spherical     26 deg  0.78     84 / 65          -0.9             no           tau
   NE  (b)      spherical    157 deg  0.86     75 / 65          +3.3             no             -

Checkpoint 2 PASSED: 9/9 rules
```

**Reported, not judged:**

- **IL-BR's stocks are about a third of the other fields'**: a mean of
  21.8 Mg C/ha, against 50.9–65.4 in the other seven. Its median SOCc in the
  segments above 30 cm is 0.57 %, against 1.6–1.9 % elsewhere, while its bulk
  density is ordinary (median 1.12 g/cm³). The figshare description gives
  SOCc in % for every site, so this may be the field itself. The site table
  in Potash et al. (2023, https://doi.org/10.1016/j.geoderma.2023.116587) is
  where to confirm it, before Stage 4 relies on it.
- Every field's skewness lies within two standard errors of zero; the
  largest is NE's, +0.33 ± 0.20. The second-order trend explains 8–44 % of
  the variance: 44 % at IL-RT, 31 % at NE and IL-MC.
- **The small fields cannot estimate their own variograms**, as D7
  expected. Under (a) IL-DG's and IL-MC's effective ranges sit at their
  upper bound and IL-BR's and IL-PT's intervals reach it; IL-BR's
  nugget-to-sill interval spans 0 to 1. Under (b) IL-DG and IL-PT are pure
  nugget, and IL-BR's and IL-MC's range intervals run from bound to bound.
  Under (b) the four take the shared parameters; under (a) they keep these
  fits, and their wide intervals are what Stage 4 propagates (D6).
- **IL-RT under (a)** is a Matérn with ν = 0.23 and its effective range at
  the 2,401 m bound, which looks like the trend (R² 0.44) taken up by a
  long-range covariance. Under (b) it is a spherical of 73 m [64, 92].
  Stage 4's cross-validation of (a) against (b) is where that is settled.
- **D7.** Under (a) sharing costs 6.9 AIC, so every field keeps its own fit.
  Under (b) it gains 0.7, so the four small fields take the exponential
  re-estimated from all eight: 79 m [66, 106], nugget-to-sill 0.00
  [0*, 0.22]. The large fields' shared-to-own ranges under (b) run from 0.38
  (IL-RD) to 1.37 (NE): sharing wins on AIC because each field's own range
  is imprecise and sharing saves six parameters, not because the four agree
  closely.
- **Ten of the sixteen nugget-to-sill intervals reach 0**, their lower
  bound: the data cannot rule out a zero nugget there. At the five grid
  fields, with no pair closer than 20 m, any nugget is extrapolated
  (Checkpoint 1).
- **D3.** Four fields' λ intervals exclude 1: IL-RT's below it (0.62
  [0.30, 0.95]), where a log scale is the natural sensitivity, and IL-MC's,
  IL-PT's and IL-RS's above it (3.63, 2.76, 1.48), where a log transform
  would add skew rather than remove it. D3 names a log-scale sensitivity,
  which does not fit λ above 1, so Stage 4's rules have to say what the
  sensitivity is there before Stage 4 touches the data. The λ under (a) also
  carries any trend, IL-RT's most.
- **The baseline beside REML.** Where both see structure they agree at NE
  (139 against 129 m under (a), 62 against 68 m under (b)) and at IL-RD
  under (a) (220 against 237 m). Elsewhere they part: under (b) IL-RD's
  baseline range is 139 m against REML's 245 m and IL-RT's 120 m against
  73 m; under (a) IL-RS's is 291 m against 164 m. The baseline finds no
  structure at IL-DG under either mean model or at IL-PT under (b), its
  partial sill at the floor, where REML's own fits put the range at a bound;
  at IL-BR under (a) it gives 159 m where REML's interval runs from bound to
  bound. These are the comparisons D8 exists for; Stage 4 krigs with REML.
- **Cressie–Hawkins** agrees with Matheron within a few per cent at every
  field but IL-DG, where it is about a fifth lower under both mean models
  (median ratio 0.80 and 0.79): a few large differences raise Matheron's
  estimate there.
- **Anisotropy.** At IL-RD the REML refit puts the major axis near east–west
  (80° and 82°), the across-axis range about 0.3 of the along-axis one, and
  lowers the AIC by 8.6 under (a) and 7.9 under (b): Stage 4 adopts it there.
  At IL-RT it lowers the AIC by 4.2 under (a) (major axis 154°, ratio 0.47),
  with the range at its bound, but not under (b) (−0.9), so under (a) it
  most likely carries the trend; by the rule that comes with this check,
  Stage 4 adopts it for (a) only. IL-RS (+2.2, +2.1) and NE (+2.1, +3.3) stay isotropic. In
  the directional moment variograms (notebook 02), IL-RD's 0° direction
  rises above the others beyond about 300 m under both mean models.

Stage 2 has passed. Two points carry forward to Stage 4's rules: what D3's
sensitivity is where λ lies above 1, and IL-BR's low stocks, to be checked
against the paper's site table.

### If it fails

**07.11.2026: `ImportError: Unable to find a usable engine` on Windows.** An
environment set up for Stage 1 lacks pyarrow, and scikit-learn, which R1
needs: `python -m pip install -r requirements.txt` adds both.

**07.11.2026: a `Pandas4Warning` from R4 under pandas 3.** R4's test for
empty text cells used `select_dtypes(include="object")`, which pandas 3
still lets cover its new string columns but warns it will stop doing. It
now compares every cell with an empty string, which works the same under
pandas 2 and 3. The verdict did not change.

### References for Stage 2

- Alves, M. de C., da Silva, F. M., Moraes, J. C., et al. (2009). Geostatistical analysis of the spatial variation of the berry borer and leaf miner in a coffee agroecosystem. *Precision Agriculture*, 12(1), 18–31. https://doi.org/10.1007/s11119-009-9151-z
- Cambardella, C. A., Moorman, T. B., Novak, J. M., et al. (1994). Field-scale variability of soil properties in central Iowa soils. *Soil Science Society of America Journal*, 58(5), 1501–1511. https://doi.org/10.2136/sssaj1994.03615995005800050033x
- Cressie, N. (2006). Block kriging for lognormal spatial processes. *Mathematical Geology*, 38(4), 413–443. https://doi.org/10.1007/s11004-005-9022-8
- Cui, H., & Stein, A. (1995). Extension of spatial information, Bayesian kriging and updating of prior variogram parameters. *Environmetrics*, 6(4), 373–384. https://doi.org/10.1002/env.3170060406
- Harville, D. A. (1974). Bayesian inference for variance components using only error contrasts. *Biometrika*, 61(2), 383–385. https://doi.org/10.1093/biomet/61.2.383
- Lark, R. M. (2000). A comparison of some robust estimators of the variogram for use in soil survey. *European Journal of Soil Science*, 51(1), 137–157. https://doi.org/10.1046/j.1365-2389.2000.00280.x
- Lark, R. M., Cullis, B. R., & Welham, S. J. (2006). On spatial prediction of soil properties in the presence of a spatial trend: the empirical best linear unbiased predictor (E-BLUP) with REML. *European Journal of Soil Science*, 57(6), 787–799. https://doi.org/10.1111/j.1365-2389.2005.00768.x
- Marchant, B. P., & Lark, R. M. (2004). Estimating variogram uncertainty. *Mathematical Geology*, 36(8), 867–898. https://doi.org/10.1023/b:matg.0000048797.08986.a7
- Marchant, B. P., Viscarra Rossel, R. A., & Webster, R. (2013). Fluctuations in method-of-moments variograms caused by clustered sampling and their elimination by declustering and residual maximum likelihood estimation. *European Journal of Soil Science*, 64(4), 401–409. https://doi.org/10.1111/ejss.12029
- Minasny, B., & McBratney, A. B. (2005). The Matérn function as a general model for soil variograms. *Geoderma*, 128(3–4), 192–207. https://doi.org/10.1016/j.geoderma.2005.04.003
- Olagoke, A., Jeltsch, F., & Tietjen, B. (2023). Small-scale heterogeneity shapes grassland diversity in low-to-intermediate resource environments. *Journal of Vegetation Science*, 34(4). https://doi.org/10.1111/jvs.13196
- Pebesma, E. J. (2004). Multivariable geostatistics in S: the gstat package. *Computers & Geosciences*, 30(7), 683–691. https://doi.org/10.1016/j.cageo.2004.03.012; default weights from the `fit.variogram` documentation, https://r-spatial.github.io/gstat/reference/fit.variogram.html
- Takoutsing, B., Heuvelink, G. B. M., & Stoorvogel, J. J. (2022). Accounting for analytical and proximal soil sensing errors in digital soil mapping. *European Journal of Soil Science*, 73(2). https://doi.org/10.1111/ejss.13226
- von Haden, A. C., Yang, W. H., & DeLucia, E. H. (2020). Soils' dirty little secret: depth-based comparisons can be inadequate for quantifying changes in soil organic carbon and other mineral soil properties. *Global Change Biology*, 26(7), 3759–3770. https://doi.org/10.1111/gcb.15124
- Webster, R., & McBratney, A. B. (1989). On the Akaike Information Criterion for choosing models for variograms of soil properties. *Journal of Soil Science*, 40(3), 493–496. https://doi.org/10.1111/j.1365-2389.1989.tb01291.x
- Webster, R., & Oliver, M. A. (1992). Sample adequately to estimate variograms of soil properties. *Journal of Soil Science*, 43(1), 177–192. https://doi.org/10.1111/j.1365-2389.1992.tb00128.x

---

## Stage 3: terrain covariates (Potash fields)

> **Rules fixed on 8 November 2026** by the commit that adds them, before any
> elevation was fetched. The draft of 7 November was revised after review:
> the DEM now comes from The National Map in its newest versions, not from
> Earth Engine (D8); the reasons for the covariates and their scales cite
> the literature; and TWI follows a published guideline (D6). The only data
> seen are the locations' coordinates from Stage 2 and The National Map's
> listing of tile versions. No stock enters this stage.

```bash
python -m src.ingest.tnm_client      # raw/TNM/: the listings and the parts of the tiles read, not committed
python -m src.transform.terrain      # curated/potash_terrain.parquet and curated/terrain/<site>.npz
python -m src.checks.checkpoint3     # exit 0 passed, 1 a rule failed, 2 an input missing
```

The client needs no account. For each field it asks the TNM Access API
which tiles of the 1/3 arc-second DEM cover the field's patch, takes the
newest version of each (D1), and reads only the parts of it that it needs,
by HTTP range requests: the tile's header and the internal blocks that
cover the patch (D2). Every response goes to `raw/TNM/` write-once with a
manifest row, as in Stage 1, the byte range in the row's `revision`; `raw/`
stays out of git, and the manifest, which is committed, pins the versions.
For each version the client also stores the metadata record the listing
links, which names its sources (it names none: see *If it fails*). Stage 3
adds `tifffile` and `imagecodecs` (reading the tiles), `rasterio` (rule E3)
and `scikit-image` (rule E5) to `requirements.txt`.

Terrain is a covariate available at 10 m for all eight fields. In a
Michigan field sampled on a 61 m grid, maps of soil carbon that drew on
terrain attributes did better than ordinary kriging, while on a 30 m grid
the methods differed little (Mueller & Pierce 2003): the question Stage 4's
benchmark puts to these fields.

### Decisions, fixed before any elevation

| | Decision | Why |
|---|---|---|
| D1 source | USGS 3DEP 1/3 arc-second DEM (about 10 m), as The National Map distributes it: GeoTIFF tiles of 1 × 1 degree with 6 cells of overlap on every side, in geographic coordinates on NAD83, elevations in metres on NAVD88. For each block a patch touches, the version titled "USGS 1/3 Arc Second" with the latest publication date in the TNM Access API's listing on the day of the pull. A later pull fetches the versions the manifest records, not newer ones | One seamless DEM with full coverage of the conterminous US (USGS), so the eight fields stand on the same footing, and in the public domain. USGS updates the tiles as new sources arrive and keeps the older versions under their dates (ScienceBase record), so the newest version is the best available on the day of the pull, and a pinned one can be fetched again |
| D2 grid and resampling | Each field's patch in its Stage 2 UTM zone (EPSG 32614, 32615 or 32616), with 10 m cells whose edges lie on multiples of 10 m, covering the field's locations and 1,000 m beyond them on every side: 226–275 by 235–269 cells. From each tile only the internal blocks that cover the patch, with two native cells to spare, are read; the tiles are Cloud Optimized GeoTIFFs (USGS). The native values are resampled bilinearly to the patch's cell centres, which PROJ takes from UTM to NAD83 longitude and latitude; where two tiles cover the same native cell (IL-RS straddles 40° N), the newer version's value is used | The coordinates the stocks already have, so a location's cell follows from its x and y. 1,000 m holds the largest neighbourhood (D5, 450 m) with room to spare and gives the flow routing (D6) a kilometre of ground upslope. Reading the blocks alone keeps the pull to megabytes, where the five whole tiles would come to about 2.2 GB. Nearest-neighbour resampling would shift values by up to half a native cell and put steps into the slopes |
| D3 values at the locations | The elevation and every covariate's grid are interpolated bilinearly, between cell centres, at the location's Stage 2 coordinates | A location seldom falls on a cell's centre, and an interpolated value, unlike the nearest cell's, does not jump as the location moves |
| D4 slope and curvature | A quadratic in the two coordinates, fitted by least squares to the elevations of an n × n window for n = 3, 9 and 27 (30, 90 and 270 m), its coefficients computed as fixed convolution kernels; from it, the slope (degrees) and the mean curvature (Shary et al. 2002; 1/m, positive on convex ground) at the window's centre | Slope governs where soil and water move, curvature where they converge and collect. On a 5.4 ha toposequence, slope and the wetness index were the terrain attributes most closely correlated with surface organic matter, each explaining about half its variability (Moore et al. 1993), and low-lying, concave parts of a field accumulate organic matter (Adhikari et al. 2022). One method serves every scale: the Evans–Young quadratic, fitted over larger windows as MultiscaleDTM does (Ilich et al. 2023). The best scale is not known in advance: in a forested watershed, total carbon was predicted best from neighbourhoods of 150–300 m, and grids finer than 30 m represented terrain better, while 1–5 m was not necessarily better (Maynard & Johnson 2014). Hence a 10 m grid and three scales, in steps of three up to 270 m (each field's locations span 520–740 m along their longer side), among which Stage 4 selects, as multi-scale terrain analysis for soil mapping does (Behrens et al. 2010). Mean curvature is defined at any slope; profile and plan curvature are curvatures of the slope line and of the contour line (Minár et al. 2020), which do not exist on level ground and are unstable where it is nearly level |
| D5 topographic position | TPI: a cell's elevation minus the mean elevation of the cells whose centres lie within r of its own, itself excluded (De Reu et al. 2013), for r = 50, 150 and 450 m | Whether a location sits above or below the ground around it. TPI depends on the size of that neighbourhood (De Reu et al. 2013), hence three scales, the largest reaching beyond the field. In a field on Texas Blackland Prairie soils, TPI was among the top predictors of soil-health measures, organic carbon among them, in a random forest (Adhikari et al. 2022) |
| D6 wetness | TWI = ln(a / tan β) (Beven & Kirkby 1979), computed as Kopecký et al. (2021) recommend: the multiple-flow FD8 algorithm with a flow dispersion of 1.0, a flow width equal to the cell size, and the local slope. The DEM is first filled by Priority-Flood with small increments across filled ground (Barnes et al. 2014), so that every cell drains. Each cell's flow is shared among its lower neighbours in proportion to the slope towards each, drop over distance (FD8 with an exponent of 1; Freeman 1991; Quinn et al. 1991), and leaves the patch where a cell has no lower neighbour inside it. a is the area draining through a cell, itself included, divided by the cell width; tan β is the slope of the 3 × 3 fit (D4) to the filled DEM, but at least 0.001 | With slope, the wetness index was the attribute most closely correlated with surface organic matter on the toposequence (Moore et al. 1993), and in the Texas field a wetness index was the strongest predictor of organic carbon (Adhikari et al. 2022). Against soil moisture measured in 45 temperate-forest plots, the flow algorithm decided how well TWI did, while flow width and slope mattered little; a dispersion near 1.0 did best, and the downslope gradient did worse than the local slope (Kopecký et al. 2021). In tundra too, FD8 did best and withstood elevation errors, and a flow width equal to the cell size, the pseudo-specific catchment area, is the common choice (Riihimäki et al. 2021). No variant is best everywhere (Sørensen et al. 2006), so the guideline's is fixed here rather than tuned to the stocks. The floor is a numerical guard where the filled DEM is level, a rise of 2 cm across the 20 m of a 3 × 3 window, and how much it matters is reported |
| D7 left out | Aspect, profile and plan curvature (D4), and any comparison with the stocks: Stage 3 reads only the stock table's keys and coordinates, and selects or transforms nothing against the stocks | Aspect is undefined where the ground is level, and it is circular, so it needs a sine and cosine transformation before any use (Amatulli et al. 2018): two covariates for one direction. Choosing among covariates by their relation to the stocks is Stage 4's, under rules written before Stage 4 touches the data |
| D8 The National Map, not Earth Engine | No account is needed: the tiles are public, and `raw/` stays uncommitted, as in Stages 1 and 2. Earth Engine, which the review of Stage 2 proposed for this stage (ML2), is not used | The author chose the newest tiles on 8 November. Earth Engine's catalogue gives its copy of the collection dates up to May 2022, while The National Map has published newer versions of at least three of the five tiles since: n40w089, under five of the fields, in December 2022 and January 2024; n41w089 in 2022, 2024 and 2026; and n42w097, under NE, in December 2022 and February 2023. The governing rule then holds with no exception, and reading blocks keeps the pull small (D2) |
| D9 scope | No model and no map of predictions. The covariate grids are kept beside the table, in `curated/terrain/`, for Stages 4 and 5 | Stage 4 gets its own rules once Stage 3 has passed |

### Checkpoint 3: rules

| Rule | Passes when |
|---|---|
| E1 provenance | For each field the listing is stored, and each tile's version is the newest "USGS 1/3 Arc Second" version the listing gives for its block. Every response (listings, metadata records, and the header and block ranges) is in `raw/TNM/` with a manifest row whose SHA-256 it matches; the ranges of one version all report the same ETag and total size, so the file did not change between them. The transform reads stored responses only, and none of its reads falls outside a stored range |
| E2 grid | Every patch grid is in its field's Stage 2 EPSG, has 10 m cells with edges on multiples of 10 m, and reaches at least 1,000 m beyond every location of its field on every side. Every native window covers its patch's cell centres with at least one native cell to spare on every side and holds no nodata cell; where two tiles meet (IL-RS), their cells lie on one lattice |
| E3 decoding and registration | GDAL, through rasterio, reading the same stored bytes, returns the same native values exactly and the same cell geometry within 1e-9 of a cell; each tile's grid spans the bounding box its listing gives within 0.001 of a cell; and a plane laid on GDAL's cell centres, resampled by this code, is reproduced at every patch cell's centre within 1e-6 m |
| E4 local derivatives | At every location's cell and for every window, the slope and the mean curvature from the convolution kernels equal those from a direct least-squares fit to the window's elevations (within 1e-9 relative, or 1e-12 absolute near zero), and every TPI equals one computed directly, cell by cell, over its neighbourhood within 1e-9 m. The unit tests reproduce closed forms: on a plane, its slope, zero curvature and zero TPI, and its own value interpolated at any point; on a paraboloid, the slope and mean curvature at every window size, and the TPI its neighbourhood's offsets give |
| E5 flow | The filled DEM is nowhere below the original, equals it on the patch's edge, and equals scikit-image's morphological reconstruction by erosion, the same fill without the increments, within 1e-6 m; every cell off the edge has a strictly lower neighbour; the accumulated areas equal the solution, by SciPy's sparse solver, of the linear system the flow fractions define, within 1e-9 relative; and the area leaving the patch equals the patch's area within 1e-9 relative. The unit tests check that on a plane sloping along a grid axis each column's accumulated area is that of its own column and the columns upslope of it |
| E6 table | `curated/potash_terrain.parquet` has 1,163 rows, one per (site, location_id) of the stock table, with eleven covariates: the elevation (D3), slope and curvature at 30, 90 and 270 m, TPI at 50, 150 and 450 m, and TWI; every value finite, every slope in [0°, 90°) |

**Reported, not judged:**

- per field, the patch's size; per tile, the version used, the bytes read
  against the file's size, and the sources its metadata record names (for
  the whole block, not the field). The records name none, so their period
  of content is reported instead (changed after the first run: see *If it
  fails*)
- at the locations, per field: elevation (lowest, median, highest,
  interquartile range); the median slope for each window; the interquartile
  ranges of curvature and TPI at each scale; the range of TWI
- the fill: the share of each patch's cells it raised, its largest rise and
  the locations on raised cells
- the floor of tan β (D6): the share of cells, and of locations, at it, and
  the Spearman correlation at the locations between TWI with floors of
  0.0001 and 0.01, which shows how much the choice of 0.001 matters
- the locations whose upslope area includes a cell on the patch's edge,
  where the TWI is a lower bound
- collinearity among the eleven covariates at the locations: Spearman
  correlations, pooled and per field, and the pairs above 0.9 in absolute
  value. No stock is involved
- notebook 03: each field's hillshaded DEM with its locations, and the
  covariates' grids

**Known limits, accepted:**

- The 1/3 arc-second DEM is seamless but joins sources of different methods
  and dates. Current 3DEP DEMs come almost entirely from lidar and IfSAR,
  older ones also from the contours of topographic maps (USGS). A tile's
  metadata record gives only the period of its sources, for the whole
  block, not for the field (see *If it fails*).
- The source grid is geographic, on NAD83 (USGS), and is resampled to 10 m
  in UTM: values are smoothed over about a cell, and the 30 m slope inherits
  that. The locations' coordinates (WGS 84 in Stage 2) and the DEM's NAD83
  are treated as one datum, as PROJ's default transformation between them
  does, which leaves coordinates unchanged; E3 tests the registration to
  the tiles' own grids, not to the ground.
- The fields' boundaries are not in the files, so the patches include roads,
  ditches, field edges and farmsteads, and a neighbourhood near a field's
  edge describes its surroundings as much as the field.
- TWI describes surface flow inside the patch. Catchments that reach the
  patch's edge are cut off there (reported); across a filled depression,
  flow follows the order in which the fill reached its cells, a convention,
  not a measurement; and tile drainage or irrigation, if a field has them
  (the files do not say), are not represented.
- The DEM's dates and the sampling dates differ; the covariates assume the
  surface did not change in between, which land levelling would break.
- A core samples a point, while the covariates describe 10 m cells and
  larger windows, and how precisely the locations were positioned is not
  known here.
- A later pull depends on The National Map keeping the dated versions in
  its historical folder; the manifest's SHA-256 shows whether a pull
  matches the first.

### Result

**Passed on 8 November 2026: 6/6 rules**, at the checkpoint's second run.
The first found the plane of E3 off by 2.6 × 10⁻⁶ m in IL-RS, which changed
the transform, not the rule (see *If it fails*). The three commands, in
order:

```bash
python -m src.ingest.tnm_client      # run on Windows: 42 responses, 16.08 MB
python -m src.transform.terrain      # curated/potash_terrain.parquet and curated/terrain/<site>.npz
python -m src.checks.checkpoint3     # exit 0 passed, 1 a rule failed, 2 an input missing
```

The pull ran on 8 November from 11:41:46 to 11:42:08 UTC, on Windows,
because the build environment's proxy refuses The National Map (see *If it
fails*): eight listings, five metadata records and 29 byte ranges of five
tile versions, 15.8 MB of the 2.17 GB the five files hold. The transform
and the checkpoint take about half a minute each; the checkpoint
recomputes every field from `raw/`. A second pull finds nothing to fetch,
and `--refresh` asks for the listings again while the versions stay pinned
(D1).

**The tiles**, each the newest version its listing gave on 8 November.
Three were published on 2 September 2026; all five are later than May
2022, where Earth Engine's catalogue ends its copy (D8). *Content dated* is
each metadata record's period of content; no record names its sources.

| Tile | Version | Fields | Read of the file (MB) | Content dated |
|---|---|---|---|---|
| n40w089 | 2024-01-04 | IL-DG, IL-PT, IL-RD, IL-RS (south), IL-RT | 6.70 of 426 | 2012-02-17 to 2022-06-28 |
| n40w090 | 2026-09-02 | IL-MC | 2.79 of 430 | 2021-03-06 to 2024-03-29 |
| n40w091 | 2026-09-02 | IL-BR | 3.27 of 462 | 2015-04-26 to 2024-03-29 |
| n41w089 | 2026-09-02 | IL-RS (north) | 1.64 of 425 | 2019-04-16 to 2024-03-29 |
| n42w097 | 2023-02-10 | NE | 1.38 of 429 | 1965-01-01 to 2021-04-01 |

**At the locations, per field:** the patch in columns × rows; elevation,
lowest / median / highest, and its interquartile range; the median slope
for each window; the range of TWI.

| Field (locations) | Patch | Elevation (m) | IQR (m) | Median slope, 30 / 90 / 270 m (°) | TWI |
|---|---|---|---|---|---|
| IL-BR (60) | 263 × 238 | 190.0 / 191.5 / 192.0 | 0.87 | 0.77 / 0.85 / 0.61 | 6.2–11.5 |
| IL-DG (97) | 237 × 253 | 194.5 / 195.2 / 195.6 | 0.23 | 0.23 / 0.15 / 0.07 | 7.2–14.6 |
| IL-MC (52) | 268 × 235 | 188.3 / 191.4 / 194.4 | 1.76 | 0.91 / 0.84 / 0.55 | 6.6–11.8 |
| IL-PT (92) | 259 × 238 | 204.0 / 204.9 / 206.1 | 0.99 | 0.33 / 0.27 / 0.22 | 7.5–15.0 |
| IL-RD (248) | 275 × 241 | 200.7 / 202.7 / 205.0 | 1.14 | 0.44 / 0.34 / 0.24 | 6.9–15.3 |
| IL-RS (225) | 271 × 242 | 215.3 / 219.9 / 223.5 | 3.59 | 0.89 / 0.73 / 0.53 | 6.7–14.4 |
| IL-RT (245) | 245 × 269 | 197.1 / 198.0 / 198.7 | 0.71 | 0.24 / 0.19 / 0.16 | 7.2–16.1 |
| NE (144) | 226 × 256 | 352.3 / 353.4 / 355.1 | 0.80 | 0.76 / 0.56 / 0.24 | 6.5–16.7 |

Interquartile ranges of curvature and TPI at the locations:

| Field | Mean curvature, 30 / 90 / 270 m (1/km) | TPI, 50 / 150 / 450 m (m) |
|---|---|---|
| IL-BR | 0.448 / 0.438 / 0.210 | 0.273 / 1.516 / 2.011 |
| IL-DG | 0.158 / 0.089 / 0.014 | 0.064 / 0.166 / 0.221 |
| IL-MC | 0.268 / 0.189 / 0.114 | 0.142 / 0.871 / 1.094 |
| IL-PT | 0.178 / 0.107 / 0.020 | 0.076 / 0.231 / 0.573 |
| IL-RD | 0.230 / 0.132 / 0.027 | 0.100 / 0.314 / 0.469 |
| IL-RS | 0.451 / 0.207 / 0.062 | 0.153 / 0.530 / 1.078 |
| IL-RT | 0.181 / 0.082 / 0.019 | 0.069 / 0.158 / 0.428 |
| NE | 0.434 / 0.286 / 0.067 | 0.210 / 0.722 / 0.818 |

**Wetness (D6):** the fill, the floor of tan β, and the catchments that
the patch's edge cuts off.

| Field (locations) | Cells raised by the fill | Largest rise (m) | Locations on raised cells | tan β at the floor: cells / locations | Spearman, TWI with floors of 10⁻⁴ and 10⁻² | Locations whose upslope area reaches the edge |
|---|---|---|---|---|---|---|
| IL-BR (60) | 1.5 % | 1.06 | 0 | 0.6 % / 0.0 % | 0.858 | 0 |
| IL-DG (97) | 23.0 % | 0.75 | 21 | 20.1 % / 15.5 % | 0.897 | 0 |
| IL-MC (52) | 1.8 % | 0.83 | 0 | 0.8 % / 0.0 % | 0.966 | 39 |
| IL-PT (92) | 13.8 % | 0.63 | 16 | 10.1 % / 10.9 % | 0.930 | 0 |
| IL-RD (248) | 2.1 % | 1.16 | 3 | 1.7 % / 0.4 % | 0.901 | 182 |
| IL-RS (225) | 7.6 % | 1.53 | 7 | 4.5 % / 0.4 % | 0.980 | 33 |
| IL-RT (245) | 10.8 % | 1.26 | 42 | 9.2 % / 8.2 % | 0.888 | 26 |
| NE (144) | 28.0 % | 1.56 | 47 | 20.1 % / 16.0 % | 0.953 | 9 |

**Collinearity** at the locations, by Spearman's correlation; no stock is
involved. Pooled over all 1,163 locations, one pair exceeds 0.9 in absolute
value: mean curvature at 90 m and TPI at 50 m (+0.92), with the slopes at
30 and 90 m close behind (+0.89). Within fields, curvature at 90 m and TPI
at 50 m exceed it in five (IL-BR, IL-DG, IL-MC, IL-RT, NE); in NE also
curvature at 30 m and TPI at 50 m, TPI at 150 and at 450 m, and elevation
and TPI at 150 m; elevation and TPI at 450 m in IL-DG and IL-PT; and in
IL-MC the slopes at 30 and 90 m, and curvature at 270 m and TPI at 150 m.
TWI correlates negatively with every other covariate, most strongly with
TPI at 50 m (−0.67).

What the numbers say, beyond the rules:

- **The fields are nearly level.** At the locations the median slope at
  30 m is below 1° in every field and the steepest is 3.3°. IL-RS has the
  most relief (an interquartile range of 3.6 m in elevation), IL-DG the
  least (0.23 m).
- **On the flattest ground TWI rests partly on conventions.** In NE and
  IL-DG the fill raised a quarter of the patch's cells, and about a sixth
  of their locations sit where tan β is at its floor. How much the floor
  reorders TWI shows in the correlation between its values under floors of
  10⁻⁴ and 10⁻²: 0.86–0.98, lowest in IL-BR, IL-RT and IL-DG.
- **In IL-RD the TWI is mostly a lower bound**: for 182 of the 248
  locations the upslope area reaches the patch's edge, where D6 cuts the
  catchment off; in IL-MC, 39 of 52.
- **Curvature at 90 m and TPI at 50 m nearly duplicate each other** at
  these locations.
- **The 30 m curvature carries a trace of the resampling.** Seen in notebook
  03's maps as faint striping, and not among the items fixed for
  reporting. Bilinear interpolation smooths least where a patch cell's
  centre falls near a native cell's centre and most halfway between. In
  every field the 30 m curvature's magnitude rises with the share of a
  white noise's variance the interpolation keeps at the cell (Spearman
  +0.02 to +0.05), and its median is 9–25 % higher where that share is
  above 0.8 than where it is below 0.35. At 90 and 270 m the correlation
  lies within ±0.05, of either sign. Part of the 30 m curvature's variation
  is the resampling's, not the ground's.

Notebook 03 (`notebooks/03_terrain.py`) draws each field's hillshaded DEM
with its locations; field by field, the eleven covariates' grids and the
cells whose upslope area reaches the edge; and computes the resampling's
trace above. Its figures go to `notebooks/scratch/`.

Choices made while building, inside the decisions above. None of them
depends on the stocks:

- **Which items are versions.** A listing also returns items under other
  titles, such as "USGS 13 arc-second n40w089 1 x 1 degree"; only those
  titled "USGS 1/3 Arc Second <block> <date>" count (D1). The newest is the
  one with the latest publication date, then the latest date in its name.
- **The header** is the first 64 KiB of each version. In all five the image
  data begin within the first 7.4 KB, so the header holds every tag and the
  offset and size of every block.
- **One range per block.** The 512 × 512 blocks of these files lie 8 bytes
  apart (GDAL writes a leader and a trailer around each), so none adjoin,
  and each of the 24 blocks read was requested on its own.
- **A range response** must be HTTP 206 with a Content-Range naming the
  bytes asked for and the file's size, and must carry the same ETag as the
  version's other ranges; anything else is refused unread (E1).
- **Strict reads.** The transform reads a tile through a file object that
  holds only the stored ranges and raises on any read outside them.
- **The metadata record.** The listing's `vendorMetaUrl` is the FGDC XML
  for some items and an S3 browser page for others: the client takes the
  XML where the listing gives one, and otherwise ScienceBase's record as
  JSON (`metaUrl?format=json`). All five records are FGDC XML.
- **A tile's grid** comes from its GeoTIFF tags: tiepoint, pixel scale,
  PixelIsArea, NAD83 and the nodata value −999999. E3 checks it against
  GDAL's.
- **Each tile resampled on its own grid** (changed after the first run: see
  *If it fails*). At every patch cell centre, the newest version whose four
  native cells around the centre are all in its window gives the value. In
  IL-RS that is n41w089 north of a line about six native cells south of
  40° N, and n40w089 south of it; the field's `.npz` records which version
  gave each cell.
- **Edges.** A cell whose window (D4) or disc (D5) would reach past the
  patch is left empty; the 1,000 m buffer keeps every location clear of
  them. In the flow routing (D6) the patch's edge cells are outlets: the
  fill leaves them as they are, and flow that reaches them leaves the
  patch.
- **E2's tolerance for one lattice** is 10⁻³ of a cell, about 1 cm. The rule
  gives none; it was set after the first run found IL-RS's tiles 1.4 ×
  10⁻⁵ of a cell apart. For a field on two tiles, E2 counts the cells to
  spare on each window's sides except where the patch runs on into the
  neighbouring tile, and checks that some window covers every patch cell
  centre.
- **E3's plane** rises 1,000 m per degree of longitude and 2,000 m per
  degree of latitude, about 1.2 % and 1.8 %; it is laid on GDAL's cell
  centres and passed through the transform's own resampling.
- **What `curated/terrain/<site>.npz` holds:** the eleven covariates' grids
  (float32), the filled DEM (float64), the accumulated area, the area that
  comes from the patch's edge, tan β, the TWI under the two reported
  floors, which version gave each cell, the versions' URLs, and the grid's
  EPSG, corner and cell size.

`requirements.txt` now has tifffile, imagecodecs, rasterio and
scikit-image. `curated/` is not committed: the transform rebuilds it from
`raw/`, and this file records what it produced.

Stage 3 has passed. Three points carry forward to Stage 4's rules: where
TWI is a lower bound (IL-RD above all) and how much its floor reorders it;
the near-duplicate pair of curvature at 90 m and TPI at 50 m; and the
resampling's share in the 30 m curvature.

### If it fails

**08.11.2026: HTTP 403 from the build environment's proxy.** As in Stage 1,
the environment the repository is written in does not reach the source:
its egress proxy refuses `tnmaccess.nationalmap.gov`, where the client
starts. That is the environment, not the source. The pull ran on Windows;
the transform and the checkpoint read only `raw/` and ran in the build
environment.

**08.11.2026: two tiles 1.4 × 10⁻⁵ of a cell off one lattice.** The first
transform stopped at IL-RS: it placed n41w089's cells on n40w089's lattice
and required the two to share it exactly, and they do not. The tiles store
their cell size as 9.259259269 × 10⁻⁵ degrees, against 1/10800 =
9.259259259 × 10⁻⁵, and their tiepoints lie up to 8 × 10⁻¹⁰ degrees from
the nominal corners, so n41w089's first row lies 10,799.999986 rows north
of n40w089's, not 10,800: 0.14 mm on the ground. The transform then
accepted a lattice within 10⁻³ of a cell.

**08.11.2026: the plane caught the lattice.** The first run of the
checkpoint failed E3, 5/6: the plane came through 2.6 × 10⁻⁶ m off in
IL-RS, against 10⁻⁶ m. That is the lattice above: n41w089's values, moved
1.4 × 10⁻⁵ of a cell onto n40w089's lattice, sat that far from their place,
and the plane rises 2,000 m per degree of latitude (1.4 × 10⁻⁵ × 9.26 ×
10⁻⁵ degrees × 2,000 m per degree = 2.6 × 10⁻⁶ m). The transform now
resamples each tile on its own grid and takes the newest version wherever
its native cells surround a patch cell's centre, which is what D2 says,
and moves no tile's cells; the plane comes through within 1.5 × 10⁻¹¹ m,
and Checkpoint 3 passes, 6/6. The rule did not change. At IL-RS's
locations the covariates moved by at most 3.9 × 10⁻⁶ m in elevation,
1.3 × 10⁻⁵ degrees in slope and 5.6 × 10⁻⁴ in TWI; in the other fields by
rounding alone, 2.3 × 10⁻¹¹ m at most in elevation. E2's tolerance of
10⁻³ of a cell remains, as a check on the tiles alone.

**08.11.2026: the metadata records name no sources.** The rules expected
each tile's FGDC record to name the sources of its block, in a lineage
section, and the tile table to report them. None of the five has a
lineage section: each gives its period of content, a first and a last
date, and the tile table reports that instead (*content dated*). Which
source lies under a field is beyond what the records say, and Stage 3
reads nothing else about the sources.

**08.11.2026: three `DeprecationWarning`s from scikit-image under NumPy
2.5.** On Windows, under Python 3.14.8 with the newest libraries (NumPy
2.5.4, pandas 3.1.1, SciPy 1.16.3, scikit-image 0.26.0, tifffile
2026.10.15, imagecodecs 2026.10.10, rasterio 1.5.2), the 149 tests pass,
with three warnings. All three come from scikit-image's `reconstruction`,
which E5's check and one unit test call: at its end it sets an array's
`shape`, which NumPy 2.5 deprecates. That is scikit-image's code, not this
repository's, and it changes no result; 0.26.0, the newest release, still
has the line. The warnings are left visible rather than filtered. Should
NumPy remove the setter before scikit-image changes the line, E5's check
will fail, and the weekly CI run will show it.

### References for Stage 3

- Adhikari, K., Smith, D. R., Collins, H. P., et al. (2022). Mapping within-field soil health variations using apparent electrical conductivity, topography, and machine learning. *Agronomy*, 12(5), 1019. https://doi.org/10.3390/agronomy12051019
- Amatulli, G., Domisch, S., Tuanmu, M.-N., et al. (2018). A suite of global, cross-scale topographic variables for environmental and biodiversity modeling. *Scientific Data*, 5, 180040. https://doi.org/10.1038/sdata.2018.40
- Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. *Computers & Geosciences*, 62, 117–127. https://doi.org/10.1016/j.cageo.2013.04.024
- Behrens, T., Zhu, A.-X., Schmidt, K., et al. (2010). Multi-scale digital terrain analysis and feature selection for digital soil mapping. *Geoderma*, 155(3–4), 175–185. https://doi.org/10.1016/j.geoderma.2009.07.010
- Beven, K. J., & Kirkby, M. J. (1979). A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 24(1), 43–69. https://doi.org/10.1080/02626667909491834
- De Reu, J., Bourgeois, J., Bats, M., et al. (2013). Application of the topographic position index to heterogeneous landscapes. *Geomorphology*, 186, 39–49. https://doi.org/10.1016/j.geomorph.2012.12.015
- Freeman, T. G. (1991). Calculating catchment area with divergent flow based on a regular grid. *Computers & Geosciences*, 17(3), 413–422. https://doi.org/10.1016/0098-3004(91)90048-I
- Ilich, A. R., Misiuk, B., Lecours, V., et al. (2023). MultiscaleDTM: An open-source R package for multiscale geomorphometric analysis. *Transactions in GIS*, 27(4), 1164–1204. https://doi.org/10.1111/tgis.13067
- Kopecký, M., Macek, M., & Wild, J. (2021). Topographic Wetness Index calculation guidelines based on measured soil moisture and plant species composition. *Science of the Total Environment*, 757, 143785. https://doi.org/10.1016/j.scitotenv.2020.143785
- Maynard, J. J., & Johnson, M. G. (2014). Scale-dependency of LiDAR derived terrain attributes in quantitative soil-landscape modeling: Effects of grid resolution vs. neighborhood extent. *Geoderma*, 230–231, 29–40. https://doi.org/10.1016/j.geoderma.2014.03.021
- Minár, J., Evans, I. S., & Jenčo, M. (2020). A comprehensive system of definitions of land surface (topographic) curvatures, with implications for their application in geoscience modelling and prediction. *Earth-Science Reviews*, 211, 103414. https://doi.org/10.1016/j.earscirev.2020.103414
- Moore, I. D., Gessler, P. E., Nielsen, G. A., et al. (1993). Soil attribute prediction using terrain analysis. *Soil Science Society of America Journal*, 57(2), 443–452. https://doi.org/10.2136/sssaj1993.03615995005700020026x
- Mueller, T. G., & Pierce, F. J. (2003). Soil carbon maps: Enhancing spatial estimates with simple terrain attributes at multiple scales. *Soil Science Society of America Journal*, 67(1), 258–267. https://doi.org/10.2136/sssaj2003.2580
- Quinn, P., Beven, K., Chevallier, P., et al. (1991). The prediction of hillslope flow paths for distributed hydrological modelling using digital terrain models. *Hydrological Processes*, 5(1), 59–79. https://doi.org/10.1002/hyp.3360050106
- Riihimäki, H., Kemppinen, J., Kopecký, M., et al. (2021). Topographic Wetness Index as a proxy for soil moisture: The importance of flow-routing algorithm and grid resolution. *Water Resources Research*, 57(10), e2021WR029871. https://doi.org/10.1029/2021WR029871
- Shary, P. A., Sharaya, L. S., & Mitusov, A. V. (2002). Fundamental quantitative methods of land surface analysis. *Geoderma*, 107(1–2), 1–32. https://doi.org/10.1016/S0016-7061(01)00136-7
- Sørensen, R., Zinko, U., & Seibert, J. (2006). On the calculation of the topographic wetness index: evaluation of different methods based on field observations. *Hydrology and Earth System Sciences*, 10(1), 101–112. https://doi.org/10.5194/hess-10-101-2006
- U.S. Geological Survey. 1/3rd arc-second Digital Elevation Models (DEMs), USGS National Map 3DEP Downloadable Data Collection: the tiles, their updates and their historical versions, https://sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5; the versions, sizes, bounding boxes and metadata links as the TNM Access API lists them for the fields' extents (8 November 2026), https://tnmaccess.nationalmap.gov/api/v1/products
- USGS FAQs on the DEMs' datums and coverage, https://www.usgs.gov/faqs/what-projection-horizontal-datum-vertical-datum-and-resolution-a-usgs-digital-elevation-model; their sources, https://www.usgs.gov/faqs/what-data-sources-are-used-produce-3d-elevation-program-3dep-dems; and their conversion to Cloud Optimized GeoTIFF, https://www.usgs.gov/faqs/what-happened-gridfloat-arcgrid-and-img-formats-elevation-data-national-map
- Earth Engine's catalogue page for its copy of the collection, for the dates it covers: https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_10m_collection
