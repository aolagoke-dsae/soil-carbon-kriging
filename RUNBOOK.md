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
| 2 | Potash: stocks in 0–30 cm; covariance models by REML, with the pooling of small fields tested; moment variograms as the gstat baseline | **in build**: rules fixed 05.11.2026; the stock table and the REML code pass S1, S2, R1 and R2 (07.11.2026); R3, R4 and V1–V3 not built |
| 3 | Terrain covariates for the eight fields: USGS 3DEP 10 m elevation and its derivatives, extracted through Earth Engine, the values committed | not yet specified |
| 4 | Kriging per field with the Stage 2 models: (a) against (b), the parameter uncertainty propagated, cross-validation and interval coverage (PICP, QCP); and a machine-learning benchmark against it: a quantile regression forest, and a random forest with kriged residuals, on coordinates and terrain | not yet specified |
| 5 | Each field's mean stock: block kriging against the stratified, design-based estimate the paper evaluates | not yet specified |
| 6 | BZE-LW: a quantile regression forest with kriged residuals at national scale, its covariates averaged within 4 km of each published point | not yet specified |

Stages 3–6 are a direction, not instructions. Each gets its rules, written
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

### Built so far

**07.11.2026: the stock table and the REML code; S1, S2, R1 and R2 pass.**
R3, R4 and V1–V3 are not built, so Checkpoint 2 exits 3 (no rule failed,
some not built) and Stage 2 has not passed. Two of the four commands exist:

```bash
python -m src.transform.stocks       # curated/potash_stocks_0_30.parquet
python -m src.checks.checkpoint2     # exit 0 passed, 1 a rule failed, 2 inputs missing, 3 not all built
```

`requirements.txt` now has pyproj, pyarrow and scikit-learn; scikit-gstat
comes with V2 and V3. `curated/` is not committed: the transforms rebuild it
from `raw/`, and this file records what they produced.

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
- **Bounds**, against which R3 will judge the fits: the effective range from
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

Run on Linux (Python 3.10.12) in 60 seconds; 80 tests pass, 47 of them from
Stage 1.

```text
Checkpoint 2, run 2026-11-07T18:51:27Z

PASS  S1 stocks: 1163 stocks, one per (site, location_id), none of the seven incomplete locations; all finite, 13.2 to 117.1 Mg C/ha
PASS  S2 arithmetic: largest difference from the loop-based recomputation 1.4e-14 Mg C/ha (rule: 1e-09); the hand-worked example is in src/transform/test_stocks.py
PASS  R1 likelihood: Gaussian log-likelihood against scikit-learn's GaussianProcessRegressor, largest relative difference 6.2e-13 over 128 parameter sets; REML criterion against its formula with explicit inverses, 6.5e-13 over 384, both mean models and all three families (rule: 1e-08)
PASS  R2 recovery: medians over 50 fields of 400 points: effective range 151.3 m (+0.8%; quartiles 126 to 178), nugget-to-sill 0.296 (-1.3%; 0.255 to 0.346), sill 99.4 (-0.6%; 91.5 to 108); 50/50 fits converged (rule: each median within 10%)
----  R3 fits: not built yet
----  R4 record: not built yet
----  V1 bins: not built yet
----  V2 independent estimate: not built yet
----  V3 baseline fits: not built yet

Stocks in 0-30 cm per field, Mg C/ha (reported, not judged; the Box-Cox profile of D3 comes with the fits):
 site  stocks mean   sd cv (%) skewness ± se trend R2
IL-BR      60 21.8  3.8   17.2  +0.19 ± 0.31     0.20
IL-DG      97 61.0  8.2   13.5  +0.26 ± 0.24     0.09
IL-MC      52 50.9  6.6   13.1  +0.24 ± 0.33     0.31
IL-PT      92 61.5  7.4   12.0  +0.21 ± 0.25     0.17
IL-RD     248 65.4 14.2   21.7  +0.20 ± 0.15     0.08
IL-RS     225 64.0 14.7   23.0  -0.18 ± 0.16     0.14
IL-RT     245 61.7 17.4   28.2  +0.18 ± 0.16     0.44
   NE     144 64.4  7.4   11.5  +0.33 ± 0.20     0.31

Checkpoint 2 NOT COMPLETE: 4/4 built rules passed, 5 not built yet
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
  largest is NE's, +0.33 ± 0.20. The Box-Cox profiles of D3 come with the
  fits.
- The second-order trend explains 8–44 % of the variance: 44 % at IL-RT,
  31 % at NE and IL-MC.

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
