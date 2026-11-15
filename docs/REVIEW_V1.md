# Review of v1.0: an adversarial pre-publication review of Stages 1–5

15 November 2026 · reviews release v1.0 (commit `074a6b6`, with `6f0aa36`):
the README, the report (`docs/index.qmd`), `RUNBOOK.md` Stages 1–5, the
code behind the published numbers, CI and the dependencies, before Stage 6
is specified

**How this review was done, and its limits.** The reviewer is the same
assistant that helped write the code and the rules, so this is not an
independent review in the strict sense; it was asked to read the repository
as an external referee would, looking for what a sceptical one would attack.
It concentrated on what the published claims rest on: `src/transform/stocks.py`,
`src/transform/field_means.py`, the Stage 4 and 5 rules and results in the
RUNBOOK, the README, the report, the workflows and the requirements. It did
not audit all of the roughly 20,000 lines, and it re-ran no stage. Numbers
quoted below come from the RUNBOOK's recorded results, except those under M5,
which come from `raw/` on 15 November 2026. Every reference was checked
against its DOI record on that day.

**Status:** section 3's findings but the last, and the wording of m1, m3,
m5 and m6, adopted on 15 November 2026 (RUNBOOK, *Review of v1.0*); m2 is
listed among the report's limits. M5 addressed on 15 November 2026 (RUNBOOK,
Stage 1, *Addenda*): IL-BR's level is its carbon content's and agrees with
the USDA soil survey; the author's answer awaited. M1–M4 and m4 are to enter the RUNBOOK as
dated addenda with rules fixed before they run, as Stage 4's did; this file
records what was found, and the RUNBOOK records what was done about it.

## Verdict

The engineering is unusually strong. Rules are committed before the data
they judge, failures stay in the record, and the numerics are checked
against GSTools, R's gstat, the bordered kriging system and explicit
inverses to about 10⁻¹⁴. As a portfolio piece or a technical report, v1.0 is
publishable after the minor fixes below.

As a scientific claim it needs major revision. Three headline results rest on
choices that none of the checks test:

1. the mean model of each field is chosen on the same cross-validation
   scores that are then reported (M1)
2. the field means rest on a prior bounded where the correlation range is
   least identified, and their sensitivity to that bound is untested (M2)
3. the field-mean intervals are checked only against data simulated from the
   fitted model (M3), and the sample-size comparison changes the design and
   the estimator at once (M4)

One field, IL-BR, enters the headline range while its level is unexplained,
and its bulk densities suggest a data problem (M5).

## 1. Major findings

| # | Finding | Evidence | Proposed amendment |
|---|---|---|---|
| M1 | **Selection on the reported scores.** Stage 4's D5 selects, per field, the mean model ((a), (b) or (c)) with the lowest cross-validated CRPS, and the same cross-validation scores then carry "better CRPS than a quantile regression forest at seven fields of eight" and the calibration statements. Without nested selection the comparison favours kriging | Error estimates from the same cross-validation that selected a model are optimistically biased (Varma & Simon 2006). In the Stage 4 result: with model (a) fixed beforehand, kriging beats QRF at **5 of 8** fields, not 7 (QRF lower at IL-DG 4.301 < 4.377, IL-RD 6.453 < 7.038, IL-RS 6.702 < 7.112); and two of the seven wins under selection are within 0.4 % (IL-DG 4.286 against 4.301; IL-PT under (a), 4.142 against 4.153). (c)'s drift is already chosen inside each training fold (D4); the choice among (a), (b) and (c) is not | Select the mean model inside each training fold, by the training folds' own cross-validated CRPS, so the held-out scores judge the whole procedure, as D4 does for the drift; or report all three mean models against the forests and drop the selected column from comparisons. Report differences below about 1 % of CRPS as ties |
| M2 | **The range's upper bound may decide the field means.** D3's prior is uniform within Stage 2's bounds. At IL-RT, Stage 2's estimate of the range lies *at* the upper bound (2,401 m) and the posterior's 90 % interval reaches it (443–2,401 m); at IL-DG and IL-MC the estimates sit at their bounds too. A field mean depends most on the long-range behaviour that the bound truncates, and IL-RT is where the block-kriging interval is 46 % narrower than the sample mean's | A range beyond the extent of the data is weakly identified, and with flat priors on it the posterior can be improper without a bound (Berger et al. 2001), so the bound acts as information. The direction of its effect on a field mean is not obvious: with an unknown mean, longer ranges confound with the mean and can narrow the block variance as well as widen it. The RUNBOOK notes that at the small fields "the prior decides much of their intervals" (Stage 4, Result), but no rule tests the field means against the bound | A sensitivity in Stage 5's style: rerun the block kriging at IL-RT, IL-DG and IL-MC with the upper bound on the range doubled and tripled, the grid otherwise unchanged, and report how far each mean and the width of its interval move. If either moves by more than a stated amount, the README reports the field mean with that caveat |
| M3 | **The field-mean intervals are validated only against the model.** B3 simulates 400 fields *from the fitted model* at IL-MC and finds the right coverage (0.518, 0.898, 0.950): that shows the code is consistent with its model, not that the model is right for these fields. "No field's two estimates disagree beyond their uncertainty" compares two estimates from the same samples, which cannot validate either | Model-based inference is only as good as its model (Brus & de Gruijter 1997); a coverage test under the model's own simulations cannot detect a misspecified model. The narrower interval is a headline result (README, report) | An empirical check at the five grid fields (IL-DG, IL-PT, IL-RD, IL-RS, IL-RT): split each field's samples repeatedly into two interleaved half-grids (or thin the grid), block-krige the field mean from one part with the parameters' posterior refitted on it, and score that predictive against the other part's mean with its own sampling error added. Report coverage of the 95 % interval and the CRPS against the same split with the sample mean |
| M4 | **The sample-size comparison changes two things at once.** "13 to 52 samples on a grid against 25 to 103 by simple random sampling" compares a square grid *with block kriging* to simple random sampling *with the sample mean*, so the gain mixes the design with a model-based estimator. The grid figures also assume the current model is known for the future campaign: a 13-point grid could not re-estimate it, and at NE it would fit a six-coefficient quadratic mean from 13 points | Comparing designs under a model is done by the model-expected variance of each design with a stated estimator, separating the design's effect from the estimator's (Brus & de Gruijter 1997; Brus 2022). The RUNBOOK's table names the estimator for each column; the README and the report give the numbers without it | Add the missing case: the grid with the sample mean, its model-expected variance under the same posterior, so the table shows SRS + mean, grid + mean and grid + block kriging. State in the README and the report that the kriging figures assume the fitted model carries over to the next campaign |
| M5 | **IL-BR is probably a data problem, and it sets the headline range.** "21.8 to 65.7 Mg C/ha" begins with a field whose level the RUNBOOK calls unexplained (Stage 4, D9; Stage 5, D6). Above 30 cm, IL-BR's mean SOCc is 0.70 % (minimum 0.13 %), against 1.6–1.9 % elsewhere; and 12 of its 186 segments have a bulk density below 0.85 g/cm³, down to 0.58, against 3 of 1,634 segments in the other seven fields together. Bulk densities that low in a mineral soil this poor in carbon are implausible | `raw/` on 15 November 2026. Checkpoint 1's P6 checks only that BD lies in (0, 2.65] g/cm³. Stage 4's D9 found that Table 1 of Potash et al. (2023) gives each site's texture, but the RUNBOOK does not record IL-BR's | A plausibility rule beside P6, fixed before it runs: flag segments with BD below 0.9 g/cm³ and SOCc below 2 %, and report them per field. Record IL-BR's texture from Table 1; ask the dataset's author whether IL-BR's SOCc and BD are on the same basis as the other sites'. Until it is settled, the README gives the headline range without IL-BR, or marks IL-BR as provisional |

## 2. Moderate findings

| # | Finding | Proposed amendment |
|---|---|---|
| m1 | **The showcase is the best field.** The README's figure and its caption show IL-RT, whose modelling efficiency (MEC) of 0.57 is the highest of the eight; MEC runs from −0.04 to 0.57, and at IL-BR and IL-DG kriging scores within 2 % of the field's own distribution of stocks, so their maps are close to a constant | Say so under the figure and in **What Stage 4 found**: the range of MEC, and that two fields' maps add almost nothing to their mean |
| m2 | **Choices made on all the data are held fixed in the cross-validation.** Stage 2 chose each field's family, anisotropy and Box-Cox λ from all its stocks, and Stage 4's D1 holds them fixed inside every fold. The leak is small, since the choices are coarse, but it is one | List it under Stage 4's limits and the report's **Limits** |
| m3 | **The gstat gap is attributed without a cause.** The report calls the block variance's difference of 2 × 10⁻⁵ "a systematic gap inside gstat's block kriging", while the RUNBOOK says its cause inside gstat was not found. The tolerance was also widened tenfold after the failure; that is recorded honestly, and a referee will still note it | Call it an unexplained discrepancy in the report; keep the RUNBOOK's account. Optionally, isolate it: the same block from gstat's `vgm` covariance evaluated directly in R, without `krige` |
| m4 | **REML's own term has no outside check.** R1 checks the Gaussian likelihood against scikit-learn's, which has no REML; the REML criterion is checked against a second implementation by the same author (R1, K2) | Compare the REML criterion and estimates with an independent implementation, for example `nlme::gls(method = "REML")` with a spatial correlation structure, or geoR's `likfit(lik.method = "REML")`, in the R job that G1 already runs |
| m5 | **The forest is one forest, not machine learning.** QRF is trained across the fields on coordinates and terrain only, without the covariates Potash et al. used (soil maps, Landsat), and untuned by design (D7). That is a fair stated choice, and "kriging beats machine learning" does not follow from it | Phrase the result as "beats this forest" in the README and the report |
| m6 | **Fixed depth for monitoring.** The detectable change is computed for fixed-depth stocks, whereas change in soil carbon is better compared on an equivalent soil mass (Wendt & Hauser 2013; von Haden et al. 2020). The report's Limits say so; the figure and its summary do not | Add the caveat to the caption of `detectable_change.png` and to the README's Stage 5 bullet |

## 3. Minor findings and presentation

- The README's first paragraph still says "*Ordinary* kriging", and reads as
  if the BZE-LW analysis were done. Stage 4 is Bayesian universal kriging,
  and BZE-LW is pulled and checked, not modelled.
- `src/report/figures.py` imports **matplotlib** and
  `notebooks/01_data_checks.py` imports **geopandas**; neither is in
  `requirements.txt` or `requirements-lock.txt`, and CI runs neither, so the
  published figures cannot be rebuilt from the documented environment.
  *Corrected 15 November 2026: both are declared in `requirements-figures.txt`,
  committed with Stage 1, which the notebooks and the RUNBOOK name; the
  review missed the file. What stands is that the lock file did not pin
  them and CI never ran `figures.py`.*
- The report says the block-kriging interval is "up to 46 % narrower", and
  leaves out that it is 2 % wider at IL-BR, which the RUNBOOK records.
- **What is real** in the README is one sentence of about 350 words. Bullets
  with a number each would be read; the paragraph will not be.
- CRPS, PICP and MSDR are given without their spread over the three repeats.
  The minimum and maximum over the repeats are cheap and show how stable
  each figure is.

## 4. What holds up

- Rules committed before the data they judge, and failures kept in the
  record with their amendments; the commit history is the evidence.
- Independent numerical agreement: GSTools (Stage 2's moment estimates and
  Stage 4's kriging), R's gstat (Stage 5), bordered systems and explicit
  inverses, each to the stated tolerance.
- Proper scores of the whole predictive distribution (Gneiting & Raftery
  2007), with PICP, QCP and MSDR beside them and a baseline without location.
- Provenance: a SHA-256 manifest of every response, pinned versions, data
  not redistributed, licences stated.

## 5. Order of the amendments

1. Wording, under an hour: m1, m3, m5, the README's first paragraph and the
   IL-BR caveat in the report.
2. `matplotlib` and `geopandas` into the requirements and the lock file, and
   a CI step that runs `src/report/figures.py` on its committed tables.
3. M5, IL-BR: the plausibility rule, Table 1, the question to the author.
4. M1 and M2, as a Stage 4 and 5 addendum with rules fixed first.
5. M3 and M4, as a Stage 5 addendum.

M1 to M3 bear on Stage 6 as well: the national model will face the same
selection, prior-bound and calibration questions, and settling them on eight
fields first is cheaper.

## References

- Berger, J. O., De Oliveira, V., & Sansó, B. (2001). Objective Bayesian analysis of spatially correlated data. *Journal of the American Statistical Association*, 96(456), 1361–1374. https://doi.org/10.1198/016214501753382282
- Brus, D. J. (2022). *Spatial Sampling with R*. Chapman and Hall/CRC. https://doi.org/10.1201/9781003258940
- Brus, D. J., & de Gruijter, J. J. (1997). Random sampling or geostatistical modelling? Choosing between design-based and model-based sampling strategies for soil (with discussion). *Geoderma*, 80(1–2), 1–44. https://doi.org/10.1016/s0016-7061(97)00072-4
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. https://doi.org/10.1198/016214506000001437
- Potash, E., Guan, K., Margenot, A. J., et al. (2023). Multi-site evaluation of stratified and balanced sampling of soil organic carbon stocks in agricultural fields. *Geoderma*, 438, 116587. https://doi.org/10.1016/j.geoderma.2023.116587
- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91. https://doi.org/10.1186/1471-2105-7-91
- von Haden, A. C., Yang, W. H., & DeLucia, E. H. (2020). Soils' dirty little secret: depth-based comparisons can be inadequate for quantifying changes in soil organic carbon and other mineral soil properties. *Global Change Biology*, 26(7), 3759–3770. https://doi.org/10.1111/gcb.15124
- Wendt, J. W., & Hauser, S. (2013). An equivalent soil mass procedure for monitoring soil organic carbon in multiple soil layers. *European Journal of Soil Science*, 64(1), 58–65. https://doi.org/10.1111/ejss.12002
