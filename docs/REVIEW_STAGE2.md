# Review of the Stage 2 draft: methods against current practice

5 November 2026 · reviews the Stage 2 draft in `RUNBOOK.md` (decisions D1–D9,
Checkpoint 2) before it is committed

**How this review was done, and its limits.** The reviewer also wrote the
draft, so this is not an independent review in the strict sense. To make up
for that, each finding rests on literature retrieved for the purpose, not on
the draft's own reasoning; a claim marked *(title)* rests on the paper's title
and record only, and one marked *(via X)* on how another paper reports it.
Nothing below has looked at a variogram: the rules were still unfixed, and no
finding depends on the data beyond what Checkpoint 1 reported.

**Adopted on 5 November 2026, before the rules were fixed:** M1–M4 and the
tested pooling (RUNBOOK, Stage 2: D3, D5–D7), m1 and m2 as reported
diagnostics, m3–m5 as notes and limits, and ML1–ML3 as Stages 3, 4 and 6.
Stage numbers below are those of the RUNBOOK after that change; decision
numbers (D1–D9) are the draft's, which the fixed rules renumber.

## Verdict

The draft reproduces the author's gstat workflow faithfully, and its
Checkpoint 2 tests the code well. As a statement of current practice it falls
short in four places, all about how the variogram is estimated and how far it
is trusted:

1. the models that Stage 4 krigs with are fitted by weighted least squares to
   method-of-moments variograms, where REML is the established better choice,
   above all for the trend model (b) and for the small fields
2. the model set and the selection rule are narrower than current practice:
   no Matérn, and a weighted sum of squares instead of an information criterion
3. the uncertainty of the variogram itself is not estimated, so Stage 4's
   kriging variances would treat fitted parameters as known
4. the transform rule is decided by a noisy statistic, and a log scale works
   against the quantity soil-carbon accounting needs, the field mean

Machine-learning methods were reviewed as an alternative (section 3). They do
not replace geostatistics for these fields, which have no covariates and 52 to
249 locations each. They belong in a benchmark at Stage 4 and in the national
BZE-LW model at Stage 6.

## 1. Major findings

| # | Finding | Evidence | Proposed amendment |
|---|---|---|---|
| M1 | **REML, not least squares on moment variograms, for the models Stage 4 krigs with.** Method-of-moments variograms of residuals from a fitted trend are biased, so D4(b) as drafted would carry that bias into second-order polynomial kriging. Small fields fare better under REML too | Lark, Cullis & Welham (2006): REML estimates are consistent and less biased than ML or moment estimates from trend residuals; moment methods are "seriously biased" when the fixed effects are more than a constant mean; REML-E-BLUP is the state of the art. Alves et al. (2009, 67 points) found REML variograms generally better by kriging error; Ferraz et al. (2011) found REML satisfactory with a small sample; both follow Kerry & Oliver (2007). Marchant et al. (2013) recommend REML or declustering for clustered data, which IL-BR, IL-MC and NE are | Fit each field's models by REML, with a constant mean (a) and with the second-order trend (b). Keep the moment variogram with gstat's weights as the R-comparable baseline and for the figures (D5, D7, V1–V3 unchanged). REML is implemented with NumPy and SciPy, since at 52–249 points the exact likelihood is cheap, and tested by recovering known parameters from simulated fields |
| M2 | **A wider model set, chosen by an information criterion.** Spherical against exponential by weighted sum of squares neither penalises parameters nor covers the smoothness of the process | Webster & McBratney (1989) introduced the AIC for choosing soil variogram models (for least-squares fits, A = n ln R + 2p). Minasny & McBratney (2005) propose the Matérn function as a general model for soil variograms *(title)*; it contains the exponential (ν = ½) and tends to the Gaussian as ν grows | Exponential, spherical and Matérn (ν fitted, bounded), each with a nugget, chosen by AIC under REML. REML likelihoods are comparable only between models with the same fixed effects, so (a) against (b) stays a Stage 4 cross-validation question, as drafted |
| M3 | **The variogram's own uncertainty.** Kriging variances computed from fitted parameters treat them as known. That matters most where the data cannot see the short range: at the five grid fields no two locations are closer than 20 m, so the nugget there is extrapolated | Marchant & Lark (2004) set out how to estimate variogram uncertainty *(title)*. Cui & Stein (1995) show that kriging which carries uncertain parameters as priors was more precise than ordinary kriging below about 60 observations, and that ordinary kriging won above about 140 | Report a profile-likelihood 95 % interval for each field's range and nugget ratio. Stage 4 then propagates it, for example by kriging with a parametric-bootstrap sample of parameter sets, and reports how much of each field's prediction variance the nugget's interval moves |
| M4 | **The transform.** Sample skewness has a standard error of about √(6/n): ±0.34 at IL-MC's 52 points. A rule switching at 1 can flip on noise, and per-field switches break the pooled model's comparability. A log scale also turns the field mean, the quantity soil-carbon accounting reports, into a back-transform problem | Lognormal block kriging needs its own predictor (Cressie 2006) *(title)*; on the original scale the field mean is a linear functional, which kriging estimates without back-transform bias | Fit on the original scale (Mg C/ha). Report each field's skewness with its standard error and a REML profile of the Box-Cox λ. A log-scale model becomes a Stage 4 sensitivity only for a field whose 95 % interval for λ excludes 1 |

**D8, pooling, follows from M1.** Pooling standardised variograms assumes the
fields share a correlation structure. With REML that assumption can be tested
instead of assumed. Fit the eight fields jointly with shared correlation
parameters (range, nugget ratio, ν) and field-specific means and variances, and
compare that with field-specific parameters by AIC, first on the four large
fields, which can carry their own. The four small fields take the shared
parameters only if the comparison supports sharing; otherwise they keep their
own REML fits, whose intervals (M3) will show how little 52–97 points fix.
Borrowing strength across comparable areas is what Cui & Stein (1995) found
pays below about 60 observations.

## 2. Moderate and minor findings

| # | Finding | Proposed amendment |
|---|---|---|
| m1 | **Anisotropy is not looked at.** Tillage and landscape position can make the range depend on direction | Report directional variograms at 0, 45, 90 and 135° and a REML fit with geometric anisotropy; adopt it in Stage 4 only where it lowers the AIC by more than 2 (fixed now) |
| m2 | **Matheron's estimator is sensitive to outliers** | Report the Cressie–Hawkins estimator beside it; robust estimators for soil survey are compared by Lark (2000) *(title)*. No data are removed |
| m3 | **Measurement error sits inside the nugget** | Laboratory error inflates the nugget and the sill; modelling it gives more realistic uncertainty (Takoutsing, Heuvelink & Stoorvogel 2022). If the Geoderma paper reports replicate precision, use it as a lower bound on the nugget; otherwise record it as a limit |
| m4 | **Fixed depth against equivalent soil mass** | Fixed depth is right for a single survey. For change over time, equivalent soil mass is the recommended standard (von Haden et al. 2020; Wendt & Hauser 2013). D1 should say why fixed depth is used, so that a later resampling stage does not inherit it |
| m5 | **Stage 4 validation and the field mean** | Write Stage 4's rules knowing that spatial cross-validation can mislead about map accuracy (Wadoux et al. 2021) *(title)*, and that the field mean can be estimated design-based or model-based (Brus & de Gruijter 1997) *(title)*; Stage 5 already plans the comparison. Validate prediction intervals with PICP and QCP plots (Szatmári & Pásztor 2019, *via* Szatmári et al. 2026) |

## 3. Machine learning: does a more capable tool fit these data better?

**Not as the core model for the Potash fields; yes as a benchmark, and as the
core at national scale.**

- **These fields have no covariates yet, and 52–249 points each.** Hengl et
  al. (2018), who introduced random forest for spatial prediction, conclude
  that with few points and covariates and near-linear relationships,
  model-based geostatistics can still predict more accurately than their
  forest. Without covariates a random forest can use only coordinates or
  distances.
- **A forest's uncertainty comes from its covariates.** The uncertainty a
  quantile regression forest gives depends mostly on its environmental
  covariates (Szatmári & Pásztor 2019, *via* Szatmári et al. 2026), and within
  a single field the soil-forming factors vary little, so dense covariates
  need large samples to calibrate (Taghizadeh-Mehrjardi et al. 2025).
- **The field mean needs spatial correlation.** Aggregating predictions over
  a field reduces the uncertainty, which is what the voluntary carbon market
  needs to show a significant change (Szatmári et al. 2026). That aggregation
  needs the spatial covariance: in a test by Kangas et al. (2023), kriging
  captured pixel-level uncertainty and random-forest ensembles slightly
  underestimated it, and simulation was needed for area-level quantities.
- **Where ML does better.** At landscape scale with covariates, regression
  kriging gave lower errors but quantified uncertainty worse than random forest
  (Takoutsing & Heuvelink 2022, *via* Zhang et al. 2023), and kriging of
  random-forest residuals commonly beat machine learning alone (*via* Szatmári
  et al. 2026). That is the BZE-LW situation: 3,104 sites and national
  covariates. Hyperparameter tuning must then respect spatial autocorrelation
  (Wadoux, Minasny & McBratney 2020).
- **The "ML" model that fits these fields is the Gaussian process.** Gaussian
  process regression is kriging under another name; scikit-learn's
  `GaussianProcessRegressor` with a Matérn kernel plus a white-noise term fits
  essentially the same model by maximum likelihood, with the mean handled by
  centring rather than estimated. Using it as an independent check of the
  REML fits (M1) shows fluency in machine-learning tools without changing the
  model.

**Proposed:**

- **ML1** A Stage 4 benchmark. Quantile regression forest, and random forest
  with kriged residuals, against REML kriging, under the same cross-validation
  and judged on the same PICP and QCP plots. The forests use coordinates and,
  once pulled, terrain.
- **ML2** A covariate stage before it (Stage 3): USGS 3DEP 10 m elevation and its
  derivatives for the eight fields, through Earth Engine. This also brings
  Earth Engine into the Potash case.
- **ML3** For BZE-LW (Stage 6), quantile regression forest plus residual
  kriging is the primary model, with covariates averaged within 4 km of each
  published point.

## 4. What stands as drafted

D2 (seven incomplete locations left out), D5 (bins centred on the grid
spacing, 30 pairs) and D7 (gstat's weights) for the moment baseline, D9
(scope), S1–S2 and V1–V4. Checkpoint 2 gains rules for the REML code: parameter
recovery on simulated fields, and the log-likelihood computed by two code paths.

## References

- Alves, M. de C., da Silva, F. M., Moraes, J. C., et al. (2009). Geostatistical analysis of the spatial variation of the berry borer and leaf miner in a coffee agroecosystem. *Precision Agriculture*, 12(1), 18–31. https://doi.org/10.1007/s11119-009-9151-z
- Brus, D. J., & de Gruijter, J. J. (1997). Random sampling or geostatistical modelling? Choosing between design-based and model-based sampling strategies for soil (with discussion). *Geoderma*, 80(1–2), 1–44. https://doi.org/10.1016/s0016-7061(97)00072-4
- Cressie, N. (2006). Block kriging for lognormal spatial processes. *Mathematical Geology*, 38(4), 413–443. https://doi.org/10.1007/s11004-005-9022-8
- Cui, H., & Stein, A. (1995). Extension of spatial information, Bayesian kriging and updating of prior variogram parameters. *Environmetrics*, 6(4), 373–384. https://doi.org/10.1002/env.3170060406
- Ferraz, G. A. e S., da Silva, F. M., Alves, M. de C., et al. (2011). Geostatistical analysis of fruit yield and detachment force in coffee. *Precision Agriculture*, 13(1), 76–89. https://doi.org/10.1007/s11119-011-9223-8
- Hengl, T., Nussbaum, M., Wright, M. N., Heuvelink, G. B. M., & Gräler, B. (2018). Random forest as a generic framework for predictive modeling of spatial and spatio-temporal variables. *PeerJ*, 6, e5518. https://doi.org/10.7717/peerj.5518
- Kangas, A. S., Myllymäki, M., Mehtätalo, L., et al. (2023). Understanding uncertainty in forest resources maps. *Silva Fennica*. https://doi.org/10.14214/sf.22026
- Kerry, R., & Oliver, M. A. (2007). Comparing sampling needs for variograms of soil properties computed by the method of moments and residual maximum likelihood. *Geoderma*, 140(4), 383–396. https://doi.org/10.1016/j.geoderma.2007.04.019
- Lark, R. M. (2000). A comparison of some robust estimators of the variogram for use in soil survey. *European Journal of Soil Science*, 51(1), 137–157. https://doi.org/10.1046/j.1365-2389.2000.00280.x
- Lark, R. M., Cullis, B. R., & Welham, S. J. (2006). On spatial prediction of soil properties in the presence of a spatial trend: the empirical best linear unbiased predictor (E-BLUP) with REML. *European Journal of Soil Science*, 57(6), 787–799. https://doi.org/10.1111/j.1365-2389.2005.00768.x
- Marchant, B. P., & Lark, R. M. (2004). Estimating variogram uncertainty. *Mathematical Geology*, 36(8), 867–898. https://doi.org/10.1023/b:matg.0000048797.08986.a7
- Marchant, B. P., Viscarra Rossel, R. A., & Webster, R. (2013). Fluctuations in method-of-moments variograms caused by clustered sampling and their elimination by declustering and residual maximum likelihood estimation. *European Journal of Soil Science*, 64(4), 401–409. https://doi.org/10.1111/ejss.12029
- Minasny, B., & McBratney, A. B. (2005). The Matérn function as a general model for soil variograms. *Geoderma*, 128(3–4), 192–207. https://doi.org/10.1016/j.geoderma.2005.04.003
- Szatmári, G., & Pásztor, L. (2019). Comparison of various uncertainty modelling approaches based on geostatistics and machine learning algorithms. *Geoderma*, 337, 1329–1340. https://doi.org/10.1016/j.geoderma.2018.09.008
- Szatmári, G., Laborczi, A., Takács, K., et al. (2026). HU-SoilCarbonGrids: an initiative for providing comprehensive information on soil organic carbon changes in Hungary. *European Journal of Soil Science*, 77(1). https://doi.org/10.1111/ejss.70292
- Taghizadeh-Mehrjardi, R., Kerry, R., Ingram, B., et al. (2025). Mapping soil volumetric water content at multiple depths for variable rate irrigation using UAV and yield monitor data with random forests. *Soil Use and Management*, 41(4). https://doi.org/10.1111/sum.70156
- Takoutsing, B., & Heuvelink, G. B. M. (2022). Comparing the prediction performance, uncertainty quantification and extrapolation potential of regression kriging and random forest while accounting for soil measurement errors. *Geoderma*, 428, 116192. https://doi.org/10.1016/j.geoderma.2022.116192
- Takoutsing, B., Heuvelink, G. B. M., & Stoorvogel, J. J. (2022). Accounting for analytical and proximal soil sensing errors in digital soil mapping. *European Journal of Soil Science*, 73(2). https://doi.org/10.1111/ejss.13226
- von Haden, A. C., Yang, W. H., & DeLucia, E. H. (2020). Soils' dirty little secret: depth-based comparisons can be inadequate for quantifying changes in soil organic carbon and other mineral soil properties. *Global Change Biology*, 26(7), 3759–3770. https://doi.org/10.1111/gcb.15124
- Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S., & Brus, D. J. (2021). Spatial cross-validation is not the right way to evaluate map accuracy. *Ecological Modelling*, 457, 109692. https://doi.org/10.1016/j.ecolmodel.2021.109692
- Wadoux, A. M. J.-C., Minasny, B., & McBratney, A. B. (2020). Machine learning for digital soil mapping: applications, challenges and suggested solutions. *Earth-Science Reviews*, 210, 103359. https://doi.org/10.1016/j.earscirev.2020.103359
- Webster, R., & McBratney, A. B. (1989). On the Akaike Information Criterion for choosing models for variograms of soil properties. *Journal of Soil Science*, 40(3), 493–496. https://doi.org/10.1111/j.1365-2389.1989.tb01291.x
- Wendt, J. W., & Hauser, S. (2013). An equivalent soil mass procedure for monitoring soil organic carbon in multiple soil layers. *European Journal of Soil Science*, 64(1), 58–65. https://doi.org/10.1111/ejss.12002
- Zhang, L., Wu, Z., Sun, X., et al. (2023). Mapping topsoil total nitrogen using random forest and modified regression kriging in agricultural areas of central China. *Plants*, 12(7), 1464. https://doi.org/10.3390/plants12071464
