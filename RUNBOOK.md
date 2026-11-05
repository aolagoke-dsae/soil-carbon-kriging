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
| 1 | Pull both datasets through their clients; Checkpoint 1 | **written** 05.11.2026 (clients, rules, tests green on fixtures); **not run** |
| 2 | Potash: depth segments harmonised to stocks in 0–30 cm; experimental variograms per field | not yet specified |
| 3 | Ordinary kriging per field with the kriging variance; spatially blocked cross-validation and interval coverage | not yet specified |
| 4 | Each field's mean stock: block kriging against the stratified, design-based estimate the paper evaluates | not yet specified |
| 5 | BZE-LW: a national variogram under the 4 km offset; covariates averaged within 4 km, through Earth Engine | not yet specified |

Stages 2–5 are a direction, not instructions. Each gets its rules, written
before its data is touched, once the stage before it has passed.

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
  `config/variogram_sample_sizes.csv`; segments; extent; median spacing; how
  many locations reach 30 cm; the deepest segment
- BZE-LW: `kv_0_30` quantiles and medians by land use, sites by land use, the
  peat flag `bze_moor`, what `kv_30_100` (stored as text) holds, the sampling
  years, and the nearest-neighbour distances of the published points

**Known, and not testable:** the BZE-LW coordinates are not the sampling
sites. For data protection each was generated at random within 4 km of the
originally planned point (the `xcoord`/`ycoord` description in the Atlas
record; `config/sources.csv`). Every later stage carries that offset.

### Result

Not run. The 45 tests pass on fixtures (05.11.2026, Python 3.10.12 on Linux);
no data has been pulled.

### If it fails

**05.11.2026: HTTP 403 from the build environment's proxy.** The environment
the repository was written in reaches neither `api.figshare.com` nor
`atlas.thuenen.de`: its egress proxy refuses both hosts (`CONNECT tunnel
failed, response 403`). That is the environment, not the sources; both answer
from an ordinary connection. Run the two pulls where they are reachable.
