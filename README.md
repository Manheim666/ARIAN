# arian-wildfire-prediction

> **ARIAN — Azerbaijani Risk Intelligence and Analysis Network**
> Production-grade wildfire risk and weather forecasting system for 16 Azerbaijani cities.
> 30-day probabilistic forecasts · Calibrated fire probability · Climate anomaly detection · REST API

---

## Problem Statement

Azerbaijan sits at the intersection of three distinct climate zones — semi-arid lowlands, a humid subtropical Caspian coast, and alpine Caucasus highlands — making it both ecologically significant and disproportionately fire-prone. Temperatures have risen **+0.4 °C/decade** since 1980 while summer precipitation declines, dramatically increasing wildfire risk across the Greater and Lesser Caucasus (≈1.2 M ha of commercial forest). The Baku–Absheron peninsula and the Kura-Araz lowlands experience extreme summer drought annually, while highland forests face prolonged dry spells with no natural firebreak management. Despite these conditions, **no regional 30-day fire-risk forecast exists for Azerbaijan**.

ARIAN addresses this gap by integrating satellite imagery, multi-source meteorological archives, and the Azerbaijani State Forest Management Inventory (MESE QURULUSU) into a unified ML pipeline that delivers daily wildfire probability scores, expected fire counts, and multi-target 30-day weather forecasts across 16 Azerbaijani cities.

---

## Why It Matters

Wildfire incidents have increased measurably since 2010. The 2021–2022 fire seasons alone caused substantial forest loss across the Greater and Lesser Caucasus, with the national forest estate (≈1.2 M hectares) at sustained risk. Early-warning forecasts give emergency services **7–14 days of operational lead time** to pre-position firefighting resources and issue timely civilian evacuation orders, directly reducing the estimated **~$70–80 M/year** in economic losses from extreme weather events in Azerbaijan.

| Problem | ARIAN Solution |
|---------|----------------|
| No regional-scale 30-day weather forecast | ML ensemble trained on 6+ years of hourly Open-Meteo archive data |
| Wildfire risk assessments are reactive, not predictive | Calibrated 30-day forward fire probability using fire-science features (FWI, drought) |
| No early-warning system for high-risk fire windows | Canadian FWI system + SPEI drought indices provide 7–14 day structural lead time |
| Climate trend reports require GIS expertise | Automated Mann-Kendall + Theil-Sen trend analysis per city, output to CSV |
| Operational data scattered across agencies | Single REST API (FastAPI) aggregates weather, wildfire risk, and climate insights |
| No structured forest inventory for fire risk zoning | MESE QURULUSU provides stand-level species, crown density, and age class per polygon |

---

## Target

| Variable | Type | Definition |
|----------|------|------------|
| **`fire_occurred`** | Binary (0/1) | 1 if ≥1 NASA FIRMS VIIRS-SNPP hotspot falls within 50 km of city centroid on forecast date. Label window: next-day 00:00–23:59 local time. |
| **`fire_count`** | Non-negative integer | Expected number of FIRMS hotspots in the same 50 km buffer; modelled with Poisson regression to enforce non-negative predictions. |

**Label construction:** VIIRS-SNPP hotspots (~375 m resolution) are spatially joined to per-city 50 km circular buffers. Each (city, date) pair receives a binary fire label and a raw hotspot count. Both targets are computed before any feature engineering to prevent leakage.

---

## Features

### Core Features (Available Now — Collected Data)

| Source | Name | Units | Aggregation |
|--------|------|-------|-------------|
| Open-Meteo | `temperature_2m` | °C | daily mean; lags 1 / 3 / 7 / 14 d |
| Open-Meteo | `wind_speed_10m` | m/s | daily mean; lags 1 / 3 / 7 / 14 d |
| Open-Meteo | `relative_humidity_2m` | % | daily mean; lags 1 / 3 / 7 / 14 d |
| Open-Meteo | `precipitation_7d_sum` | mm | rolling 7-day sum |
| Open-Meteo | `rain_30d_sum` | mm | rolling 30-day sum |
| Open-Meteo | `heatwave_flag` | binary | 3+ consecutive days above local p90 |
| FWI (computed) | `fwi`, `ffmc`, `dc`, `dsr` | — | daily value derived from Open-Meteo inputs |
| SPEI (computed) | `spei_3`, `spi_1` | σ | 3-month / 1-month standardized anomaly |
| ESA WorldCover | `land_cover_class` | class | dominant land cover class in 50 km buffer |
| AZ Forest Boundary | `forest_fraction` | fraction | forest area / buffer area (from Azerbaycan.kmz) |
| MESE QURULUSU | `dominant_species` | class | modal tree species by stand area within buffer |
| MESE QURULUSU | `mean_crown_density` | 0.1–1.0 | area-weighted canopy closure |
| MESE QURULUSU | `pct_old_growth` | fraction | stands with age class ≥ 80 yr / total area |
| WorldPop + OSM | `human_activity_score` | — | population_density × road_density_km_per_km² |
| NASA FIRMS | `days_since_last_fire` | days | days elapsed since last hotspot in city buffer |
| NASA FIRMS | `fire_count_30d` | count | FIRMS hotspot count in prior 30-day window |

### Extended Features (Pending Data Acquisition)

| Source | Name | Notes |
|--------|------|-------|
| ERA5-Land | `sm_surface_m3m3`, `sm_rootzone_m3m3` | Copernicus CDS registration required |
| MODIS MOD13Q1 | `ndvi`, `evi`, `ndwi`, `ndwi_anomaly` | Live fuel moisture proxy; NASA Earthdata / AppEEARS |
| SRTM DEM | `elevation_m`, `slope_deg`, `northness`, `tpi` | Terrain features; OpenTopography one-time download |
| SMAP L4 | `sm_rootzone_m3m3` | Daily rootzone moisture; NASA Earthdata pipeline |
| Blitzortung | `lightning_days_per_month` | Natural ignition climatology; CSV download |

---

## Horizon

**t+1 to t+30 days** — daily wildfire risk scores and multi-target weather forecasts.

Direct multi-horizon models are trained independently at checkpoints **h = {1, 3, 7, 14, 30}**, eliminating the recursive error accumulation inherent in autoregressive approaches (a 0.5 °C/step error compounds to ≈15 °C drift at h=30). Intermediate horizons (2, 4–6, 8–13, 15–29 days) are piecewise-linearly interpolated between checkpoint forecasts, preserving temporal monotonicity without architectural complexity.

---

## Dataset

### Data Source Catalog

| # | Source | Format | Resolution | Period | Status | Used For |
|---|--------|--------|------------|--------|--------|----------|
| 1 | Open-Meteo Archive | CSV | Hourly + daily, 16 cities | 2020–present | ✅ Collected | Core weather features & forecast targets |
| 2 | NASA FIRMS VIIRS-SNPP | CSV | ~375 m points | 2020–2025 | ✅ Collected | Wildfire binary labels + count targets |
| 3 | ESA WorldCover | GeoTIFF | 10 m | 2020 | ✅ Collected | Land cover classification per city buffer |
| 4 | WorldPop | GeoTIFF | 100 m, annual | 2020–2026 | ✅ Collected | Population density proxy |
| 5 | OSM Road Network | PBF | 1:10,000 | Static | ✅ Collected | Road density / human access proxy |
| 6 | Azerbaycan.kmz (forest boundary) | KMZ/KML | Vector | Static | ✅ Collected (43 MB) | National forest fuel polygon mask |
| 7 | MESE QURULUSU (forest inventory) | KMZ/Vector | Stand-level | Static | ✅ Collected | Species, crown density, age class per stand |
| 8 | Canadian FWI System | Computed | Daily, 16 cities | 2020–present | ⚙️ Computable | FFMC, DMC, DC, ISI, BUI, FWI, DSR |
| 9 | SPEI / SPI Drought Indices | Computed | Monthly, 16 cities | 2020–present | ⚙️ Computable | Standardized drought severity indices |
| 10 | ERA5-Land (Copernicus CDS) | NetCDF→CSV | 0.1°, daily | 2001–present | ⏳ Pending | Soil moisture, evapotranspiration, LAI |
| 11 | FIRMS Extended (MODIS + NOAA20) | CSV | ~375 m–1 km | 2001–2025 | ⏳ Pending | Extended 25-year fire history for training |
| 12 | MODIS MCD64A1 Burned Area | HDF5→CSV | 500 m, monthly | 2001–present | ⏳ Pending | Actual burned area per fire event |
| 13 | MODIS MOD13Q1 (EVI + NDWI) | HDF5→CSV | 250 m, 16-day | 2001–present | ⏳ Pending | Live fuel moisture, vegetation stress index |
| 14 | SRTM DEM | GeoTIFF | 30 m | Static | ⏳ Pending | Slope, aspect, northness, topographic position |
| 15 | SMAP L4 Soil Moisture | HDF5→CSV | 9 km, daily | 2015–present | ⏳ Pending | Surface and rootzone soil moisture |
| 16 | Blitzortung Lightning | CSV | ~10 km | 2020–2025 | ⏳ Pending | Natural ignition climatology per city-month |

**Legend:** ✅ Collected and stored · ⚙️ Derivable from existing data — no new API required · ⏳ Pending acquisition

### Pipeline Architecture

```
Phase 1 — Ingestion      data/raw/ ──────────────────► data/interim/
Phase 2 — Weather        data/interim/ ─────────────► HGBR / XGBoost forecast models
Phase 3 — Climate        weather_daily_clean.csv ────► Mann-Kendall trends + anomaly reports
Phase 4 — Wildfire       wildfire_features.csv ──────► HGBC classifier + HGBR-Poisson regressor
Phase 5 — API Layer      model outputs ──────────────► FastAPI /weather /wildfire /insights /health
```

**Strict directional dependency:** Weather feeds climate analysis and wildfire prediction. No circular dependencies exist in the pipeline. All uncertainty propagates in one direction only.

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Compute derived datasets — no new API key required
python compute_fire_weather_index.py
python compute_drought_indices.py
python fetch_seasonal_forecast.py

# 3. Run full ingestion pipeline
python generate_data.py

# 4. Start the REST API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Dashboard: http://localhost:8000   |   API docs: http://localhost:8000/docs
```

---

## Key Definitions

| Term | Definition |
|------|------------|
| **FWI** | Canadian Fire Weather Index — six-component fire danger rating system (FFMC, DMC, DC, ISI, BUI, FWI, DSR) computed from temperature, humidity, wind speed, and rainfall. Industry standard validated across decades of fire data (Van Wagner, 1987). |
| **FFMC** | Fine Fuel Moisture Code — moisture content of surface litter and fine fuels (0–101 scale). FFMC ≥ 82 = high; ≥ 87 = extreme fire danger. |
| **DC** | Drought Code — moisture content of deep organic layers. DC > 300 indicates extreme drought; used as a hard threshold feature in the wildfire model. |
| **DSR** | Daily Severity Rating — exponential FWI transformation measuring expected fire suppression difficulty. DSR > 20 = extreme operational demand. |
| **SPEI** | Standardized Precipitation-Evapotranspiration Index — drought measure incorporating evaporative demand. SPEI ≤ −1.0: moderate drought · ≤ −1.5: severe · ≤ −2.0: extreme. `spei_3` (3-month) is the primary seasonal signal; `spei_1` captures short-term fuel dryness entering fire season. |
| **FIRMS** | NASA Fire Information for Resource Management System — real-time and historical satellite fire hotspot detections from VIIRS-SNPP (~375 m) and MODIS (~1 km) sensors. |
| **MESE QURULUSU** | Azerbaijani State Forest Management Inventory — stand-level polygons with species composition, crown density (0.1–1.0 in 0.1 increments), age class (20-year brackets), and RMTM administrative unit. Unique competitive advantage over open-access land-cover products. |
| **HGBC / HGBR** | HistGradientBoostingClassifier / Regressor (scikit-learn) — primary wildfire models. Classifier uses isotonic calibration (5-fold CV) for reliable probability output; regressor uses Poisson loss to enforce non-negative fire count predictions. |
| **AUC-ROC** | Area Under the ROC Curve — primary classifier evaluation metric. Target ≥ 0.85. Supplemented by Average Precision (target ≥ 0.55) given class imbalance. |
| **NDWI** | Normalized Difference Water Index — (NIR − SWIR) / (NIR + SWIR). Values < −0.1 indicate vegetation water stress; < −0.3 is critical fire-risk territory. |
| **Mann-Kendall** | Non-parametric monotonic trend test applied to annual temperature and precipitation series per city. Paired with Theil-Sen slope estimation for magnitude. |

### Model Performance Targets

**Weather Forecasting (Mean Absolute Error)**

| Target | h=1 Acceptable | h=1 Good | h=30 Acceptable | h=30 Good |
|--------|---------------|----------|----------------|----------|
| `temperature_2m` | ≤ 1.5 °C | ≤ 0.8 °C | ≤ 5.0 °C | ≤ 3.5 °C |
| `wind_speed_10m` | ≤ 1.0 m/s | ≤ 0.6 m/s | — | — |
| `rain` | ≤ 1.5 mm | ≤ 0.8 mm | — | — |

**Wildfire Risk Classification & Regression**

| Metric | Acceptable | Good |
|--------|-----------|------|
| AUC-ROC (classifier) | ≥ 0.75 | ≥ 0.85 |
| Average Precision | ≥ 0.40 | ≥ 0.55 |
| Brier Score | ≤ 0.15 | ≤ 0.10 |
| Fire count MAE | ≤ 2.0 | ≤ 1.0 |

**Evaluation protocol:** Temporal split only — last 12 months held out. No shuffling. Per-city breakdowns required to identify underperforming regions. Calibration verified via reliability diagram (bins ±0.05 of diagonal).

---

## Team & Roles

| Member | Owns | Phases | Key Responsibilities |
|--------|------|--------|----------------------|
| **Raul** | Data Collection + Wildfire Modeling | 1, 4 | Fetches all raw data (Open-Meteo, FIRMS, ESA, WorldPop, MESE QURULUSU); builds ingestion pipeline & DuckDB schema; computes FWI & SPEI; trains wildfire classifier (`fire_occurred`) and Poisson regressor (`fire_count`) |
| **Aysu** | Weather Forecasting + API Backend | 2, 5 | Weather feature engineering (lags, rolling windows, sin/cos wind decomposition); trains HGBR forecast models per horizon across all 5 targets; builds FastAPI application and `/weather` endpoint |
| **Ilaha** | EDA + Model Evaluation | 2, 4 | Data cleaning and distribution diagnostics; wildfire model calibration curves, precision-recall curves, threshold optimization; per-city performance breakdown reports |
| **Asif** | Feature Engineering + Climate Analysis + API Routes | 3, 4, 5 | Computes FWI features; spatial joins of MESE QURULUSU forest inventory to 50 km city buffers; Mann-Kendall climate trend analysis; builds `/wildfire` and `/insights` API routes |
| **Nurana** | EDA + Frontend | 2, 5 | Cross-city EDA; horizon interpolation validation; dashboard UX design; `/health` endpoint; end-to-end integration testing |

---

## Daily Activities

2026-04-20 — All Team — Project kick-off; repo structure set up, Open-Meteo API explored, 16 cities and target variables selected.
2026-04-21 — Raul — Data ingestion pipeline built; full Open-Meteo historical fetch (hourly + daily, 2020–present) for all 16 cities completed.
2026-04-22 — Raul — Database design finalized; DuckDB schema implemented, raw data loaded and validated.
2026-04-23 — Aysu / Ilaha / Nurana — Data cleaning pipeline applied; feature engineering (lags, rolling windows, heatwave flag) completed for weather data.
2026-04-24 — Raul — Pipeline automation complete; orchestrator (`generate_data.py`) with incremental loading, quality gates, and logging operational.
2026-04-25 — Asif — ARIAN v3.1 blueprint finalized; all Tier-0 datasets confirmed collected (Open-Meteo, FIRMS, ESA WorldCover, WorldPop, OSM, Azerbaycan.kmz, MESE QURULUSU).
2026-04-27 — All Team — README restructured: Team & Roles updated with clear ownership table (who owns what per member); EDA phase (Day 6) begins.
2026-04-27 — All Team — README professionalized from ARIAN Blueprint v3.1 and ARIAN_README: added geographic problem context, problem-solution mapping table, dataset catalog with acquisition status tiers, pipeline architecture overview, extended feature roadmap, FWI/SPEI interpretation, dual-table model performance targets, evaluation protocol, and Quick Start guide.
