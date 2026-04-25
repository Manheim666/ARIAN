# ARIAN — Wildfire Risk & Weather Forecasting System
## Technical Blueprint v3.1

---

## 1. Executive Summary

ARIAN (Azerbaijani Risk Intelligence and Analysis Network) is a production-grade environmental intelligence system purpose-built for Azerbaijan. It integrates multi-source satellite and meteorological data into a unified pipeline that delivers:

- **30-day multi-target weather forecasts** for 16 Azerbaijani cities
- **Daily wildfire risk scores** (calibrated probability + expected fire count)
- **Seasonal climate anomaly detection** (rising temperatures, shifting rainfall, drought trends)
- **REST API + web dashboard** for operational use

**v3.1 Status:** Core dataset assembly is complete. Open-Meteo weather, NASA FIRMS fire hotspots, ESA WorldCover land cover, WorldPop population density, OSM road network, official Azerbaijan forest boundary (Azerbaycan.kmz, 43 MB), and the Azerbaijani State Forest Management Inventory (MESE QURULUSU LAYIHE SON) have all been collected. ERA5-Land, SMAP, MODIS vegetation indices, SRTM terrain, and lightning climatology are the remaining Tier 1/2 acquisitions.

The system is built around a strict directional dependency: weather feeds climate analysis, and climate feeds wildfire prediction. All uncertainty propagates in one direction and no circular dependencies exist in the pipeline.

---

## 2. Problem Context

### 2.1 Why Azerbaijan

Azerbaijan sits at the intersection of three climate zones (semi-arid lowlands, humid subtropical Caspian coast, alpine Caucasus highlands). This geographic diversity makes it both ecologically important and disproportionately fire-prone:

- The Greater and Lesser Caucasus forest belts span roughly 1.2 million hectares
- Baku's Absheron peninsula and the Kura-Araz lowlands experience extreme summer drought
- Climate trends show +0.4°C/decade temperature rise and declining summer precipitation since 1980
- Wildfire incidents have increased measurably since 2010, with the 2021 and 2022 seasons causing substantial forest loss

### 2.2 What ARIAN Solves

| Problem | ARIAN Solution |
|---------|---------------|
| No regional-scale 30-day weather forecast | ML ensemble trained on 6+ years of hourly Open-Meteo + ERA5 data |
| Wildfire risk assessments are reactive | Calibrated 30-day forward probability from fire-science features |
| No early warning for high-risk fire windows | FWI system + drought indices provide 7–14 day lead time |
| Climate trend reports require GIS expertise | Automated Mann-Kendall + Theil-Sen trend reports per city |
| Operational data scattered across agencies | Single REST API aggregates all outputs |
| No structured forest inventory for risk zoning | State Forest Management data (MESE QURULUSU) provides species, density, age class per stand |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                            │
│                                                                 │
│  Open-Meteo  ERA5-Land  NASA FIRMS  MODIS  SRTM  SMAP          │
│  ESA Cover   OSM Roads  WorldPop  Lightning  Seasonal API       │
│  AZ Forest Inventory (MESE QURULUSU LAYIHE SON)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │  data/raw/  (immutable)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1 — INGESTION                          │
│  notebooks/01  →  data/interim/                                  │
│  Cleaning · Alignment · Quality Control · City Aggregation      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 2 — WEATHER FORECASTING                   │
│  notebooks/02–05  →  data/processed/                            │
│  Feature Engineering · Multi-Horizon Training · Forecast        │
│  Models: HistGradientBoosting, XGBoost, Ridge (per target)      │
└───────────────────────────┬─────────────────────────────────────┘
                            │  weather features feed climate & wildfire
                            ▼
         ┌──────────────────┴──────────────────┐
         │                                     │
         ▼                                     ▼
┌─────────────────────┐           ┌─────────────────────────────┐
│  PHASE 3 — CLIMATE  │           │  PHASE 4 — WILDFIRE RISK    │
│  notebooks/06       │           │  notebooks/07–09            │
│  Mann-Kendall       │           │  FWI + KBDI + NDWI + SMAP   │
│  Theil-Sen slopes   │           │  Forest Inventory features   │
│  Anomaly labelling  │           │  Calibrated HGBC classifier  │
│  reports/climate/   │           │  HGBR count regressor        │
│                     │           │  processed/wildfire_*.csv    │
└──────────┬──────────┘           └─────────────┬───────────────┘
           │                                     │
           └──────────────┬──────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 5 — API LAYER                          │
│  app/  →  FastAPI + Pydantic  →  http://localhost:8000          │
│  /weather  /wildfire  /insights  /health                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Architecture

### 4.1 Complete Data Source Catalog

| # | Source | Format | Resolution | Period | Folder | Status | Used For |
|---|--------|--------|------------|--------|--------|--------|----------|
| 1 | **Open-Meteo Archive** | CSV | Hourly, 16 cities | 2020–present | `raw/openmeteo/` | ✅ COLLECTED | Core weather features & targets |
| 2 | **NASA FIRMS VIIRS-SNPP** | CSV | ~375 m points | 2020–2025 | `raw/nasa_firms/` | ✅ COLLECTED | Wildfire labels (hotspots) |
| 3 | **ESA WorldCover** | GeoTIFF | 10 m | 2020 | `raw/earth_engine/` | ✅ COLLECTED | Land cover class per city |
| 4 | **WorldPop** | GeoTIFF | 100 m, annual | 2020–2026 | `raw/population/` | ✅ COLLECTED | Population density proxy |
| 5 | **OSM Roads** | PBF | 1:10000 | Static | `raw/osm_roads/` | ✅ COLLECTED | Road density / human-access proxy |
| 6 | **Forest Boundaries (KMZ)** | KMZ/KML | Vector | Static | `raw/forest_boundaries/` | ✅ COLLECTED (43 MB) | Forest fuel polygon mask |
| 7 | **AZ Forest Inventory** | KMZ/Vector | Stand-level | Static | `raw/forest_inventory/` | ✅ COLLECTED | Species, density, age class per stand |
| 8 | **ERA5-Land** (Copernicus CDS) | NetCDF → CSV | 0.1°, daily | 2001–present | `raw/era5/` | ⏳ PENDING | Soil moisture, ET, LAI, BLH |
| 9 | **NASA FIRMS Extended** (MODIS + NOAA20) | CSV | ~1 km / 375 m | 2001–2025 | `raw/firms_extended/` | ⏳ PENDING | Extended fire history for training |
| 10 | **MODIS MCD64A1 Burned Area** | HDF5 → CSV | 500 m, monthly | 2001–present | `raw/modis_burned_area/` | ⏳ PENDING | Actual burned area per event |
| 11 | **MODIS MOD13Q1** (EVI + NDWI) | HDF5 → CSV | 250 m, 16-day | 2001–present | `raw/vegetation/` | ⏳ PENDING | Live fuel moisture, vegetation stress |
| 12 | **SRTM DEM** (OpenTopography) | GeoTIFF | 30 m | Static | `raw/srtm/` | ⏳ PENDING | Slope, aspect, northness, TPI |
| 13 | **SMAP L4** (NASA Earthdata) | HDF5 → CSV | 9 km, daily | 2015–present | `raw/smap/` | ⏳ PENDING | Soil surface & rootzone moisture |
| 14 | **Canadian FWI System** | Computed | Daily, 16 cities | 2020–present | `raw/fire_weather_index/` | ⚙️ COMPUTABLE | FFMC, DMC, DC, ISI, BUI, FWI, DSR |
| 15 | **SPEI / SPI Drought Indices** | Computed | Monthly, 16 cities | 2020–present | `raw/drought_indices/` | ⚙️ COMPUTABLE | Standardized drought severity |
| 16 | **Open-Meteo Seasonal** | CSV | Daily ensemble | 6-month rolling | `raw/seasonal_forecast/` | ⚙️ COMPUTABLE | Long-range temperature/precip signal |
| 17 | **Blitzortung Lightning** | CSV | ~10 km | 2020–2025 | `raw/lightning/` | ⏳ PENDING | Natural ignition climatology |

**Legend:** ✅ Collected and stored · ⚙️ Derivable from existing data · ⏳ Pending acquisition

### 4.2 Azerbaijan Forest Inventory Data (MESE QURULUSU LAYIHE SON)

This is a uniquely valuable dataset sourced from the Azerbaijani state forest management authority. Unlike ESA WorldCover (which only classifies land cover), this inventory provides stand-level forest attributes:

| Attribute | Description | Fire Relevance |
|-----------|-------------|----------------|
| Species composition | Dominant tree species per stand (oak, beech, hornbeam, pine) | Ignition threshold and spread rate vary by species |
| Crown density | Canopy closure class (0.1–1.0 in 0.1 increments) | Dense canopy = higher ladder fuel load |
| Age class | Stand age in 20-year classes | Old-growth stands have higher dead fuel accumulation |
| Stand area (ha) | Polygon area per management unit | Direct wildfire exposure mapping |
| Forest management unit (RMTM) | Administrative sub-unit | Enables fire reporting at management level |

**Integration:** Join forest inventory polygons to city buffers by spatial intersection. Derive per-city aggregates:
- `dominant_species` — modal species class within city 50 km radius
- `mean_crown_density` — area-weighted mean
- `pct_old_growth` — fraction of stands in age class ≥ 80 years

These replace or augment the `forest_fraction` feature from Section 6, Phase 1.

### 4.3 Data Priority Tiers (Updated)

**Already collected — integrate immediately**
- Open-Meteo weather (6 years, 16 cities)
- NASA FIRMS VIIRS-SNPP hotspots (wildfire labels)
- ESA WorldCover (land cover classes)
- WorldPop population density
- OSM road network
- Forest boundaries (Azerbaycan.kmz)
- Azerbaijan Forest Inventory (MESE QURULUSU) ← unique competitive advantage

**Tier 1 — Critical next steps (high model lift, free, easy to obtain)**
- Canadian FWI — computed from existing Open-Meteo weather, **zero new API needed**
- SPEI/SPI drought indices — computed from existing data, **zero new API needed**
- ERA5-Land soil moisture + ET (register at Copernicus CDS, 1-day setup)
- FIRMS Extended 2001–2019 (MODIS) — same API already in use

**Tier 2 — Important (significant lift, moderate setup effort)**
- MODIS MOD13Q1 EVI + NDWI (NASA Earthdata account, AppEEARS)
- SRTM DEM (OpenTopography API key, one-time 30-min download)
- MODIS MCD64A1 Burned Area (same AppEEARS account as EVI)

**Tier 3 — Enhancement (incremental lift, more setup)**
- SMAP L4 Soil Moisture (daily pipeline, ~100 MB/year)
- Open-Meteo Seasonal Forecast (no key, run monthly)
- Blitzortung Lightning (CSV download or API)

---

## 5. Directory Structure

```
ARIAN-main/
├── data/
│   ├── raw/
│   │   ├── openmeteo/              # Open-Meteo hourly/daily archive ✅
│   │   ├── nasa_firms/             # VIIRS-SNPP 2020–2025 ✅
│   │   ├── earth_engine/           # ESA WorldCover land cover ✅
│   │   ├── population/             # WorldPop annual rasters ✅
│   │   ├── osm_roads/              # OpenStreetMap road network ✅
│   │   ├── forest_boundaries/      # Azerbaycan.kmz (43 MB) ✅
│   │   ├── forest_inventory/       # MESE QURULUSU stand-level data ✅
│   │   ├── era5/                   # ERA5-Land soil, ET, LAI, BLH ⏳
│   │   ├── firms_extended/         # MODIS 2001–2019 + NOAA20 ⏳
│   │   ├── modis_burned_area/      # MCD64A1 burned area polygons ⏳
│   │   ├── vegetation/             # MOD13Q1 EVI, NDVI, NDWI ⏳
│   │   ├── srtm/                   # DEM + computed terrain features ⏳
│   │   ├── smap/                   # SMAP L4 soil moisture ⏳
│   │   ├── fire_weather_index/     # Computed FWI system outputs ⚙️
│   │   ├── drought_indices/        # Computed SPEI/SPI ⚙️
│   │   ├── seasonal_forecast/      # Monthly Open-Meteo seasonal ⚙️
│   │   └── lightning/              # Blitzortung lightning events ⏳
│   ├── interim/                    # Phase 1 outputs (cleaned, city-aligned)
│   └── processed/                  # Phase 2–4 model inputs and outputs
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_weather_cleaning.ipynb
│   ├── 03_weather_feature_engineering.ipynb
│   ├── 04_weather_modeling.ipynb
│   ├── 05_weather_forecasting.ipynb
│   ├── 06_climate_analysis.ipynb
│   ├── 07_wildfire_feature_engineering.ipynb
│   ├── 08_wildfire_modeling.ipynb
│   ├── 09_wildfire_prediction.ipynb
│   └── 10_final_analysis.ipynb
│
├── src/
│   ├── weather/        ingestion · cleaning · features · train · forecast
│   ├── wildfire/       features · train · predict
│   ├── climate/        trends
│   └── utils/          config · logging_utils
│
├── models/
│   ├── weather/        per-target, per-horizon .joblib files
│   └── wildfire/       classifier/ + regressor/ .joblib files
│
├── reports/
│   └── climate/        climatology_daily/monthly, annual_trends, anomalies
│
├── app/
│   ├── main.py         FastAPI application
│   ├── schemas.py      Pydantic response models
│   ├── routes/
│   │   ├── weather.py
│   │   ├── wildfire.py
│   │   └── insights.py
│   └── static/
│       └── index.html  Single-page dashboard
│
├── generate_data.py    Convenience orchestrator (runs all phases)
├── setup_wildfire.py   Wildfire pipeline entry point
├── requirements.txt
└── README.md
```

---

## 6. Processing Pipeline

### Phase 1 — Data Ingestion (notebook 01)

**Inputs:** All `data/raw/` sources  
**Outputs:** `data/interim/`

| Step | Action | Data Available |
|------|--------|---------------|
| Weather | Fetch Open-Meteo hourly + daily per city via API | ✅ Now |
| Fire labels | Load FIRMS VIIRS; aggregate hotspots by city-day within 50 km radius | ✅ Now |
| Land cover | Load ESA WorldCover; clip to city buffers | ✅ Now |
| Population | Sample WorldPop rasters at city centroids | ✅ Now |
| Roads | Compute road density within 5 km buffer of each city centroid | ✅ Now |
| Forest mask | Load Azerbaycan.kmz; compute `forest_fraction` per city buffer | ✅ Now |
| Forest inventory | Parse MESE QURULUSU KMZ; derive species, crown density, age class | ✅ Now |
| FWI | Run `compute_fire_weather_index.py`; outputs FFMC/DMC/DC/ISI/BUI/FWI/DSR daily | ⚙️ Now |
| Drought | Run `compute_drought_indices.py`; outputs SPI-1/3, SPEI-1/3 monthly | ⚙️ Now |
| ERA5 backfill | ERA5-Land soil moisture, ET; fills gaps in weather record back to 2001 | ⏳ Pending |
| Burned area | Parse MCD64A1 CSVs; derive `area_burned_ha` per city | ⏳ Pending |
| Vegetation | Load MOD13Q1 EVI/NDVI/NDWI (16-day); interpolate to daily | ⏳ Pending |
| Terrain | Load pre-computed `terrain_features.csv`; static join by city | ⏳ Pending |
| Soil moisture | Load SMAP L4 daily; fill gaps with ERA5-Land swvl1 | ⏳ Pending |
| Lightning | Aggregate Blitzortung events by city-month | ⏳ Pending |

**Quality controls applied:**
- Duplicate date removal (keep last)
- Physiologically implausible value clipping (RH > 100, soil moisture > 0.55 m³/m³)
- Minimum coverage check: ≥ 80% non-null per city-year or raise warning
- City alignment: all interim files share the same (City, date) index

### Phase 2 — Weather Forecasting (notebooks 02–05)

**Feature engineering rules (no leakage):**
- All lag features: `shift(1).rolling(w)` — strictly backward-looking
- Wind direction: decomposed into `sin_wd` / `cos_wd` throughout; reconstructed via `atan2` for API output
- Train/test split: temporal — last 6 months held out; no shuffle
- City representation: one-hot city dummies pooled across all 16 cities

**Models per target per horizon:**

| Target | Horizons | Primary Model | Secondary |
|--------|----------|---------------|-----------|
| temperature_2m | 1, 3, 7, 14, 30 | HistGradientBoostingRegressor | XGBoost |
| wind_speed_10m | 1, 3, 7, 14, 30 | HistGradientBoostingRegressor | Ridge |
| wind_direction_10m | 1, 3, 7, 14, 30 | HistGradientBoostingRegressor | — |
| rain | 1, 3, 7, 14, 30 | HistGradientBoostingRegressor | XGBoost |
| precipitation | 1, 3, 7, 14, 30 | HistGradientBoostingRegressor | XGBoost |

Intermediate horizons (2, 4–6, 8–13, 15–29) are piecewise-linearly interpolated — eliminates recursive error accumulation from autoregressive approaches.

**Best model selection:** Evaluated on held-out 6-month period by MAE (regression). Selected model per target/horizon saved as `models/weather/<target>/h<N>/hgbr.joblib`.

### Phase 3 — Climate Analysis (notebook 06)

Analyses run on `data/interim/weather_daily_clean.csv` over the full historical window.

| Analysis | Method | Output |
|----------|--------|--------|
| Annual temperature trend | Mann-Kendall + Theil-Sen | `reports/climate/annual_trends.csv` |
| Monthly climatology | 20-year DOY mean ± 1σ | `reports/climate/climatology_monthly.csv` |
| Daily climatology | 7-day smoothed DOY | `reports/climate/climatology_daily.csv` |
| Anomaly labelling | z-score vs DOY climatology | Flags in processed weather |
| Forecast vs climatology | Forecast minus monthly mean | `reports/climate/forecast_anomalies.csv` |
| Seasonal peaks | Hottest / driest / wettest month per city | `reports/climate/seasonal_peaks.csv` |

### Phase 4 — Wildfire Prediction (notebooks 07–09)

#### Feature Set — Current (using collected data)

**Weather-derived (from Phase 2 output):**
- temperature_2m, wind_speed_10m, relative_humidity_2m (lags 1, 3, 7, 14 days)
- precipitation_7d_sum, rain_30d_sum
- heatwave_flag (3+ consecutive days above local p90)
- wind_spread_factor (speed × direction persistence)

**Fire Weather Index — computable now from Open-Meteo:**
- ffmc, dmc, dc, isi, bui, fwi, dsr

**Drought indices — computable now from Open-Meteo:**
- spi_1, spi_3, spei_1, spei_3

**Land cover (from ESA WorldCover):**
- land_cover_class — dominant class in city buffer
- forest_fraction — from Azerbaycan.kmz forest boundary polygon

**Forest inventory (from MESE QURULUSU — unique feature set):**
- dominant_species — modal species within 50 km buffer
- mean_crown_density — area-weighted canopy closure [0.1–1.0]
- pct_old_growth — fraction of stands age class ≥ 80 years
- forest_mgmt_unit — RMTM administrative identifier for reporting

**Human activity (from WorldPop + OSM):**
- population_density — WorldPop estimate for city buffer
- road_density_km_per_km2 — OSM road network in 5 km radius
- human_activity_score — product of the two

**Historical fire context (from FIRMS VIIRS):**
- days_since_last_fire — from FIRMS hotspot history
- fire_count_30d — FIRMS hotspot count in prior 30 days

#### Feature Set — Extended (requires pending data)

**Soil moisture (ERA5-Land + SMAP — pending):**
- sm_surface_m3m3, sm_rootzone_m3m3, swvl1, swvl2, swvl3

**Vegetation (MODIS MOD13Q1 — pending):**
- ndvi, evi, ndwi, ndwi_anomaly

**Terrain (SRTM — pending):**
- elevation_m, slope_deg, northness, tpi

**Burned area context (MCD64A1 — pending):**
- area_burned_ha, days_since_last_burn

**Lightning (Blitzortung — pending):**
- lightning_days_per_month

#### Model Architecture

**Classifier — fire_occurred (binary):**
```
CalibratedClassifierCV(
    estimator = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05,
        max_depth=6, min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    ),
    method="isotonic",
    cv=5
)
```
Threshold: optimised on PR curve for F1 on held-out temporal split.

**Regressor — fire_count (count):**
```
HistGradientBoostingRegressor(
    loss="poisson",
    max_iter=500, learning_rate=0.05,
    max_depth=5, min_samples_leaf=30,
    random_state=42
)
```

**Why Poisson loss:** Fire counts are non-negative integers; Poisson loss enforces non-negative predictions and penalises over-prediction on rare large events more appropriately than squared error.

---

## 7. Feature Engineering Details

### 7.1 Forest Inventory Features (MESE QURULUSU)

The state forest management data provides stand-level polygons. Derive city-level aggregates as follows:

```python
# Spatial join: clip inventory polygons to city 50 km buffer
stands_in_buffer = gpd.sjoin(inventory_gdf, city_buffer_gdf, how='inner')

# Area-weighted crown density
city_features['mean_crown_density'] = (
    stands_in_buffer.groupby('city')
    .apply(lambda g: np.average(g['crown_density'], weights=g['area_ha']))
)

# Old-growth fraction (age class >= 80 years)
city_features['pct_old_growth'] = (
    stands_in_buffer.groupby('city')
    .apply(lambda g: g.loc[g['age_class'] >= 80, 'area_ha'].sum() / g['area_ha'].sum())
)

# Dominant species
city_features['dominant_species'] = (
    stands_in_buffer.groupby('city')
    .apply(lambda g: g.groupby('species')['area_ha'].sum().idxmax())
)
```

### 7.2 Live Fuel Moisture (NDWI — pending MODIS)

NDWI = (NIR − SWIR) / (NIR + SWIR)

Values below −0.1 indicate vegetation water stress. Below −0.3 is critical.

Derive `ndwi_anomaly = ndwi − rolling_30d_mean(ndwi_same_doy_historical)`.

Until MODIS data is acquired, use ERA5-Land `sm_rootzone_m3m3` as a soil-moisture proxy for fuel moisture.

### 7.3 Canadian FWI Interpretation

| Index | Low | Moderate | High | Very High | Extreme |
|-------|-----|----------|------|-----------|---------|
| FFMC  | 0–72 | 72–77 | 77–82 | 82–87 | 87+ |
| FWI   | 0–4 | 4–9 | 9–17 | 17–29 | 30+ |
| DSR   | 0–1 | 1–4 | 4–10 | 10–20 | 20+ |

DC > 300 indicates extreme drought — used as a hard threshold feature.

### 7.4 Terrain Interaction Features (pending SRTM)

Fire spread is driven by slope × wind alignment:

`upslope_wind = wind_speed × max(sin(wind_dir − aspect), 0)`

Until SRTM is acquired, this feature is omitted. The forest inventory data partially compensates: steep-terrain stands are identifiable from RMTM unit geography.

### 7.5 SPEI Interpretation

SPEI ≤ −1.0: moderate drought  
SPEI ≤ −1.5: severe drought  
SPEI ≤ −2.0: extreme drought  

Use `spei_3` as the primary seasonal drought signal (3-month memory). Use `spei_1` as the short-term fuel dryness signal entering the fire season.

---

## 8. API Specification

### Base URL
```
http://localhost:8000
```

### Endpoints

| Method | Path | Description | Key Parameters |
|--------|------|-------------|----------------|
| GET | `/health` | Liveness probe | — |
| GET | `/weather` | 30-day weather forecast | `city`, `horizon`, `target` |
| GET | `/wildfire` | 30-day wildfire risk | `city`, `risk_category` |
| GET | `/insights` | Climate trend summaries | `city`, `metric` |
| GET | `/docs` | Swagger UI | — |

### Response Schemas

**GET /wildfire**
```json
{
  "anchor_date": "2026-04-25",
  "cities": ["Baku", "Ganja"],
  "rows": [
    {
      "city": "Baku",
      "forecast_date": "2026-05-01",
      "horizon_days": 6,
      "fire_probability": 0.23,
      "risk_category": "moderate",
      "expected_fire_count": 0.4,
      "fwi": 18.2,
      "spei_3": -1.3,
      "dominant_species": "oak",
      "mean_crown_density": 0.6
    }
  ]
}
```

**GET /weather**
```json
{
  "anchor_date": "2026-04-25",
  "cities": ["Baku"],
  "rows": [
    {
      "city": "Baku",
      "forecast_date": "2026-05-01",
      "horizon_days": 6,
      "temperature_2m": 22.4,
      "wind_speed_10m": 4.8,
      "wind_direction_10m": 142.0,
      "rain": 1.2,
      "precipitation": 1.2
    }
  ]
}
```

---

## 9. Performance Evaluation Framework

### Weather Model Targets

| Target | Acceptable MAE | Good MAE |
|--------|----------------|----------|
| temperature_2m (h1) | ≤ 1.5 °C | ≤ 0.8 °C |
| temperature_2m (h7) | ≤ 3.0 °C | ≤ 2.0 °C |
| temperature_2m (h30) | ≤ 5.0 °C | ≤ 3.5 °C |
| wind_speed_10m (h1) | ≤ 1.0 m/s | ≤ 0.6 m/s |
| rain (h1) | ≤ 1.5 mm | ≤ 0.8 mm |

### Wildfire Model Targets

| Metric | Acceptable | Good |
|--------|-----------|------|
| AUC-ROC (classifier) | ≥ 0.75 | ≥ 0.85 |
| Average Precision | ≥ 0.40 | ≥ 0.55 |
| Brier Score | ≤ 0.15 | ≤ 0.10 |
| Fire count MAE | ≤ 2.0 | ≤ 1.0 |

**Calibration requirement:** Reliability diagram bins (0.0–0.1, 0.1–0.2, …) should fall within ±0.05 of the diagonal. Isotonic calibration is already applied; verify with a calibration curve plot in notebook 08.

### Evaluation Protocol

- **Temporal split only** — never shuffle, never random split on time-series data
- **Hold-out period:** Last 12 months (after adding ERA5 back to 2001, use 2024 as test year)
- **Blocking:** Evaluate per-city to identify underperforming regions

---

## 10. Setup and Deployment

### 10.1 Dependencies

```bash
pip install -r requirements.txt
```

Core packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `fastapi`, `uvicorn`,
`geopandas`, `rasterio`, `cdsapi`, `h5py`, `scipy`, `requests`

### 10.2 Data Acquisition — Current State

**Already collected — no action needed**
```
data/raw/openmeteo/         # ✅ Open-Meteo weather data
data/raw/nasa_firms/        # ✅ FIRMS VIIRS-SNPP hotspots
data/raw/earth_engine/      # ✅ ESA WorldCover
data/raw/population/        # ✅ WorldPop
data/raw/osm_roads/         # ✅ OpenStreetMap road network
data/raw/forest_boundaries/ # ✅ Azerbaycan.kmz (43 MB)
data/raw/forest_inventory/  # ✅ MESE QURULUSU LAYIHE SON
```

**Step 1 — Compute derived datasets immediately (no new API)**
```bash
python compute_fire_weather_index.py   # uses existing interim weather
python compute_drought_indices.py      # uses existing interim weather
python fetch_seasonal_forecast.py      # Open-Meteo seasonal, no key
```

**Step 2 — NASA Earthdata (free, register once)**
```
1. Register at: https://urs.earthdata.nasa.gov/
2. python fetch_firms_extended.py      # MODIS 2001-2019 + NOAA20
3. python fetch_modis_burned_area.py   # MCD64A1
4. python fetch_vegetation_indices.py  # MOD13Q1 EVI/NDWI
5. python fetch_smap_soil_moisture.py  # SMAP L4
```

**Step 3 — Copernicus CDS (free, register once)**
```
1. Register at: https://cds.climate.copernicus.eu/user/register
2. Accept ERA5-Land terms
3. Add ~/.cdsapirc with UID:KEY
4. python fetch_era5_land.py           # runs overnight for 2001-present
```

**Step 4 — OpenTopography (free, register once)**
```
1. Register at: https://opentopography.org/
2. python fetch_srtm_terrain.py        # one-time, ~15 min
```

### 10.3 Running the Pipeline

```bash
# Run notebooks in order
jupyter nbconvert --to notebook --execute notebooks/01_data_ingestion.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_weather_cleaning.ipynb
# ... through 10

# Or use the convenience script
python generate_data.py
```

### 10.4 Starting the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Dashboard: `http://localhost:8000`  
API docs: `http://localhost:8000/docs`

---

## 11. Data Flow Diagram

```
Open-Meteo ──────────────────────────────────────────┐
ERA5-Land (soil, ET, BLH) ⏳ ────────────────────────┤
                                                      │
                                              weather_daily_clean.csv
                                              weather_hourly_clean.csv
                                                      │
                              ┌───────────────────────┤
                              │                       │
                              ▼                       ▼
                    FWI system ⚙️          Weather Feature Engineering
                    SPEI/SPI ⚙️            (lags, rollings, targets)
                    Drought indices                   │
                              │                       │
                              └───────────┬───────────┘
                                          │
FIRMS VIIRS-SNPP ✅ ──────────────────────┤
FIRMS Extended (MODIS) ⏳ ────────────────┤
MCD64A1 Burned Area ⏳ ───────────────────┤
MODIS EVI / NDWI ⏳ ──────────────────────┤
SMAP Soil Moisture ⏳ ─────────────────────►  wildfire_features.csv
SRTM Terrain ⏳ ──────────────────────────┤
WorldPop ✅ + OSM Roads ✅ ────────────────┤
Forest Boundaries (Azerbaycan.kmz) ✅ ────┤
Forest Inventory (MESE QURULUSU) ✅ ──────┤
ESA WorldCover ✅ ────────────────────────┘
                                          │
                                          ▼
                                Wildfire Classifier (HGBC)
                                Wildfire Regressor (HGBR-Poisson)
                                          │
                                          ▼
                                wildfire_risk_forecast.csv
                                          │
                                          ▼
                                    REST API /wildfire
```

---

## 12. Roadmap

### v3.1 — Current Sprint (active)
- [x] Collect Open-Meteo weather data (16 cities, 2020–present)
- [x] Collect NASA FIRMS VIIRS-SNPP fire hotspots
- [x] Collect ESA WorldCover land cover
- [x] Collect WorldPop population density
- [x] Collect OSM road network
- [x] Collect Azerbaycan.kmz forest boundary (43 MB, full national coverage)
- [x] Collect Azerbaijan State Forest Inventory (MESE QURULUSU LAYIHE SON)
- [ ] Compute FWI from existing weather data
- [ ] Compute SPEI/SPI from existing weather data
- [ ] Ingest and parse forest inventory KMZ into stand-level features
- [ ] Run Phase 1 ingestion with currently collected data
- [ ] Train baseline wildfire classifier on available features
- [ ] Evaluate baseline AUC-ROC and Average Precision

### v3.2 — Near Term (1–2 months)
- [ ] Ingest ERA5-Land back to 2001 and retrain all models on the full 25-year window
- [ ] Add MODIS EVI/NDWI to wildfire feature set and evaluate AUC delta
- [ ] Add FWI indices to wildfire feature set
- [ ] Add terrain features (slope, northness) from SRTM
- [ ] Add FIRMS Extended 2001–2019 for longer fire history
- [ ] Extend API response to include FWI, SPEI, forest inventory attributes in `/wildfire`

### v3.3 — Medium Term (3–6 months)
- [ ] LSTM/Transformer model for temperature_2m at h30
- [ ] Spatial wildfire spread model: GNN on city adjacency graph weighted by terrain + forest type
- [ ] Real-time pipeline: daily cron that re-runs ingestion + inference
- [ ] Alert system: push notification when FWI > 25 or fire_probability > 0.5 for any city
- [ ] Burned area polygon mapping: overlay MCD64A1 on the dashboard map
- [ ] Forest stand-level risk map: render per-RMTM fire risk derived from inventory + weather

### v3.4 — Long Term (6–12 months)
- [ ] Generalize config.py study-area to any bounding box (multi-country deployment)
- [ ] Ensemble calibration: stacked generalization across HGBR + XGBoost + Ridge
- [ ] Add smoke/air quality output (CAMS fire radiative power → PM2.5 estimate)
- [ ] Agriculture module: combine SPEI + NDWI + temperature anomaly for crop stress index
- [ ] HTTP webhook: real-time Open-Meteo update trigger without manual re-run

---

## 13. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pooled city models (not per-city) | 16 cities × 6 years = ~35k training rows per city; pooling gives 560k rows with city dummies — order-of-magnitude more data |
| Direct multi-horizon (not recursive) | Recursive errors compound; at h30, a 0.5 °C error/day → 15 °C drift. Direct models each have independent errors |
| Temporal train/test split | Fire events are spatially and temporally correlated; shuffled split leaks future into past |
| Isotonic calibration | Calibrated probabilities let emergency services use threshold-based alerts; raw scores cannot |
| Poisson loss for count regression | Fire counts are non-negative integers; Poisson is the correct likelihood, not Gaussian |
| SPEI over SPI | SPEI accounts for ET demand; a hot dry day in July has higher fire risk than the same precip deficit in cool weather — SPI misses this |
| FWI over KBDI-counter | Van Wagner (1987) validated against decades of Canadian fire data; the KBDI-like counter is an engineering workaround with no fire-science backing |
| Forest inventory over ESA WorldCover alone | Stand-level species, crown density, and age class data from Azerbaijani state management provides fuel load information that 10 m land cover classification cannot — old-growth oak stands have fundamentally different fire behavior than young beech regrowth |
| Two-phase feature set (current vs. extended) | Building the model pipeline now with available data allows training and validation to begin immediately, with pending data sources slotting in as additional feature columns without architectural changes |

---

*ARIAN Blueprint v3.1 — Updated 2026-04-25*  
*Dataset status reflects Drive inventory as of 2026-04-25*
