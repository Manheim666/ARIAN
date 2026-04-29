# ARIAN — Azerbaijan Regional Intelligence for Atmospheric & Wildfire Networks

> **End-to-end wildfire risk intelligence pipeline for 16 Azerbaijani cities.**
> Four sequential Jupyter notebooks covering data ingestion, exploratory analysis, weather forecasting, and wildfire prediction — producing interactive geospatial risk maps and automated hypothesis reports.

Runs identically on **Google Colab** and **local environments** (JupyterLab / VS Code) with zero configuration changes.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Team & Task Allocation](#2-team--task-allocation)
3. [10-Day Project Timeline](#3-10-day-project-timeline)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Notebook Descriptions](#5-notebook-descriptions)
6. [Folder Structure](#6-folder-structure)
7. [Setup & Execution](#7-setup--execution)
8. [Dependencies](#8-dependencies)
9. [Cities Covered](#9-cities-covered)
10. [Data Sources & Provenance](#10-data-sources--provenance)
11. [Key Outputs & Deliverables](#11-key-outputs--deliverables)

---

## 1. Project Overview

**ARIAN** builds a complete data science pipeline that:

- **Ingests** multi-source data: Open-Meteo weather APIs (historical + forecast), NASA FIRMS satellite fire detections (MODIS C6.1, VIIRS C2), and Open-Elevation terrain data
- **Explores & engineers** 90+ features through city-by-city statistical analysis, FWI indices, lag/rolling aggregates, and cyclical time encodings
- **Forecasts** 9 weather variables hourly over a 30-day horizon using a Prophet × SARIMA × XGBoost stacking ensemble (144 model bundles: 16 cities × 9 features)
- **Predicts** hourly wildfire risk probabilities using a calibrated XGBoost classifier trained on 2M+ hourly observations
- **Generates** automated hypothesis reports comparing projected conditions against historical baselines
- **Visualizes** results through interactive Folium risk maps, Plotly dashboards, and city-level timelines

---

## 2. Team & Task Allocation

| Team Member | Role | Assigned Tasks |
|---|---|---|
| **Asif Habilov** | Data Ingestion / Data Analysis / ML Engineering | T1 — Data ingestion pipeline (NB1); T5 — XGBoost fire classifier & calibration (NB4); T8 — ML model optimization & evaluation |
| **Raul Ibrahimov** | Data Ingestion / Presentation Design | T2 — Fire label normalization & static geography (NB1); T10 — Presentation design & delivery; T11 — Documentation & README |
| **Ilaha Shafizada** | Data Analysis / Feature Engineering | T3 — Exploratory data analysis (NB2 §1–§5); T6 — FWI & lag feature engineering (NB2 §7); T9 — Outlier detection & data quality |
| **Nurana Aliyarli** | Data Analysis / Feature Engineering | T4 — Fire-weather relationship analysis (NB2 §6); T7 — Calendar & rolling feature engineering (NB2 §7); T9 — Outlier detection & data quality |
| **Aysu Mammadova** | Data Analysis / ML Engineering | T5 — Weather forecasting ensemble (NB3); T8 — Wildfire prediction & hypothesis testing (NB4); T12 — Geospatial visualization |

### Task Breakdown

| ID | Task | Description | Owner(s) | Notebook |
|----|------|-------------|----------|----------|
| T1 | Weather Data Ingestion | Historical hourly weather (Open-Meteo Archive), live 16-day forecast, HTTP caching & retry logic | Asif, Raul | NB1 |
| T2 | Fire Labels & Geography | NASA FIRMS archive parsing, daily fire-label normalization (20 km buffer), Open-Elevation terrain fetch | Raul | NB1 |
| T3 | Exploratory Data Analysis | Per-city quality audit, descriptive statistics, distribution analysis, correlation heatmaps, seasonal decomposition | Ilaha | NB2 |
| T4 | Fire-Weather Analysis | Fire-day vs. non-fire-day comparison per city, Welch's t-test for significance, feature–fire correlation ranking | Nurana | NB2 |
| T5 | Weather Forecasting Ensemble | Prophet + SARIMA + XGBoost stacking per city per feature, non-negative SLSQP weight optimization, 30-day hourly predictions | Aysu | NB3 |
| T6 | FWI Feature Engineering | FFMC, DMC, DC, ISI, BUI, FWI index computation; dry-spell tracking; rolling rain/temperature/humidity aggregates | Ilaha | NB2 |
| T7 | Temporal Feature Engineering | Lag features (1–14 day), rolling statistics (7/14/30 day), cyclical calendar encodings, seasonal flags | Nurana | NB2 |
| T8 | Wildfire Prediction & Evaluation | XGBoost classifier with isotonic calibration, PR-AUC evaluation, operational threshold selection, 30-day risk scoring | Asif, Aysu | NB4 |
| T9 | Data Quality & Outlier Detection | IQR-based outlier analysis per city, missing-value audit, data integrity validation | Ilaha, Nurana | NB2 |
| T10 | Presentation Design & Delivery | Slide deck creation, narrative structure, visual design, rehearsal, final presentation | Raul | — |
| T11 | Documentation | README, code comments, pipeline documentation, reproducibility guide | Raul | — |
| T12 | Geospatial Visualization | Folium risk maps (date-selectable), Plotly heatmaps, city-level fire-risk timelines, hypothesis report formatting | Aysu | NB3, NB4 |

---

## 3. 10-Day Project Timeline

| Day | Phase | Milestones | Team Focus |
|-----|-------|-----------|------------|
| **1** | Data Collection | Open-Meteo API integration, FIRMS archive loading, legacy CSV fallback | Asif, Raul |
| **2** | Data Collection | Static geography fetch, fire-label normalization, `master_daily.parquet` finalized | Asif, Raul |
| **3** | EDA & Feature Engineering | Per-city quality audit, descriptive statistics, distribution analysis | Ilaha, Nurana |
| **4** | EDA & Feature Engineering | Correlation analysis, seasonal decomposition, fire-weather comparison (t-tests) | Ilaha, Nurana, Aysu |
| **5** | Feature Engineering | FWI indices, lag/rolling features, calendar encodings; `engineered_daily.parquet` finalized | Ilaha, Nurana |
| **6** | ML — Weather Forecasting | Prophet + SARIMA + XGBoost stacking ensemble per city; `phase3_weather_hourly_30d.parquet` | Asif, Aysu |
| **7** | ML — Wildfire Prediction | XGBoost fire classifier, isotonic calibration, evaluation metrics | Asif, Aysu |
| **8** | Hypothesis Testing & Visualization | Automated hypothesis generation, Folium risk maps, Plotly dashboards | Asif, Aysu |
| **9** | Integration & Presentation Prep | End-to-end pipeline validation, slide deck design, narrative drafting | All team |
| **10** | **Presentation** | Final rehearsal and delivery | All team |

---

## 4. Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ARIAN Pipeline Flow                              │
│                                                                          │
│  ┌──────────────┐   ┌──────────────────┐    ┌──────────────────────────┐ │
│  │   NB1        │   │   NB2            │    │   NB3                    │ │
│  │   Data       │──▶│   EDA &          │    │   Weather Forecasting    │ │
│  │   Ingestion  │   │   Feature Eng.   │    │   (Prophet+SARIMA+XGB)   │ │
│  └──────┬───────┘   └──────────────────┘    └───────────┬────────────┬─┘ │
│         │                                               │            │   │
│         │           weather_hourly.parquet              │            │   │
│         ├──────────────────────────────────────────────▶│            │   │
│         │                                               │            │   │
│         │  master_daily.parquet                         │            │   │
│         │  fires_daily.parquet                          ▼            │   │
│         │  static_geography.parquet      phase3_weather_hourly_30d   │   │
│         │                                               │            │   │
│         │         ┌────────────────────────────────────┐│            │   │
│         └────────▶│   NB4                              ││            │   │
│                   │   Wildfire Prediction &            │◀────────────┘   │
│                   │   Hypothesis Testing               │                 │
│                   │   (XGBoost + Calibration)          │                 │
│                   └────────────┬───────────────────────┘                 │
│                                │                                         │
│                                ▼                                         │
│                   phase4_wildfire_hourly_30d.parquet                     │
│                   Risk Maps · Dashboards · Hypotheses                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Notebook Descriptions

### NB1 — Data Ingestion (`01_Data_Ingestion.ipynb`)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Collect, unify, and persist all raw data for the pipeline |
| **Weather Source** | Open-Meteo Archive API (ERA5-Land, hourly, 2012–present) + 16-day live forecast |
| **Weather Variables** | Temperature, humidity, rain, wind speed/direction, pressure, solar radiation, soil temperature, soil moisture (9 features) |
| **Fire Source** | NASA FIRMS CSVs (MODIS C6.1, SUOMI-NPP VIIRS, J1 VIIRS, J2 VIIRS) |
| **Fire Normalization** | Binary daily label per city within a 20 km buffer radius |
| **Geography** | Open-Elevation API + derived slope; land-cover and population from supplementary CSVs |
| **Fallback Logic** | Local parquet cache → legacy `merged.csv` → API (with retry/rate-limit handling) |
| **Outputs** | `master_daily.parquet` (36,928 rows × 29 cols), `weather_hourly.parquet` (886K+ rows), `fires_daily.parquet`, `cities.parquet`, `static_geography.parquet` |
| **Runtime** | ~5–15 min (cached) |

### NB2 — EDA & Data Engineering (`02_Weather_Forecasting.ipynb`)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Comprehensive city-by-city analysis and feature engineering |
| **Quality Audit** | Per-city missing-value report across all 12 weather columns |
| **Descriptive Stats** | Mean, std, quartiles, skewness, kurtosis per city per feature |
| **Visualizations** | Boxplots, histograms, correlation heatmaps, seasonal decomposition, monthly time series |
| **Fire-Weather Analysis** | Welch's t-test comparing fire-day vs. non-fire-day weather per city (111/192 tests significant) |
| **Feature Engineering** | FWI family (FFMC, DMC, DC, ISI, BUI, FWI), dry-spell tracking, lag features (1–14 day), rolling stats (7/14/30 day), cyclical calendar encodings |
| **Outlier Detection** | IQR-based per-city per-feature outlier counts |
| **Outputs** | `engineered_daily.parquet` (83,392 rows × 90 cols), per-city profiles CSV, correlation heatmaps, fire comparison CSV |
| **Runtime** | ~3–5 min |

### NB3 — Weather Prediction (`03_Fire_Risk_Bridge.ipynb`)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Forecast all 9 weather features hourly for 30 days using a stacking ensemble |
| **Models** | **Prophet** (yearly/weekly/daily seasonality, Azerbaijan holidays), **SARIMA** (downsampled 6-hourly for tractability), **XGBoost** (recursive multi-step with lag/rolling features) |
| **Stacking** | Non-negative weight optimization via SLSQP on a 3-day validation window |
| **Scale** | 144 model bundles (16 cities × 9 features), 7-day test holdout for evaluation |
| **Visualization** | Interactive Folium map with date-selectable layers, temperature-coded city markers |
| **Outputs** | `phase3_weather_hourly_30d.parquet` (11,520 rows), `phase3_weather_leaderboard.csv`, `phase3_weather_map.html` |
| **Runtime** | ~60–120 min |

### NB4 — Wildfire Prediction & Hypothesis Testing (`04_Evaluation_and_Geospatial.ipynb`)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Train a fire classifier, score the 30-day forecast, and generate automated hypotheses |
| **Classifier** | XGBoost (1500 estimators, early stopping) with isotonic calibration via `FrozenEstimator` |
| **Training Data** | 2M+ hourly observations with 33 features (weather + geography + temporal + FWI) |
| **Evaluation** | PR-AUC, ROC-AUC, F1/Precision/Recall at operational threshold; 85/15 time-respecting split |
| **Risk Scoring** | 30-day hourly fire-risk probabilities with 4-tier classification (Low / Moderate / High / Extreme) |
| **Hypothesis Testing** | 5 automated hypotheses: rainfall vs. last year, rainfall vs. decade, wildfire frequency change, temperature change, humidity change |
| **Visualizations** | Folium risk map (date-selectable + heatmap layers), Plotly City × Date heatmap, per-city hourly risk timelines |
| **Outputs** | `phase4_wildfire_hourly_30d.parquet` (11,520 rows), `phase4_wildfire_scores.csv`, `phase4_hypotheses.csv`, `phase4_risk_map.html`, `phase4_xgb_fire.joblib` |
| **Runtime** | ~5–10 min |

---

## 6. Folder Structure

```
ARIAN_ASIF/
├── notebooks/                            Run in order: 01 → 02 → 03 → 04
│   ├── 01_Data_Ingestion.ipynb           Phase 1 — Data collection & unification
│   ├── 02_Weather_Forecasting.ipynb      Phase 2 — EDA & feature engineering
│   ├── 03_Fire_Risk_Bridge.ipynb         Phase 3 — Weather forecasting ensemble
│   └── 04_Evaluation_and_Geospatial.ipynb Phase 4 — Wildfire prediction & viz
│
├── data/
│   ├── raw/
│   │   ├── firms/                        NASA FIRMS sensor archives
│   │   │   ├── modis_c61/
│   │   │   ├── suomi_viirs_c2/
│   │   │   ├── j1_viirs_c2/
│   │   │   └── j2_viirs_c2/
│   │   └── legacy/                       Fallback: merged.csv + supplementary CSVs
│   ├── processed/                        master_daily.parquet, fires_daily.parquet,
│   │                                     engineered_daily.parquet
│   └── reference/                        cities.parquet, static_geography.parquet
│
├── outputs/                              All pipeline artefacts (.csv/.parquet/.html)
├── models/                               Trained models + manifests (.joblib/.json)
└── README.md
```

> All files under `outputs/` and `models/` are generated programmatically — nothing is hand-edited.

---

## 7. Setup & Execution

### Google Colab

1. Upload the project folder to your Google Drive root (→ `/MyDrive/ARIAN_Data/`)
2. Open any notebook in Colab
3. Run the first cell — it auto-mounts Drive, detects the project root, and creates directories
4. Execute cells top-to-bottom

Override the project path if needed:
```python
import os; os.environ["ARIAN_ROOT"] = "/content/drive/MyDrive/path/to/ARIAN_Data"
```

### Local (JupyterLab / VS Code)

```bash
cd ARIAN_ASIF
pip install -r requirements.txt
jupyter lab notebooks/
```

The setup cell auto-detects the project root by walking up the directory tree. Override with:
```bash
export ARIAN_ROOT=/absolute/path/to/project
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ARIAN_ROOT` | Absolute path to project root. Overrides auto-detection. |

---

## 8. Dependencies

```
# Core
pandas>=2.1
numpy
pyarrow
requests
requests-cache
retry-requests
openmeteo-requests
tqdm

# Phase 2 — EDA
matplotlib
seaborn
scipy
statsmodels

# Phase 3 — Forecasting
prophet
xgboost>=2.0
scikit-learn>=1.4
holidays

# Phase 4 — Prediction & Visualization
folium
plotly
joblib
```

Each notebook installs missing packages automatically via `%pip install` in its first code cell.

---

## 9. Cities Covered

| City | Latitude | Longitude | Key Characteristics |
|------|----------|-----------|-------------------|
| Baku | 40.409 | 49.867 | Capital, highest urban %, highest fire rate (30.1%) |
| Sumqayit | 40.590 | 49.669 | Industrial, coastal |
| Ganja | 40.683 | 46.361 | Western highlands |
| Mingachevir | 40.764 | 47.060 | Central lowlands |
| Shirvan | 39.932 | 48.930 | Kura-Araxes lowland |
| Lankaran | 38.752 | 48.848 | Southern subtropical |
| Shaki | 41.198 | 47.169 | Northern foothills |
| Nakhchivan | 39.209 | 45.412 | Exclave, arid continental |
| Yevlakh | 40.618 | 47.150 | Central plains |
| Quba | 41.361 | 48.526 | Northern mountainous |
| Khachmaz | 41.464 | 48.806 | Northeastern coastal |
| Gabala | 40.998 | 47.847 | Highest elevation (996 m), most forested (50.5%) |
| Shamakhi | 40.630 | 48.641 | Mountain plateau |
| Jalilabad | 39.209 | 48.299 | Southern lowlands |
| Barda | 40.374 | 47.127 | Karabakh region |
| Zaqatala | 41.630 | 46.643 | Northwestern, lowest fire rate (3.5%) |

Fire labels are aggregated daily within a **20 km radius** of each city centroid.

---

## 10. Data Sources & Provenance

| Source | Data | Access | Notes |
|--------|------|--------|-------|
| **Open-Meteo Archive** | Historical hourly weather (2012–present) | Free, no API key | ERA5-Land reanalysis; 9 variables |
| **Open-Meteo Forecast** | 16-day ahead hourly weather | Free, no API key | Updated daily |
| **NASA FIRMS** | Active fire detections | Free, archive CSVs | MODIS C6.1 + VIIRS C2 (3 sensors); confidence ≥ "n" (VIIRS) / ≥ 30 (MODIS) |
| **Open-Elevation** | Terrain elevation | Free, no API key | Slope derived from 4-neighbour 1 km DEM cross |
| **Supplementary CSVs** | Land cover %, urban %, population | Local reference files | Static per-city attributes |

### Caveats

- **Fire risk** represents the probability of any FIRMS-detected hotspot occurring, **not** predicted burn area
- The classifier is calibrated with **isotonic regression**; use the operational threshold from `phase4_manifest.json`
- Evaluate with **PR-AUC and recall at the operational threshold** — accuracy is misleading given ~10% fire-day prevalence
- SARIMA convergence warnings during NB3 training are expected and handled gracefully via fallback models

---

## 11. Key Outputs & Deliverables

### Data Artefacts

| File | Description | Produced By |
|------|-------------|-------------|
| `data/processed/master_daily.parquet` | 36,928 city-day rows × 29 columns | NB1 |
| `data/raw/weather_hourly.parquet` | 886K+ hourly weather observations | NB1 |
| `data/processed/fires_daily.parquet` | 4,536 fire-day labels across 16 cities | NB1 |
| `data/processed/engineered_daily.parquet` | 83,392 rows × 90 engineered features | NB2 |
| `outputs/phase3_weather_hourly_30d.parquet` | 11,520-row 30-day hourly weather forecast | NB3 |
| `outputs/phase4_wildfire_hourly_30d.parquet` | 11,520-row 30-day hourly fire-risk forecast | NB4 |

### Interactive Visualizations

| File | Description |
|------|-------------|
| `outputs/phase3_weather_map.html` | Folium map — predicted weather by date (temperature-coded markers) |
| `outputs/phase4_risk_map.html` | Folium map — wildfire risk by date (4-tier color coding + heatmap) |
| `outputs/phase4_risk_heatmap.html` | Plotly heatmap — City × Date fire risk matrix |
| `outputs/phase4_city_timelines.html` | Plotly dashboard — hourly risk timelines per city |

### ML Models & Reports

| File | Description |
|------|-------------|
| `models/phase4_xgb_fire.joblib` | Trained & calibrated XGBoost fire classifier |
| `models/phase4_manifest.json` | Feature order, operational threshold, test metrics |
| `outputs/phase4_wildfire_scores.csv` | PR-AUC, ROC-AUC, F1, Precision, Recall |
| `outputs/phase4_hypotheses.csv` | 5 automated climate-vs-fire hypotheses |
| `outputs/phase4_feature_importance.csv` | Feature importance ranking (gain-based) |

---

*ARIAN — Built by the team at ADA University, 2026.*
