# ARIAN — Azerbaijani Risk Intelligence and Analysis Network

> Wildfire risk & weather forecasting system for 16 Azerbaijani cities.  
> 30-day forecasts · Calibrated fire probability · Climate anomaly detection · REST API

---

## What It Does

| Output | Description |
|--------|-------------|
| Weather forecast | 30-day multi-target forecast (temperature, wind, rain) for 16 cities |
| Wildfire risk | Daily fire probability + expected fire count for the next 30 days |
| Climate trends | Mann-Kendall trend detection and seasonal anomaly reports |
| REST API | FastAPI endpoints for all outputs at `http://localhost:8000` |

---

## Dataset Status

| Source | Status | Notes |
|--------|--------|-------|
| Open-Meteo weather | ✅ Collected | Hourly + daily, 16 cities, 2020–present |
| NASA FIRMS VIIRS-SNPP | ✅ Collected | Fire hotspots 2020–2025 |
| ESA WorldCover | ✅ Collected | 10 m land cover |
| WorldPop | ✅ Collected | Population density |
| OSM Road Network | ✅ Collected | Road density proxy |
| Azerbaycan.kmz | ✅ Collected | 43 MB national forest boundary |
| MESE QURULUSU Inventory | ✅ Collected | Stand-level species, crown density, age class |
| Canadian FWI (computed) | ⚙️ Ready to compute | Uses existing weather data |
| SPEI / SPI (computed) | ⚙️ Ready to compute | Uses existing weather data |
| ERA5-Land | ⏳ Pending | Copernicus CDS registration required |
| MODIS EVI / NDWI | ⏳ Pending | NASA Earthdata / AppEEARS |
| SRTM Terrain | ⏳ Pending | OpenTopography, one-time download |
| SMAP Soil Moisture | ⏳ Pending | NASA Earthdata daily pipeline |
| FIRMS Extended (2001–2019) | ⏳ Pending | Same API as FIRMS |
| Lightning | ⏳ Pending | Blitzortung CSV |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Compute derived datasets (no new API needed)
```bash
python compute_fire_weather_index.py
python compute_drought_indices.py
python fetch_seasonal_forecast.py
```

### 3. Run the ingestion pipeline
```bash
python generate_data.py
# or notebook by notebook:
jupyter nbconvert --to notebook --execute notebooks/01_data_ingestion.ipynb
```

### 4. Start the API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the dashboard, `/docs` for Swagger UI.

---

## Architecture

```
Open-Meteo + ERA5 → weather_daily_clean.csv → FWI / SPEI / Weather Models
                                                         │
FIRMS + Forest KMZ + ESA Cover + OSM + WorldPop ────────►  Wildfire Features
Forest Inventory (MESE QURULUSU) ───────────────────────►  stand-level features
                                                         │
                                                Wildfire Classifier (HGBC)
                                                Wildfire Regressor (HGBR-Poisson)
                                                         │
                                                   REST API /wildfire
```

**Pipeline order:** Phase 1 (Ingestion) → Phase 2 (Weather) → Phase 3 (Climate) → Phase 4 (Wildfire) → Phase 5 (API)

No circular dependencies. Weather feeds wildfire; climate feeds both.

---

## Forest Inventory (MESE QURULUSU LAYIHE SON)

The Azerbaijan State Forest Management Inventory is a unique competitive advantage over open-access datasets. It provides stand-level attributes that ESA WorldCover cannot:

- **Species composition** — oak, beech, hornbeam, pine per stand
- **Crown density** — canopy closure class [0.1–1.0]
- **Age class** — 20-year brackets; old-growth (≥ 80 yr) has higher dead fuel load
- **RMTM unit** — administrative forest management unit for reporting

These features are derived via spatial join of inventory polygons to city 50 km buffers.

---

## API Endpoints

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/health` | Liveness check |
| `GET` | `/weather?city=Baku&horizon=7` | 7-day weather forecast |
| `GET` | `/wildfire?city=Ganja` | 30-day fire risk |
| `GET` | `/insights?city=Baku&metric=temperature` | Climate trend summary |
| `GET` | `/docs` | Swagger UI |

### Example `/wildfire` response
```json
{
  "anchor_date": "2026-04-25",
  "rows": [{
    "city": "Baku",
    "forecast_date": "2026-05-01",
    "fire_probability": 0.23,
    "risk_category": "moderate",
    "expected_fire_count": 0.4,
    "fwi": 18.2,
    "spei_3": -1.3,
    "dominant_species": "oak",
    "mean_crown_density": 0.6
  }]
}
```

---

## Model Performance Targets

### Weather
| Target | Acceptable | Good |
|--------|-----------|------|
| temperature_2m h1 | ≤ 1.5 °C MAE | ≤ 0.8 °C |
| temperature_2m h7 | ≤ 3.0 °C MAE | ≤ 2.0 °C |
| temperature_2m h30 | ≤ 5.0 °C MAE | ≤ 3.5 °C |
| wind_speed_10m h1 | ≤ 1.0 m/s | ≤ 0.6 m/s |

### Wildfire
| Metric | Acceptable | Good |
|--------|-----------|------|
| AUC-ROC | ≥ 0.75 | ≥ 0.85 |
| Average Precision | ≥ 0.40 | ≥ 0.55 |
| Brier Score | ≤ 0.15 | ≤ 0.10 |
| Fire count MAE | ≤ 2.0 | ≤ 1.0 |

---

## Directory Structure

```
ARIAN-main/
├── data/raw/          # Immutable source data (see Dataset Status table)
├── data/interim/      # Phase 1 outputs: cleaned, city-aligned
├── data/processed/    # Phase 2–4 model inputs and outputs
├── notebooks/         # 01–10 in execution order
├── src/               # weather/ · wildfire/ · climate/ · utils/
├── models/            # weather/ · wildfire/ .joblib files
├── reports/climate/   # Trend reports and anomaly outputs
├── app/               # FastAPI app + Pydantic schemas + static dashboard
├── generate_data.py   # Full pipeline orchestrator
└── setup_wildfire.py  # Wildfire pipeline entry point
```

---

## Next Steps

1. **Run FWI + SPEI computation** — no new API, uses existing weather
2. **Register Copernicus CDS** — unlocks ERA5-Land (largest single model improvement)
3. **Register NASA Earthdata** — unlocks FIRMS Extended, MODIS EVI/NDWI, SMAP
4. **Parse MESE QURULUSU KMZ** — extract stand polygons into `forest_inventory_features.csv`
5. **Train baseline classifier** — evaluate AUC on collected data before adding pending sources

See `ARIAN_Blueprint_v3.md` for full technical specification.

---

*ARIAN v3.1 · 2026-04-25 · Azerbaijan wildfire & weather intelligence*
