# 🔥 Azerbaijan Wildfire-Risk System — Phase 1

> **Scope of Phase 1:** System architecture + data ingestion layer. No EDA,
> no modelling yet. We build the spine (grid, weather, fires) end-to-end
> *first*, validate it, then layer on everything else.

---

## 1. Why this architecture

Six decisions drive the design:

| # | Decision | Why |
|---|---|---|
| 1 | **5 km grid in EPSG:32638** | Unifies raster/vector/point data. 5 km balances fire-signal density vs. compute (~3 500 cells for AZ). UTM 38N is metric, equal-area-ish, and universally supported. |
| 2 | **Open-Meteo as the single weather backbone** | One API covers historical ERA5, 16-day forecast, 9-month seasonal, CMIP6 projection. Fire-grade variables (VPD, soil moisture, ET₀) all in one schema. |
| 3 | **GeoParquet + Zarr (not PostGIS)** | Columnar + chunked = fast partition-pruned scans without a DB. PostGIS only when concurrent writes become a constraint. |
| 4 | **Prefect 2 orchestration** | Research-team-friendly DAGs from inside notebooks. Airflow's overhead isn't justified for a 5-person team. |
| 5 | **BaseIngestor pattern** | Every source normalised into the same `discover() → _ingest_one()` contract, with manifests for idempotency. |
| 6 | **Forecast-replay training mode** | Model trained on past *forecasts* (via `previous_model_run`), not past reality. Eliminates train/serve skew — the single biggest production bug in weather-driven ML. |

---

## 2. Project layout

```
wildfire_az/
├── requirements.txt
├── src/
│   ├── config/
│   │   └── settings.py          # Pydantic settings (CRS, paths, variables)
│   ├── grid/
│   │   └── builder.py           # 5 km UTM-38N grid, clipped to AZ
│   ├── ingestion/
│   │   ├── base.py              # BaseIngestor abstract class
│   │   ├── openmeteo.py         # async archive/forecast/seasonal client
│   │   └── firms.py             # NASA FIRMS multi-sensor loader + dedupe
│   ├── utils/
│   │   ├── geo.py               # CRS, reprojection, grid generation
│   │   └── io.py                # Parquet + manifest helpers
│   └── main.py                  # Phase-1 orchestration entrypoint
├── tests/
│   └── test_smoke.py            # imports, VPD math, grid sanity
├── data/
│   ├── raw/                     # untouched source copies
│   ├── cache/openmeteo/         # sharded sha256-keyed parquet cache
│   ├── bronze/                  # cleaned + CRS-normalised source data
│   ├── silver/                  # feature store (grid + aligned features)
│   └── manifests/               # JSONL per-source ingest audit logs
└── notebooks/                   # exploratory only, no logic lives here
```

---

## 3. Fire-weather variable canon (the ones that actually matter)

Pulled from Open-Meteo and stored in bronze:

- **`vapour_pressure_deficit`** — #1 predictor in modern fire-weather research (Williams et al. 2019).
- `temperature_2m`, `relative_humidity_2m`, `dewpoint_2m`
- `precipitation`, `rain` (snowfall ignored for AZ)
- `wind_speed_10m`, `wind_gusts_10m`, `wind_direction_10m`
- `shortwave_radiation`, `surface_pressure`
- `soil_moisture_0_to_7cm`, `soil_moisture_7_to_28cm`, `soil_moisture_28_to_100cm`
- `et0_fao_evapotranspiration`

Computed downstream from the above (see `openmeteo.compute_*`):

- **KBDI** (Keetch–Byram Drought Index) — 0–800 scale, fire-agency standard.
- **FWI** (Canadian Fire Weather Index) — to be added in Phase-3 via `pyfwi`.

---

## 4. Running Phase 1

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. configure data paths (override with .env in repo root)
cat > .env <<EOF
WFIRE_FIRMS_DIR=/data/FIRMS
WFIRE_NDVI_CSV=/data/EarthEngine/AZE_City_NDVI_2020-2026.csv
WFIRE_LANDCOVER_TIF=/data/EarthEngine/aze_landcover.tif
WFIRE_FOREST_KMZ=/data/Forest_border_kmz/azerbaijan.kmz
WFIRE_OSM_PBF=/data/OpenStreetMapRoadNetwork/azerbaijan-latest.osm.pbf
WFIRE_POPULATION_DIR=/data/Population
WFIRE_LIGHTNING_DIR=/data/lighting_data
EOF

# 3. smoke test
pytest -xvs tests/test_smoke.py

# 4. full phase-1 run
python -m src.main
```

**Expected outputs after a successful run:**
- `data/silver/grid/grid_cells.parquet` — ~3 500 cell grid
- `data/bronze/firms/2020.parquet` … `2024.parquet` — deduped fire events
- `data/bronze/openmeteo_history/2020-01-01_YYYY-MM-DD.parquet` — ERA5 daily
- `data/bronze/openmeteo_seasonal/issued_YYYY-MM-DD.parquet` — 1-mo forecast
- `data/manifests/*.jsonl` — per-source ingest audit

---

## 5. What's deferred to Phase 2+

Deliberately **not** in this phase:

- NDVI / land-cover / population / lightning / roads ingestors — pattern is
  identical to `FirmsIngestor`, I'll scaffold them next.
- DEM ingestion (SRTM 30 m) — needs AWS Open Data client.
- Spatial alignment layer (`grid/aligner.py`) — takes ingested bronze + grid
  and produces silver features. That's Phase 3 (feature engineering).
- Anything model-shaped.

**Rationale:** getting grid + fires + weather wired end-to-end first lets us
test temporal alignment, CRS handling, and API ergonomics on the *critical
path* before adding mass. Once those three sources round-trip cleanly, every
other source is a ~50-line class.

---

## 6. Known limitations of Phase 1

1. **Open-Meteo seasonal API** currently serves CFSv2 at ~80 km resolution —
   it's coarse for a 5 km grid. We mitigate by fetching on grid centroids
   (nearest-neighbor effectively) and documenting the resolution mismatch in
   the feature store metadata. An alternative (ECMWF SEAS5 via Copernicus
   CDS) gives finer output but requires API key management — deferred.
2. **KBDI implementation** is the classic 1968 formulation. Production fire
   agencies use site-calibrated versions. We'll swap in `pyfwi`'s full
   Canadian FWI system in Phase 3.
3. **Country-boundary clip** uses convex hull of the forest KMZ as a
   fallback. For publication-grade outputs, replace with GADM level-0 AZ.
