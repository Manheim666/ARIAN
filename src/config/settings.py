"""
src/config/settings.py
──────────────────────
Single source of truth for paths, CRS, grid geometry, API endpoints and
fire-weather variables. Loaded once at program start via `get_settings()`.

Design notes
────────────
* `pydantic_settings` is used so every value can be overridden from an .env
  file or an environment variable without touching code. This is how
  prod/stage/dev diverge cleanly.
* CRS is a *named constant* not a magic number scattered across modules.
  Changing the projection is a one-line change here.
* Paths are `pathlib.Path` objects so os.path.join never leaks into calling
  code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────
#  Azerbaijan bounding box (WGS84 lon/lat)
#  Source: official country polygon, rounded outward to avoid edge-clipping.
# ─────────────────────────────────────────────────────────────────────────
AZ_BBOX_WGS84: tuple[float, float, float, float] = (44.77, 38.39, 50.38, 41.91)
#                                                  (min_lon, min_lat, max_lon, max_lat)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="WFIRE_",
        extra="ignore",
    )

    # ── Paths ────────────────────────────────────────────────────────────
    project_root: Path = Field(default=Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default=Path("data"))
    raw_dir:  Path = Field(default=Path("data/raw"))
    cache_dir: Path = Field(default=Path("data/cache"))
    bronze_dir: Path = Field(default=Path("data/bronze"))   # cleaned raw
    silver_dir: Path = Field(default=Path("data/silver"))   # feature store
    manifest_dir: Path = Field(default=Path("data/manifests"))

    # External files (replace with real paths on your box)
    ndvi_csv: Path = Field(default=Path("EarthEngine/AZE_City_NDVI_2020-2026.csv"))
    landcover_tif: Path = Field(default=Path("EarthEngine/aze_landcover.tif"))
    forest_kmz: Path = Field(default=Path("Forest_border_kmz/azerbaijan.kmz"))
    osm_pbf: Path = Field(default=Path("OpenStreetMapRoadNetwork/azerbaijan-latest.osm.pbf"))
    firms_dir: Path = Field(default=Path("FIRMS"))
    population_dir: Path = Field(default=Path("Population"))
    lightning_dir: Path = Field(default=Path("lighting_data"))

    # ── CRS ──────────────────────────────────────────────────────────────
    # UTM zone 38N — Azerbaijan mostly falls here. ~0.5 % distortion on the
    # easternmost Absheron is acceptable for a 5 km grid.
    crs_working: str = "EPSG:32638"        # metric, equal-area-ish over AZ
    crs_wgs84:   str = "EPSG:4326"         # input/output to web clients
    crs_web_mercator: str = "EPSG:3857"    # only for tile serving

    # ── Grid ─────────────────────────────────────────────────────────────
    grid_resolution_m: int = 5_000         # 5 km cells
    bbox_wgs84: tuple[float, float, float, float] = AZ_BBOX_WGS84

    # ── Open-Meteo ───────────────────────────────────────────────────────
    openmeteo_archive_url:  str = "https://archive-api.open-meteo.com/v1/archive"
    openmeteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    openmeteo_seasonal_url: str = "https://seasonal-api.open-meteo.com/v1/seasonal"
    openmeteo_climate_url:  str = "https://climate-api.open-meteo.com/v1/climate"
    openmeteo_ensemble_url: str = "https://ensemble-api.open-meteo.com/v1/ensemble"

    # free tier: 10 k calls / day, 600 / hour, 5 / second
    openmeteo_rate_per_sec: float = 4.0    # leave headroom
    openmeteo_timeout_s: float = 30.0
    openmeteo_max_retries: int = 5

    # ── Fire-weather variable canon ──────────────────────────────────────
    # These names must match Open-Meteo's API parameter names exactly.
    openmeteo_hourly_vars: tuple[str, ...] = (
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "wind_gusts_10m",
        "wind_direction_10m",
        "shortwave_radiation",
        "surface_pressure",
        "vapour_pressure_deficit",       # ← the key fire variable
        "et0_fao_evapotranspiration",
        "soil_temperature_0_to_7cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
    )

    openmeteo_daily_vars: tuple[str, ...] = (
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_max",
        "precipitation_sum",
        "rain_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    )

    # ── FIRMS ────────────────────────────────────────────────────────────
    firms_confidence_min: int = 50         # drop low-confidence pixels (<50%)
    firms_sensors: tuple[str, ...] = ("MODIS", "VIIRS_SNPP", "VIIRS_NOAA20")

    # ── Runtime ──────────────────────────────────────────────────────────
    env: Literal["dev", "stage", "prod"] = "dev"
    log_level: str = "INFO"

    # ── Helpers ──────────────────────────────────────────────────────────
    def abs(self, relative: Path) -> Path:
        """Resolve a relative data path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else (self.project_root / p)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached accessor. Import this, not the class."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
