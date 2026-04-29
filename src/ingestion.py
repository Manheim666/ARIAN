from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


try:
    import requests
    import requests_cache
    import openmeteo_requests
    from retry_requests import retry
except Exception:
    requests = None
    requests_cache = None
    openmeteo_requests = None
    retry = None


from .config import CITIES, PipelineConfig


@dataclass(frozen=True)
class IngestionPaths:
    raw_dir: Path
    processed_dir: Path
    firms_dir: Path

    @property
    def weather_hourly_cache(self) -> Path:
        return self.raw_dir / "weather_hourly.parquet"

    @property
    def weather_forecast_cache(self) -> Path:
        return self.raw_dir / "weather_forecast.parquet"

    @property
    def firms_raw_parquet(self) -> Path:
        return self.raw_dir / "firms_raw.parquet"

    @property
    def fires_daily_parquet(self) -> Path:
        return self.processed_dir / "fires_daily.parquet"


_COLUMN_MAP: dict[str, str] = {
    "temperature_2m": "Temperature_C",
    "relative_humidity_2m": "Humidity_percent",
    "precipitation": "Precipitation_mm",
    "wind_speed_10m": "Wind_Speed_kmh",
    "wind_direction_10m": "Wind_Direction_deg",
}


def validate_weather_hourly_schema(df: pd.DataFrame) -> None:
    required = {"Timestamp", "City", "Latitude", "Longitude", *list(_COLUMN_MAP.values())}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Weather hourly schema missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        raise ValueError("Weather hourly Timestamp must be datetime")
    if df["City"].isna().any():
        raise ValueError("Weather hourly has null City values")


def validate_firms_schema(df: pd.DataFrame) -> None:
    if df.empty:
        return
    required = {"latitude", "longitude", "acq_date"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"FIRMS schema missing columns: {missing}")
    for c in ["latitude", "longitude"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"FIRMS column {c} must be numeric")


def _ensure_openmeteo_client(cache_dir: Path):
    if openmeteo_requests is None or requests_cache is None or retry is None:
        raise RuntimeError(
            "Missing ingestion dependencies. Install: openmeteo-requests requests-cache retry-requests"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_session = requests_cache.CachedSession(str(cache_dir / "openmeteo"), expire_after=86400)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
    return openmeteo_requests.Client(session=retry_session)


def _fetch_with_retry(om, url: str, params: dict[str, Any], max_retries: int = 10, timeout_s: float = 60.0):
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return om.weather_api(url, params=params)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg or "too many" in msg:
                time.sleep(min(60, 2 * attempt))
                continue
            time.sleep(min(30, 0.5 * attempt))

    raise RuntimeError(f"Open-Meteo failed after {max_retries} attempts") from last_exc


def _parse_hourly(resp, var_list: list[str]) -> pd.DataFrame:
    h = resp.Hourly()
    timestamps = pd.date_range(
        start=pd.to_datetime(h.Time(), unit="s", utc=True),
        end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h.Interval()),
        inclusive="left",
    )
    data: dict[str, Any] = {"Timestamp": timestamps}
    for i, var in enumerate(var_list):
        data[_COLUMN_MAP[var]] = h.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)


def fetch_historical_hourly(paths: IngestionPaths, cfg: PipelineConfig, cache_dir: Path, incremental_start: str | None) -> pd.DataFrame:
    """Fetch historical hourly weather incrementally.

    If cache exists, only fetch missing range and append.
    """

    hourly_vars = list(_COLUMN_MAP.keys())
    om = _ensure_openmeteo_client(cache_dir)

    cached: pd.DataFrame | None = None
    if paths.weather_hourly_cache.exists():
        cached = pd.read_parquet(paths.weather_hourly_cache)
        cached["Timestamp"] = pd.to_datetime(cached["Timestamp"], utc=True)

    start_date = cfg.history_start
    if incremental_start:
        start_date = incremental_start
    elif cached is not None and not cached.empty:
        last_ts = cached["Timestamp"].max()
        start_date = (last_ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")

    end_date = cfg.history_end
    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        if cached is not None:
            validate_weather_hourly_schema(cached)
            return cached
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for city, (lat, lon) in CITIES.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
            "hourly": hourly_vars,
        }
        resp = _fetch_with_retry(om, "https://archive-api.open-meteo.com/v1/archive", params)[0]
        df = _parse_hourly(resp, hourly_vars)
        df["City"], df["Latitude"], df["Longitude"] = city, lat, lon
        frames.append(df)
        time.sleep(cfg.polite_delay_s)

    new_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if cached is None or cached.empty:
        out = new_data
    else:
        out = pd.concat([cached, new_data], ignore_index=True)
        out = out.drop_duplicates(subset=["City", "Timestamp"], keep="last")

    out.to_parquet(paths.weather_hourly_cache, index=False)
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True)
    validate_weather_hourly_schema(out)
    return out


def fetch_forecast_hourly(paths: IngestionPaths, cfg: PipelineConfig, cache_dir: Path) -> pd.DataFrame:
    hourly_vars = list(_COLUMN_MAP.keys())

    if paths.weather_forecast_cache.exists():
        age_h = (time.time() - paths.weather_forecast_cache.stat().st_mtime) / 3600
        if age_h < 12:
            df = pd.read_parquet(paths.weather_forecast_cache)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
            return df

    om = _ensure_openmeteo_client(cache_dir)

    frames: list[pd.DataFrame] = []
    for city, (lat, lon) in CITIES.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "UTC",
            "hourly": hourly_vars,
            "forecast_days": int(cfg.forecast_horizon_days),
        }
        resp = _fetch_with_retry(om, "https://api.open-meteo.com/v1/forecast", params)[0]
        df = _parse_hourly(resp, hourly_vars)
        df["City"], df["Latitude"], df["Longitude"] = city, lat, lon
        frames.append(df)
        time.sleep(cfg.polite_delay_s)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_parquet(paths.weather_forecast_cache, index=False)
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True)
    if not out.empty:
        validate_weather_hourly_schema(out)
    return out


def load_firms_csvs(paths: IngestionPaths) -> pd.DataFrame:
    firms = paths.firms_dir
    if not firms.exists():
        return pd.DataFrame()

    csvs = list(firms.rglob("*.csv")) + list(firms.rglob("*.CSV"))
    frames: list[pd.DataFrame] = []
    for f in csvs:
        try:
            df = pd.read_csv(f, low_memory=False, on_bad_lines="skip")
            df["source_file"] = f.name
            df["sensor_folder"] = f.parent.name
            frames.append(df)
        except Exception:
            continue

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    out.columns = [c.strip().lower() for c in out.columns]
    out = out.rename(columns={"lat": "latitude", "lon": "longitude"})

    dedupe_cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time", "satellite"] if c in out.columns]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols)

    for col in ("brightness", "scan", "track", "frp"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "confidence" in out.columns:
        out["confidence"] = out["confidence"].astype(str)
    if "version" in out.columns:
        out["version"] = out["version"].astype(str).fillna("")

    out = out.reset_index(drop=True)
    out.to_parquet(paths.firms_raw_parquet, index=False)
    validate_firms_schema(out)
    return out


_KM_PER_DEG_LAT = 110.574


def _km_to_deg_lat(km: float) -> float:
    return km / _KM_PER_DEG_LAT


def _km_to_deg_lon(km: float, lat: float) -> float:
    import numpy as np

    return km / (111.32 * float(np.cos(np.radians(lat))))


def _bbox_around(lat: float, lon: float, buffer_km: float):
    dlat, dlon = _km_to_deg_lat(buffer_km), _km_to_deg_lon(buffer_km, lat)
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def normalise_fires_daily(fires_raw: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    cols = ["City", "Date", "Fire_Occurred", "fire_count", "mean_brightness", "max_frp"]
    if fires_raw.empty:
        return pd.DataFrame(columns=cols)

    import numpy as np

    df = fires_raw.copy()
    if "acq_date" not in df.columns or "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame(columns=cols)

    df["Date"] = pd.to_datetime(df["acq_date"]).dt.date
    bboxes = {c: _bbox_around(lat, lon, cfg.fire_buffer_km) for c, (lat, lon) in CITIES.items()}

    def find_city(lat: float, lon: float):
        for c, (w, s, e, n) in bboxes.items():
            if s <= lat <= n and w <= lon <= e:
                return c
        return None

    df["City"] = [find_city(la, lo) for la, lo in zip(df["latitude"], df["longitude"]) ]
    df = df.dropna(subset=["City"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["brightness_norm"] = df.get("bright_ti4", df.get("brightness", pd.Series(np.nan, index=df.index)))
    df["frp"] = df.get("frp", pd.Series(0.0, index=df.index))

    daily = (
        df.groupby(["City", "Date"])
        .agg(
            fire_count=("latitude", "count"),
            mean_brightness=("brightness_norm", "mean"),
            max_frp=("frp", "max"),
        )
        .reset_index()
    )
    daily["Fire_Occurred"] = 1
    # Standardize to naive daily timestamps for consistent joins and DuckDB DATE loading
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.floor("D")
    return daily
