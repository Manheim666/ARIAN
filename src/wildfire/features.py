"""
Wildfire feature engineering (Phase 4 - Step 1).

Builds a daily per-city dataset that joins:
  - weather history + predicted weather (Phase 2)
  - vegetation (NDVI, Phase 1)
  - lightning climatology (Phase 1)
  - static human-access proxies: population & road density (Phase 1)
  - FIRMS historical fires (Phase 1) - used to derive the target label

And derives fire-science features on top:
  - **Drought index (KBDI-like / running dry-day counter)**
  - **Heatwave indicator** (3+ consecutive days above local 90th-percentile temp)
  - **Wind spread factor** (speed * direction persistence)
  - **Fuel dryness** (rolling NDVI dip vs climatology)
  - **Human activity proxy** (pop density * road density, static)

Target definitions
------------------
Two harmonised labels per (city, date):

  ``fire_occurred``    : binary {0, 1}. 1 if ≥ 1 FIRMS vegetation hotspot
                         (type=0, confidence≠l) within the city's radius
                         on that date.
  ``fire_count``       : non-negative integer. Number of such hotspots.
  ``fire_frp_total``   : float. Summed FRP (MW) — intensity proxy.

Regression target = ``fire_count``, classification target = ``fire_occurred``.

Public API
----------
- :func:`count_firms_within_radius` - aggregate fire labels per city-day
- :func:`add_drought_index` - KBDI-like running dry-day index
- :func:`add_heatwave_indicator` - consecutive-hot-day counter vs local p90
- :func:`add_wind_spread_factor` - speed * direction-persistence interaction
- :func:`add_human_activity` - join static population + road density
- :func:`build_wildfire_features` - orchestrator producing the modelling table
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    AZERBAIJAN_CITIES,
    INTERIM_DIR,
    PROCESSED_DIR,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

FIRE_RADIUS_KM: float = 50.0          # hotspots within this ring -> city-day
FIRMS_MIN_CONFIDENCE: str = "n"        # drop "l" (low)
FIRMS_VEG_TYPE: int = 0                # type=0 == vegetation fire

# Temperature percentile defining a "hot" day (local, per city)
HOT_DAY_PERCENTILE: float = 90.0
HEATWAVE_MIN_DAYS: int = 3             # consecutive hot days to flag

# KBDI-like dry-day counter parameters
DRY_THRESHOLD_MM: float = 1.0          # day counts as "dry" if rain < this
WET_RESET_MM: float = 10.0             # a wet spell of this size resets index


# ============================================================================
# 1. Fire target aggregation
# ============================================================================

def haversine_km(
    lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorised great-circle distance in km."""
    R = 6371.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlam = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def count_firms_within_radius(
    firms: pd.DataFrame,
    cities: pd.DataFrame,
    radius_km: float = FIRE_RADIUS_KM,
    veg_type: int = FIRMS_VEG_TYPE,
    exclude_confidence: Iterable[str] = ("l",),
) -> pd.DataFrame:
    """Return per (city, date) fire aggregates.

    Output columns: ``City``, ``date``, ``fire_count``, ``fire_frp_total``,
    ``fire_frp_max``, ``fire_occurred``.
    """
    f = firms.copy()
    if "type" in f.columns:
        f = f[f["type"] == veg_type]
    if "confidence" in f.columns:
        f = f[~f["confidence"].isin(list(exclude_confidence))]
    logger.info("FIRMS filtered: %d vegetation hotspots", len(f))

    lat = f["latitude"].to_numpy()
    lon = f["longitude"].to_numpy()

    all_rows: List[pd.DataFrame] = []
    for _, c in cities.iterrows():
        d_km = haversine_km(c["Latitude"], c["Longitude"], lat, lon)
        mask = d_km <= radius_km
        if not mask.any():
            continue
        near = f.loc[mask, ["acq_date", "frp"]].copy()
        agg = near.groupby("acq_date").agg(
            fire_count=("frp", "size"),
            fire_frp_total=("frp", "sum"),
            fire_frp_max=("frp", "max"),
        ).reset_index().rename(columns={"acq_date": "date"})
        agg["City"] = c["City"]
        all_rows.append(agg)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["date", "fire_count", "fire_frp_total", "fire_frp_max", "City"]
    )
    out["fire_occurred"] = (out["fire_count"] > 0).astype(np.int8)
    logger.info("Fire-day rows (non-zero): %d across %d cities",
                len(out), out["City"].nunique())
    return out[["City", "date", "fire_count", "fire_frp_total", "fire_frp_max", "fire_occurred"]]


def _fill_zero_fire_days(
    fire_per_day: pd.DataFrame,
    city_days: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join fire counts onto the complete (city, date) grid; missing -> 0."""
    merged = city_days.merge(fire_per_day, on=["City", "date"], how="left")
    for c in ("fire_count", "fire_frp_total", "fire_frp_max", "fire_occurred"):
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)
    merged["fire_occurred"] = merged["fire_occurred"].astype(np.int8)
    merged["fire_count"] = merged["fire_count"].astype(np.int32)
    return merged


# ============================================================================
# 2. Drought index (KBDI-like, simplified)
# ============================================================================

def add_drought_index(
    df: pd.DataFrame,
    rain_col: str = "rain_sum",
    temp_col: str = "temperature_2m_max",
    group_col: str = "City",
) -> pd.DataFrame:
    """Append two drought features per city:

    - ``dry_days_run`` : consecutive dry days (rain < DRY_THRESHOLD) ending today.
      Resets to 0 on any wet day (>= WET_RESET_MM).
    - ``drought_index`` : cumulative temp × (1 - min(rain/WET_RESET_MM, 1)),
      running sum, resets on wet-day thresholds. Proxy for KBDI without
      needing ET inputs.

    Both use **strictly past-through-today** data (no look-ahead).
    """
    if rain_col not in df.columns or temp_col not in df.columns:
        logger.warning("Skipping drought: %s or %s missing", rain_col, temp_col)
        return df

    df = df.sort_values([group_col, "date"]).reset_index(drop=True)
    dry_runs = np.zeros(len(df), dtype=np.int16)
    drought = np.zeros(len(df), dtype=np.float32)

    # Work per-city (simple Python loop -- O(n) and clear)
    for city, g in df.groupby(group_col, sort=False):
        idx = g.index.to_numpy()
        rain = g[rain_col].to_numpy(dtype=float)
        tmax = g[temp_col].to_numpy(dtype=float)
        run = 0.0
        cum = 0.0
        for i, k in enumerate(idx):
            # Dry-day counter
            if np.isfinite(rain[i]) and rain[i] < DRY_THRESHOLD_MM:
                run += 1
            else:
                run = 0
            dry_runs[k] = int(run)
            # Drought index: accumulate daily stress, discount with rain
            if np.isfinite(rain[i]) and rain[i] >= WET_RESET_MM:
                cum = 0.0
            else:
                stress = max(0.0, (tmax[i] if np.isfinite(tmax[i]) else 0.0))
                wet_discount = 0.0 if not np.isfinite(rain[i]) else min(rain[i] / WET_RESET_MM, 1.0)
                cum += stress * (1.0 - wet_discount)
            drought[k] = cum

    df["dry_days_run"] = dry_runs
    df["drought_index"] = drought
    logger.info("Added drought_index and dry_days_run")
    return df


# ============================================================================
# 3. Heatwave indicator
# ============================================================================

def add_heatwave_indicator(
    df: pd.DataFrame,
    temp_col: str = "temperature_2m_max",
    group_col: str = "City",
    percentile: float = HOT_DAY_PERCENTILE,
    min_consecutive: int = HEATWAVE_MIN_DAYS,
) -> pd.DataFrame:
    """Append two heatwave features:

    - ``hot_day`` : 1 if temp > local p90 (computed per city over history).
    - ``heatwave_active`` : 1 if today is part of a run of >= ``min_consecutive``
      consecutive ``hot_day == 1`` days.

    The percentile threshold is computed **once from the training history**
    (whatever is passed in), so future rows (forecast) are compared against
    the same threshold. This is a deliberate choice: we don't want the
    threshold to shift as new forecasts arrive.
    """
    if temp_col not in df.columns:
        logger.warning("Skipping heatwave: %s missing", temp_col)
        return df

    df = df.sort_values([group_col, "date"]).reset_index(drop=True)

    hot = np.zeros(len(df), dtype=np.int8)
    heatwave = np.zeros(len(df), dtype=np.int8)
    thresholds: Dict[str, float] = {}

    for city, g in df.groupby(group_col, sort=False):
        # Threshold from historical rows only (date < today's max by default
        # here we use all rows that pass, which in training == history)
        vals = g[temp_col].dropna().values
        if len(vals) == 0:
            continue
        thr = float(np.percentile(vals, percentile))
        thresholds[city] = thr
        is_hot = (g[temp_col].values >= thr).astype(np.int8)
        hot[g.index] = is_hot

        # Consecutive-hot-day run-length
        run = 0
        hw = np.zeros(len(g), dtype=np.int8)
        for i, h in enumerate(is_hot):
            run = run + 1 if h else 0
            hw[i] = 1 if run >= min_consecutive else 0
        heatwave[g.index] = hw

    df["hot_day"] = hot
    df["heatwave_active"] = heatwave
    logger.info("Heatwave thresholds (p%d): %s",
                int(percentile), {k: round(v, 2) for k, v in thresholds.items()})
    return df


# ============================================================================
# 4. Wind spread factor
# ============================================================================

def add_wind_spread_factor(
    df: pd.DataFrame,
    speed_col: str = "wind_speed_10m_mean",
    dir_col: str = "wind_direction_10m",
    group_col: str = "City",
    persistence_window: int = 3,
) -> pd.DataFrame:
    """Append wind-driven fire spread risk.

    - ``dir_persistence`` : 1 - sd(sin(dir), cos(dir)) over the past
      ``persistence_window`` days (0 = highly variable, 1 = highly persistent).
    - ``wind_spread`` : ``speed * dir_persistence`` -- strong steady winds score
      highest; gusty-but-rotating winds score lower.
    """
    if speed_col not in df.columns or dir_col not in df.columns:
        logger.warning("Skipping wind_spread: %s or %s missing", speed_col, dir_col)
        return df

    df = df.sort_values([group_col, "date"]).reset_index(drop=True).copy()

    rad = np.deg2rad(df[dir_col])
    df["_dir_sin"] = np.sin(rad)
    df["_dir_cos"] = np.cos(rad)

    # Rolling SDs of the direction components, per city, past-only
    def _rolling_sd(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(persistence_window, min_periods=1).std()

    sd_s = df.groupby(group_col, sort=False)["_dir_sin"].transform(_rolling_sd)
    sd_c = df.groupby(group_col, sort=False)["_dir_cos"].transform(_rolling_sd)

    # Combined spread of the unit-vector components -> higher = less persistent
    # In extreme cases sd -> ~0.707 for uniform random, so clip + invert to 0..1
    combined = np.hypot(sd_s.fillna(0.707), sd_c.fillna(0.707))
    persistence = (1.0 - np.clip(combined / 0.707, 0.0, 1.0)).astype(np.float32)

    df["dir_persistence"] = persistence
    df["wind_spread"] = (df[speed_col].fillna(0).astype(np.float32) * persistence).astype(np.float32)
    df = df.drop(columns=["_dir_sin", "_dir_cos"])
    logger.info("Added dir_persistence and wind_spread (window=%dd)", persistence_window)
    return df


# ============================================================================
# 5. NDVI-based fuel dryness
# ============================================================================

def add_fuel_dryness(
    df: pd.DataFrame,
    ndvi_df: pd.DataFrame,
    group_col: str = "City",
) -> pd.DataFrame:
    """Merge NDVI, then append ``ndvi_anomaly`` = NDVI - DOY climatology mean.

    Negative values imply vegetation is drier / more senescent than usual for
    the season, which raises fire risk.
    """
    if ndvi_df.empty:
        logger.warning("NDVI data empty; skipping fuel dryness")
        df["ndvi"] = np.nan
        df["ndvi_anomaly"] = np.nan
        return df

    ndvi = ndvi_df.copy()
    ndvi["date"] = pd.to_datetime(ndvi["date"])
    if ndvi["date"].dt.tz is not None:
        ndvi["date"] = ndvi["date"].dt.tz_localize(None)
    ndvi["date"] = ndvi["date"].dt.floor("D")

    # Per-city DOY climatology -- interpolate across all 366 DOYs so every
    # date gets a defined anomaly value (NDVI is 16-day composite, so direct
    # DOY matching leaves gaps)
    ndvi["doy"] = ndvi["date"].dt.dayofyear
    clim_raw = ndvi.groupby([group_col, "doy"])["NDVI"].mean().reset_index()

    # Build a dense per-city DOY grid 1..366 and interpolate
    dense_rows: List[pd.DataFrame] = []
    for city, g in clim_raw.groupby(group_col, sort=False):
        full = pd.DataFrame({"doy": range(1, 367)})
        full[group_col] = city
        merged = full.merge(g[["doy", "NDVI"]], on="doy", how="left")
        # Interpolate linearly, then back/forward fill endpoints
        merged["NDVI"] = merged["NDVI"].interpolate(method="linear", limit_direction="both")
        dense_rows.append(merged)
    clim = pd.concat(dense_rows, ignore_index=True).rename(columns={"NDVI": "ndvi_clim"})

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["doy"] = df["date"].dt.dayofyear

    df = df.merge(ndvi[[group_col, "date", "NDVI"]], on=[group_col, "date"], how="left")
    df = df.merge(clim, on=[group_col, "doy"], how="left")
    df = df.rename(columns={"NDVI": "ndvi"})

    # Forward-fill NDVI per city (daily NDVI can be cloudy/missing)
    df["ndvi"] = df.groupby(group_col)["ndvi"].transform(lambda s: s.ffill().bfill())
    df["ndvi_anomaly"] = df["ndvi"] - df["ndvi_clim"]
    df = df.drop(columns=["doy", "ndvi_clim"])

    logger.info("Added ndvi + ndvi_anomaly (%.1f%% non-null after ffill)",
                100 * df["ndvi"].notna().mean())
    return df


# ============================================================================
# 6. Human-activity static features
# ============================================================================

def add_human_activity(
    df: pd.DataFrame,
    roads: pd.DataFrame,
    population: pd.DataFrame,
    group_col: str = "City",
) -> pd.DataFrame:
    """Join static per-city road density and per-year population density.

    Output adds ``human_access_road_meters``, ``pop_density``, and
    ``human_activity_score`` (log-normalised product, unitless).
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    # Road density: static per city
    df = df.merge(roads[[group_col, "human_access_road_meters"]],
                  on=group_col, how="left")

    # Population: per city per year
    if {"City", "Year", "Pop_Density"}.issubset(population.columns):
        pop = population.rename(columns={"Year": "year", "Pop_Density": "pop_density"})[
            [group_col, "year", "pop_density"]
        ]
        df = df.merge(pop, on=[group_col, "year"], how="left")
        df["pop_density"] = df.groupby(group_col)["pop_density"].transform(
            lambda s: s.ffill().bfill())
    else:
        df["pop_density"] = np.nan
        logger.warning("Population frame missing expected columns")

    # Normalised activity score: log1p both, multiply, then standardise
    rd_log = np.log1p(df["human_access_road_meters"].fillna(0))
    pd_log = np.log1p(df["pop_density"].fillna(0))
    combo = rd_log * pd_log
    std = combo.std() or 1.0
    df["human_activity_score"] = ((combo - combo.mean()) / std).astype(np.float32)

    df = df.drop(columns=["year"])
    logger.info("Added human_access_road_meters, pop_density, human_activity_score")
    return df


# ============================================================================
# 7. Lightning climatology
# ============================================================================

def add_lightning_climatology(
    df: pd.DataFrame,
    lightning: pd.DataFrame,
    cities: pd.DataFrame,
    group_col: str = "City",
) -> pd.DataFrame:
    """Attach ``lightning_thunder_hours`` -- avg thunder-hours in the city's
    lightning grid cell for the row's calendar month. Natural-ignition proxy.
    """
    if lightning.empty:
        df["lightning_thunder_hours"] = np.nan
        return df

    # For each city, find the nearest grid point and compute monthly mean TH
    lat_c = cities["Latitude"].to_numpy()
    lon_c = cities["Longitude"].to_numpy()
    city_lookup = dict(zip(cities["City"].to_list(), zip(lat_c.tolist(), lon_c.tolist())))

    rows: List[Dict] = []
    for city, (clat, clon) in city_lookup.items():
        d = haversine_km(clat, clon,
                         lightning["lat"].to_numpy(),
                         lightning["lon"].to_numpy())
        # Take all grid points within ~50km, average by month
        near = lightning[d <= 50.0]
        if near.empty:
            # Fall back to the nearest single cell
            near = lightning.iloc[[int(np.argmin(d))]]
        monthly = near.groupby("month")["thunder_hours"].mean().reset_index()
        for _, r in monthly.iterrows():
            rows.append({"City": city, "month": int(r["month"]),
                         "lightning_thunder_hours": float(r["thunder_hours"])})

    clim = pd.DataFrame(rows)
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df = df.merge(clim, on=[group_col, "month"], how="left")
    df = df.drop(columns=["month"])
    logger.info("Added lightning_thunder_hours (monthly per-city climatology)")
    return df


# ============================================================================
# 8. Orchestrator
# ============================================================================

DEFAULT_WEATHER_COLS: List[str] = [
    "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
    "relative_humidity_2m_mean", "dew_point_2m_mean",
    "precipitation_sum", "rain_sum",
    "cloud_cover_mean", "vapour_pressure_deficit_mean",
    "wind_speed_10m_mean", "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m",
    "soil_temperature_0_to_7cm_mean",
    "sunshine_duration_sum", "shortwave_radiation_mean",
]


def build_wildfire_features(
    weather_daily_path: Optional[Path] = None,
    firms_path: Optional[Path] = None,
    ndvi_path: Optional[Path] = None,
    roads_path: Optional[Path] = None,
    population_path: Optional[Path] = None,
    lightning_path: Optional[Path] = None,
    cities_path: Optional[Path] = None,
    radius_km: float = FIRE_RADIUS_KM,
    save: bool = True,
    output_name: str = "wildfire_features",
) -> pd.DataFrame:
    """Full feature-build pipeline.

    Input sources default to ``data/interim/*.csv`` from Phase 1+2.
    Output goes to ``data/processed/<output_name>.csv``.
    """
    weather_daily_path = weather_daily_path or (INTERIM_DIR / "weather_daily_clean.csv")
    firms_path        = firms_path        or (INTERIM_DIR / "firms.csv")
    ndvi_path         = ndvi_path         or (INTERIM_DIR / "ndvi.csv")
    roads_path        = roads_path        or (INTERIM_DIR / "roads.csv")
    population_path   = population_path   or (INTERIM_DIR / "population.csv")
    lightning_path    = lightning_path    or (INTERIM_DIR / "lightning.csv")
    cities_path       = cities_path       or (INTERIM_DIR / "cities_reference.csv")

    logger.info("=" * 72)
    logger.info("PHASE 4.1 - Wildfire feature engineering")
    logger.info("=" * 72)

    # --- Load ---
    weather = pd.read_csv(weather_daily_path, parse_dates=["date"])
    if weather["date"].dt.tz is not None:
        weather["date"] = weather["date"].dt.tz_localize(None)
    weather["date"] = weather["date"].dt.floor("D")

    cities_all = pd.read_csv(cities_path)
    modelled = weather["City"].unique().tolist()
    cities = cities_all[cities_all["City"].isin(modelled)].copy()
    logger.info("Cities in weather: %s", modelled)

    # Keep only weather columns we'll use, plus meta
    keep = ["City", "date"] + [c for c in DEFAULT_WEATHER_COLS if c in weather.columns]
    weather = weather[keep]

    # --- Fire target ---
    firms = pd.read_csv(firms_path, parse_dates=["acq_date"])
    fire_days = count_firms_within_radius(firms, cities, radius_km=radius_km)
    fire_days["date"] = pd.to_datetime(fire_days["date"]).dt.floor("D")
    # Left-join onto complete city-day grid so zero-fire days are filled
    city_days = weather[["City", "date"]].drop_duplicates()
    fire_aligned = _fill_zero_fire_days(fire_days, city_days)

    df = weather.merge(fire_aligned, on=["City", "date"], how="left")
    for c in ("fire_count", "fire_frp_total", "fire_frp_max", "fire_occurred"):
        df[c] = df[c].fillna(0)
    df["fire_occurred"] = df["fire_occurred"].astype(np.int8)
    df["fire_count"] = df["fire_count"].astype(np.int32)
    # Restrict full-target rows to FIRMS coverage period (2020-2024); beyond that, NaN
    coverage_end = firms["acq_date"].max().floor("D")
    beyond = df["date"] > coverage_end
    df.loc[beyond, ["fire_occurred", "fire_count", "fire_frp_total", "fire_frp_max"]] = np.nan
    logger.info("Labeled rows: %d (fire-days: %d, %.2f%% positive)",
                (~beyond).sum(),
                int(df.loc[~beyond, "fire_occurred"].sum()),
                100 * df.loc[~beyond, "fire_occurred"].mean())

    # --- Derived fire-science features ---
    df = add_drought_index(df)
    df = add_heatwave_indicator(df)
    df = add_wind_spread_factor(df)

    # --- NDVI ---
    try:
        ndvi = pd.read_csv(ndvi_path)
        df = add_fuel_dryness(df, ndvi)
    except FileNotFoundError:
        logger.warning("NDVI file missing; skipping fuel dryness")

    # --- Human activity ---
    try:
        roads = pd.read_csv(roads_path)
    except FileNotFoundError:
        roads = pd.DataFrame(columns=["City", "human_access_road_meters"])
    try:
        pop = pd.read_csv(population_path)
    except FileNotFoundError:
        pop = pd.DataFrame(columns=["City", "Year", "Pop_Density"])
    df = add_human_activity(df, roads, pop)

    # --- Lightning ---
    try:
        lightning = pd.read_csv(lightning_path)
        df = add_lightning_climatology(df, lightning, cities)
    except FileNotFoundError:
        logger.warning("Lightning file missing; skipping")

    # --- Calendar features for seasonality ---
    d = pd.to_datetime(df["date"])
    df["month"] = d.dt.month.astype(np.int8)
    df["doy"] = d.dt.dayofyear.astype(np.int16)
    df["year"] = d.dt.year.astype(np.int16)

    # Cleanup / final sort
    df = df.sort_values(["City", "date"]).reset_index(drop=True)
    logger.info("Output: %s rows x %d cols", f"{len(df):,}", df.shape[1])

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / f"{output_name}.csv"
        df.to_csv(out, index=False)
        logger.info("Saved -> %s (%.2f MB)", out, out.stat().st_size / 1024 / 1024)

    return df


# ============================================================================
# 9. Helpers for modelling phase
# ============================================================================

TARGET_COLUMNS: List[str] = ["fire_occurred", "fire_count", "fire_frp_total", "fire_frp_max"]


def predictor_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of predictor columns (everything except meta + targets)."""
    meta = {"City", "date"}
    out = [c for c in df.columns
           if c not in meta and c not in TARGET_COLUMNS
           and pd.api.types.is_numeric_dtype(df[c])]
    return out


if __name__ == "__main__":
    build_wildfire_features()
