from __future__ import annotations

import numpy as np
import pandas as pd


def add_features_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.floor("D")

    out["month"] = out["Date"].dt.month
    out["day_of_year"] = out["Date"].dt.dayofyear

    if "City" in out.columns:
        out = out.sort_values(["City", "Date"])

    # Core lags + rolling stats for temperature
    if "Temperature_C" in out.columns and "City" in out.columns:
        s = out.groupby("City")["Temperature_C"]
        out["lag_1"] = s.shift(1)
        out["lag_7"] = s.shift(7)
        out["rolling_mean_7"] = s.shift(1).rolling(7, min_periods=3).mean().reset_index(level=0, drop=True)
        out["rolling_std_7"] = s.shift(1).rolling(7, min_periods=3).std().reset_index(level=0, drop=True)

        # Extreme flags based on per-city historical quantiles
        q_heat = s.transform(lambda x: x.quantile(0.95))
        out["heatwave"] = (out["Temperature_C"] >= q_heat).astype(int)

    if "Precipitation_mm" in out.columns and "City" in out.columns:
        p = out.groupby("City")["Precipitation_mm"]
        q_rain = p.transform(lambda x: x.quantile(0.95))
        out["heavy_rain"] = (out["Precipitation_mm"] >= q_rain).astype(int)

    if "Wind_Speed_kmh" in out.columns and "City" in out.columns:
        w = out.groupby("City")["Wind_Speed_kmh"]
        q_wind = w.transform(lambda x: x.quantile(0.95))
        out["high_wind"] = (out["Wind_Speed_kmh"] >= q_wind).astype(int)

    # Wildfire-related features (expects fires already merged)
    if "fire_count" in out.columns:
        out["fire_count"] = pd.to_numeric(out["fire_count"], errors="coerce").fillna(0).astype(int)
    if "Fire_Occurred" in out.columns:
        out["Fire_Occurred"] = pd.to_numeric(out["Fire_Occurred"], errors="coerce").fillna(0).astype(int)

    if "max_frp" in out.columns:
        out["fire_intensity_proxy"] = pd.to_numeric(out["max_frp"], errors="coerce")
    elif "mean_brightness" in out.columns:
        out["fire_intensity_proxy"] = pd.to_numeric(out["mean_brightness"], errors="coerce")
    else:
        out["fire_intensity_proxy"] = np.nan

    return out
