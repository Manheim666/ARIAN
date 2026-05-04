"""
Unit tests for src.features — Feature Engineering Utilities
=============================================================
Run with:  pytest tests/test_features.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (
    add_calendar_features,
    add_hourly_calendar,
    build_lag_features,
    build_rolling_features,
    compute_fwi_proxy,
    compute_vpd,
    compute_dew_point,
    compute_heat_index,
    add_wildfire_weather_features,
    add_historical_fire_features,
    add_vegetation_interactions,
    add_anomaly_features,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def daily_df() -> pd.DataFrame:
    """Minimal daily DataFrame with 2 cities × 60 days."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rows = []
    for city in ["CityA", "CityB"]:
        for d in dates:
            rows.append({
                "City": city,
                "Date": d,
                "Temperature_C_mean": np.random.uniform(5, 35),
                "Temperature_C_min": np.random.uniform(0, 20),
                "Temperature_C_max": np.random.uniform(20, 40),
                "Humidity_percent_mean": np.random.uniform(20, 90),
                "Rain_mm_sum": np.random.exponential(2),
                "Wind_Speed_kmh_mean": np.random.uniform(0, 50),
                "Pressure_hPa_mean": np.random.uniform(990, 1030),
                "Solar_Radiation_Wm2_mean": np.random.uniform(0, 300),
                "Soil_Temp_C_mean": np.random.uniform(5, 30),
                "Soil_Moisture_mean": np.random.uniform(0.1, 0.5),
                "Fire_Occurred": int(np.random.random() < 0.1),
                "NDVI": np.random.uniform(0.1, 0.8),
                "Trees_pct": np.random.uniform(0, 50),
            })
    df = pd.DataFrame(rows)
    df["Month"] = df["Date"].dt.month
    return df


@pytest.fixture
def hourly_df() -> pd.DataFrame:
    """Minimal hourly DataFrame — 2 cities × 48 hours."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-06-01", periods=48, freq="h")
    rows = []
    for city in ["CityA", "CityB"]:
        for ts in timestamps:
            rows.append({
                "City": city,
                "Timestamp": ts,
                "Date": ts.date(),
                "Temperature_C": np.random.uniform(15, 35),
                "Humidity_percent": np.random.uniform(20, 80),
                "Wind_Speed_kmh": np.random.uniform(0, 40),
                "Solar_Radiation_Wm2": np.random.uniform(0, 500),
                "Fire_Occurred": 0,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Calendar Features
# ═══════════════════════════════════════════════════════════════════════════

class TestCalendarFeatures:
    def test_adds_expected_columns(self, daily_df: pd.DataFrame) -> None:
        result = add_calendar_features(daily_df.copy())
        expected = {"Year", "Month", "DayOfYear", "DayOfWeek", "WeekOfYear",
                    "Month_sin", "Month_cos", "DoY_sin", "DoY_cos",
                    "DoW_sin", "DoW_cos", "Season", "is_summer",
                    "is_winter", "is_fire_season"}
        assert expected.issubset(set(result.columns))

    def test_cyclical_range(self, daily_df: pd.DataFrame) -> None:
        result = add_calendar_features(daily_df.copy())
        for col in ["Month_sin", "Month_cos", "DoY_sin", "DoY_cos",
                     "DoW_sin", "DoW_cos"]:
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0

    def test_season_flags_exclusive(self, daily_df: pd.DataFrame) -> None:
        result = add_calendar_features(daily_df.copy())
        # A January row should be winter, not summer
        jan_rows = result[result["Month"] == 1]
        assert (jan_rows["is_winter"] == 1).all()
        assert (jan_rows["is_summer"] == 0).all()

    def test_fire_season_months(self, daily_df: pd.DataFrame) -> None:
        result = add_calendar_features(daily_df.copy())
        fire_season = result[result["is_fire_season"] == 1]["Month"].unique()
        assert set(fire_season).issubset({5, 6, 7, 8, 9})

    def test_no_nans_introduced(self, daily_df: pd.DataFrame) -> None:
        result = add_calendar_features(daily_df.copy())
        new_cols = ["Year", "Month", "DayOfYear", "DayOfWeek",
                    "Month_sin", "Month_cos", "is_summer"]
        for col in new_cols:
            assert result[col].isna().sum() == 0


class TestHourlyCalendar:
    def test_adds_hour_columns(self, hourly_df: pd.DataFrame) -> None:
        result = add_hourly_calendar(hourly_df.copy())
        assert "Hour" in result.columns
        assert "Hour_sin" in result.columns
        assert "Hour_cos" in result.columns
        assert "is_daytime" in result.columns

    def test_is_daytime_logic(self, hourly_df: pd.DataFrame) -> None:
        result = add_hourly_calendar(hourly_df.copy())
        daytime = result[result["is_daytime"] == 1]
        assert daytime["Hour"].between(6, 20).all()

    def test_cyclical_range(self, hourly_df: pd.DataFrame) -> None:
        result = add_hourly_calendar(hourly_df.copy())
        assert result["Hour_sin"].min() >= -1.0
        assert result["Hour_cos"].max() <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Lag & Rolling Features
# ═══════════════════════════════════════════════════════════════════════════

class TestLagFeatures:
    def test_creates_correct_columns(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].copy()
        result = build_lag_features(city_a, ["Temperature_C_mean"], [1, 7])
        assert "Temperature_C_mean_lag1" in result.columns
        assert "Temperature_C_mean_lag7" in result.columns

    def test_lag1_shifts_correctly(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].sort_values("Date").copy()
        result = build_lag_features(city_a, ["Temperature_C_mean"], [1])
        # First row should be NaN (no previous value)
        assert pd.isna(result["Temperature_C_mean_lag1"].iloc[0])
        # Second row's lag should equal first row's original value
        assert result["Temperature_C_mean_lag1"].iloc[1] == pytest.approx(
            city_a.sort_values("Date")["Temperature_C_mean"].iloc[0])

    def test_skips_missing_variable(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].copy()
        result = build_lag_features(city_a, ["NONEXISTENT_COL"], [1])
        assert "NONEXISTENT_COL_lag1" not in result.columns

    def test_preserves_row_count(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].copy()
        result = build_lag_features(city_a, ["Temperature_C_mean"], [1, 3, 7])
        assert len(result) == len(city_a)


class TestRollingFeatures:
    def test_creates_mean_and_std(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].copy()
        result = build_rolling_features(city_a, ["Temperature_C_mean"], [7])
        assert "Temperature_C_mean_roll7_mean" in result.columns
        assert "Temperature_C_mean_roll7_std" in result.columns

    def test_shift_prevents_leakage(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].sort_values("Date").copy()
        result = build_rolling_features(city_a, ["Temperature_C_mean"], [3])
        # First row's rolling mean should be NaN (shifted by 1)
        assert pd.isna(result["Temperature_C_mean_roll3_mean"].iloc[0])

    def test_preserves_row_count(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].copy()
        result = build_rolling_features(city_a, ["Temperature_C_mean"], [3, 7])
        assert len(result) == len(city_a)


# ═══════════════════════════════════════════════════════════════════════════
# FWI Proxy
# ═══════════════════════════════════════════════════════════════════════════

class TestFWIProxy:
    def test_creates_all_fwi_columns(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].copy()
        result = compute_fwi_proxy(city_a)
        expected = {"FFMC_proxy", "DMC_proxy", "DC_proxy",
                    "ISI_proxy", "BUI_proxy", "FWI_proxy"}
        assert expected.issubset(set(result.columns))

    def test_ffmc_in_range(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].copy()
        result = compute_fwi_proxy(city_a)
        assert result["FFMC_proxy"].min() >= 0
        assert result["FFMC_proxy"].max() <= 100

    def test_fwi_non_negative(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].copy()
        result = compute_fwi_proxy(city_a)
        assert (result["FWI_proxy"] >= 0).all()

    def test_preserves_row_count(self, daily_df: pd.DataFrame) -> None:
        city_a = daily_df[daily_df["City"] == "CityA"].copy()
        result = compute_fwi_proxy(city_a)
        assert len(result) == len(city_a)


# ═══════════════════════════════════════════════════════════════════════════
# Wildfire-Specific Scalar Functions
# ═══════════════════════════════════════════════════════════════════════════

class TestVPD:
    def test_vpd_non_negative(self) -> None:
        temp = pd.Series([10, 20, 30, 40])
        rh = pd.Series([50, 60, 70, 80])
        result = compute_vpd(temp, rh)
        assert (result >= 0).all()

    def test_vpd_increases_with_temperature(self) -> None:
        rh = pd.Series([50, 50, 50])
        low = compute_vpd(pd.Series([10, 10, 10]), rh)
        high = compute_vpd(pd.Series([35, 35, 35]), rh)
        assert (high > low).all()

    def test_vpd_zero_at_saturation(self) -> None:
        temp = pd.Series([20.0])
        rh = pd.Series([100.0])
        result = compute_vpd(temp, rh)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-6)


class TestDewPoint:
    def test_dew_point_below_temp(self) -> None:
        temp = pd.Series([20.0, 30.0, 10.0])
        rh = pd.Series([50.0, 50.0, 50.0])
        dp = compute_dew_point(temp, rh)
        assert (dp < temp).all()

    def test_dew_point_equals_temp_at_100rh(self) -> None:
        temp = pd.Series([20.0])
        rh = pd.Series([100.0])
        dp = compute_dew_point(temp, rh)
        assert dp.iloc[0] == pytest.approx(20.0, abs=0.5)


class TestHeatIndex:
    def test_returns_temp_when_cool(self) -> None:
        temp = pd.Series([15.0, 20.0, 25.0])
        rh = pd.Series([50.0, 50.0, 50.0])
        result = compute_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(result, temp.values)

    def test_heat_index_higher_than_temp_when_hot(self) -> None:
        temp = pd.Series([35.0, 38.0])
        rh = pd.Series([70.0, 80.0])
        result = compute_heat_index(temp, rh)
        assert (result > temp.values).all()


# ═══════════════════════════════════════════════════════════════════════════
# Composite Feature Builders
# ═══════════════════════════════════════════════════════════════════════════

class TestWildfireWeatherFeatures:
    def test_adds_vpd_and_flags(self, daily_df: pd.DataFrame) -> None:
        result = add_wildfire_weather_features(daily_df.copy())
        expected = {"VPD_kPa", "Dew_Point_C", "Heat_Index",
                    "heatwave_flag", "low_humidity_flag", "high_wind_flag",
                    "dry_spell_flag", "temp_x_low_hum", "temp_x_wind",
                    "hot_dry_windy"}
        assert expected.issubset(set(result.columns))

    def test_flags_are_binary(self, daily_df: pd.DataFrame) -> None:
        result = add_wildfire_weather_features(daily_df.copy())
        for flag in ["heatwave_flag", "low_humidity_flag",
                     "high_wind_flag", "dry_spell_flag"]:
            assert set(result[flag].unique()).issubset({0, 1})

    def test_dry_days_streak_exists(self, daily_df: pd.DataFrame) -> None:
        result = add_wildfire_weather_features(daily_df.copy())
        assert "dry_days_streak" in result.columns
        assert (result["dry_days_streak"] >= 0).all()

    def test_no_nans_in_vpd(self, daily_df: pd.DataFrame) -> None:
        result = add_wildfire_weather_features(daily_df.copy())
        assert result["VPD_kPa"].isna().sum() == 0


class TestHistoricalFireFeatures:
    def test_adds_fire_count_columns(self, daily_df: pd.DataFrame) -> None:
        result = add_historical_fire_features(daily_df.copy())
        for w in [7, 14, 30, 90]:
            assert f"fire_count_{w}d" in result.columns

    def test_adds_days_since_last_fire(self, daily_df: pd.DataFrame) -> None:
        result = add_historical_fire_features(daily_df.copy())
        assert "days_since_last_fire" in result.columns
        assert result["days_since_last_fire"].max() <= 365

    def test_fire_counts_are_lagged(self, daily_df: pd.DataFrame) -> None:
        result = add_historical_fire_features(daily_df.copy())
        # First row for each city should have NaN or 0 fire count
        for city in result["City"].unique():
            first = result[result["City"] == city].sort_values("Date").iloc[0]
            val = first["fire_count_7d"]
            assert pd.isna(val) or val == 0

    def test_returns_unchanged_without_target(self) -> None:
        df = pd.DataFrame({"City": ["A"], "Date": ["2024-01-01"], "X": [1]})
        result = add_historical_fire_features(df)
        assert "fire_count_7d" not in result.columns

    def test_city_fire_rate_in_0_1(self, daily_df: pd.DataFrame) -> None:
        result = add_historical_fire_features(daily_df.copy())
        assert result["city_fire_rate"].between(0, 1).all()


class TestVegetationInteractions:
    def test_adds_interaction_columns(self, daily_df: pd.DataFrame) -> None:
        result = add_vegetation_interactions(daily_df.copy())
        assert "NDVI_x_drought" in result.columns
        assert "forest_x_dry_days" in result.columns
        assert "NDVI_x_VPD" in result.columns

    def test_preserves_row_count(self, daily_df: pd.DataFrame) -> None:
        result = add_vegetation_interactions(daily_df.copy())
        assert len(result) == len(daily_df)


class TestAnomalyFeatures:
    def test_adds_anomaly_columns(self, daily_df: pd.DataFrame) -> None:
        result = add_anomaly_features(daily_df.copy())
        assert "Temperature_C_mean_anomaly" in result.columns
        assert "Rain_mm_sum_anomaly" in result.columns

    def test_anomaly_mean_near_zero(self, daily_df: pd.DataFrame) -> None:
        result = add_anomaly_features(daily_df.copy(),
                                       variables=["Temperature_C_mean"])
        # Per city-month, anomaly should average to ~0
        group_mean = result.groupby(["City", "Month"])["Temperature_C_mean_anomaly"].mean()
        np.testing.assert_array_almost_equal(group_mean.values, 0, decimal=10)

    def test_custom_variables(self, daily_df: pd.DataFrame) -> None:
        result = add_anomaly_features(daily_df.copy(),
                                       variables=["Wind_Speed_kmh_mean"])
        assert "Wind_Speed_kmh_mean_anomaly" in result.columns
        assert "Temperature_C_mean_anomaly" not in result.columns

    def test_skips_missing_variable(self, daily_df: pd.DataFrame) -> None:
        result = add_anomaly_features(daily_df.copy(),
                                       variables=["NONEXISTENT"])
        assert "NONEXISTENT_anomaly" not in result.columns
