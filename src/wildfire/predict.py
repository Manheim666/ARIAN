"""
Wildfire 30-day risk prediction (Phase 4 - Step 3).

Consumes the weather forecast (Phase 2.4) and the trained wildfire models
(Phase 4.2), and emits a per-city daily wildfire-risk dataset covering the
next 30 days.

Design
------
The wildfire feature matrix (Phase 4.1) was built from daily weather
history. To predict forward, we must construct the **same feature schema**
from the 30-day weather forecast and then score it with the trained
classifier + regressor.

Order of operations:

1. Load the forecast (long -> wide: one row per (City, forecast_date) with
   a column per target).
2. Build the most-recent historical tail the forecast extends (needed so that
   running statistics like `drought_index` and `dry_days_run` have warm-up
   context).
3. Concatenate forecast onto the end of the tail with the forecast-derived
   columns filling in for observed values.
4. Run the feature-engineering pipeline (notebook 07 logic) in "prediction
   mode" that keeps the derived features flowing through the forecast days.
5. Score with the classifier (probability of fire) and regressor (expected
   count).
6. Emit ``data/processed/wildfire_risk_forecast.csv``.

Public entry point: :func:`predict_wildfire_risk_30day`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    INTERIM_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
)
from src.utils.logging_utils import get_logger
from src.wildfire.features import (
    add_drought_index,
    add_fuel_dryness,
    add_heatwave_indicator,
    add_human_activity,
    add_lightning_climatology,
    add_wind_spread_factor,
)

logger = get_logger(__name__)


# ============================================================================
# Forecast-to-features assembly
# ============================================================================

# Forecast target keys -> daily-history column names (same mapping as Phase 3)
FORECAST_TO_DAILY_COL: Dict[str, str] = {
    "temperature_2m":     "temperature_2m_mean",
    "wind_speed_10m":     "wind_speed_10m_mean",
    "wind_direction_10m": "wind_direction_10m",
    "rain":               "rain_sum",
    "precipitation":      "precipitation_sum",
}


def _forecast_to_wide_daily(forecast: pd.DataFrame) -> pd.DataFrame:
    """Pivot forecast to wide + rename target columns to daily schema."""
    wide = (
        forecast
        .pivot_table(index=["City", "forecast_date"], columns="target", values="y_pred")
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.rename(columns={"forecast_date": "date",
                                 **FORECAST_TO_DAILY_COL})
    return wide


def _fill_missing_forecast_cols(
    fcst_wide: pd.DataFrame,
    history_daily: pd.DataFrame,
) -> pd.DataFrame:
    """The weather forecast only predicts 5 targets. Many other columns the
    wildfire feature pipeline expects (e.g. T_max, T_min, humidity, soil T)
    are missing. Fill these with per-city DOY climatology from history so
    the derived features still compute sensibly.
    """
    fcst_wide = fcst_wide.copy()
    fcst_wide["doy"] = pd.to_datetime(fcst_wide["date"]).dt.dayofyear

    # Climatology for every numeric col not already in the forecast
    present = set(fcst_wide.columns)
    fill_cols = [c for c in history_daily.columns
                 if c not in ("City", "date") and c not in present
                 and pd.api.types.is_numeric_dtype(history_daily[c])]

    h = history_daily.copy()
    h["doy"] = pd.to_datetime(h["date"]).dt.dayofyear
    clim = h.groupby(["City", "doy"])[fill_cols].mean().reset_index()

    merged = fcst_wide.merge(clim, on=["City", "doy"], how="left")

    # Also: the forecast may not predict T_max / T_min explicitly. If the
    # downstream feature builder needs them, we approximate:
    #   - T_max ~ mean + (hist_mean_diff between max and mean for that DOY)
    # but since we merged climatology above, T_max etc. are already filled.
    merged = merged.drop(columns=["doy"])
    logger.info("Filled %d missing forecast columns via DOY climatology", len(fill_cols))
    return merged


def build_forecast_feature_frame(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    ndvi: pd.DataFrame,
    roads: pd.DataFrame,
    population: pd.DataFrame,
    lightning: pd.DataFrame,
    cities: pd.DataFrame,
    history_tail_days: int = 90,
) -> pd.DataFrame:
    """Construct the prediction-time feature table.

    Parameters
    ----------
    history_tail_days
        How many days of recent history to prepend to the forecast so that
        stateful features (drought index, heatwave run-length, wind-spread
        persistence) have warm-up context.
    """
    fcst_wide = _forecast_to_wide_daily(forecast)
    fcst_wide = _fill_missing_forecast_cols(fcst_wide, history)

    # Tail history (pre-anchor)
    anchor = pd.to_datetime(forecast["anchor_date"].iloc[0])
    tail = history[(history["date"] <= anchor) &
                   (history["date"] >= anchor - pd.Timedelta(days=history_tail_days))]
    tail = tail.copy()

    # Harmonise columns: keep the intersection, plus the City/date keys
    common_cols = ["City", "date"] + [c for c in fcst_wide.columns
                                      if c in tail.columns and c not in ("City", "date")]
    tail = tail[common_cols]
    fcst = fcst_wide[common_cols]

    # Mark which rows are forecast
    tail = tail.assign(is_forecast=0)
    fcst = fcst.assign(is_forecast=1)

    combined = (
        pd.concat([tail, fcst], ignore_index=True)
          .sort_values(["City", "date"])
          .reset_index(drop=True)
    )
    logger.info("Combined tail + forecast: %d rows (%d tail + %d forecast)",
                len(combined), len(tail), len(fcst))

    # Apply the same derived-feature builders as the training pipeline
    combined = add_drought_index(combined)
    combined = add_heatwave_indicator(combined)
    combined = add_wind_spread_factor(combined)
    combined = add_fuel_dryness(combined, ndvi)
    combined = add_human_activity(combined, roads, population)
    combined = add_lightning_climatology(combined, lightning, cities)

    # Calendar features
    d = pd.to_datetime(combined["date"])
    combined["month"] = d.dt.month.astype(np.int8)
    combined["doy"] = d.dt.dayofyear.astype(np.int16)
    combined["year"] = d.dt.year.astype(np.int16)

    # Restrict output to the forecast window only
    forecast_only = combined[combined["is_forecast"] == 1].drop(columns=["is_forecast"])
    forecast_only = forecast_only.reset_index(drop=True)
    logger.info("Forecast feature frame: %s", forecast_only.shape)
    return forecast_only


# ============================================================================
# Align columns to the model's training schema
# ============================================================================

def _align_columns_to_model(X: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Reorder / add / drop columns so X matches what the model was trained on.

    Handles:
      - plain sklearn estimators (with ``feature_names_in_``)
      - our bundled dicts ``{"scaler": ..., "clf": ...}`` — introspect the scaler
      - CalibratedClassifierCV wrapping a base estimator
    """
    cols = None

    if isinstance(model, dict):
        # Bundled: try scaler first, else inner clf/reg
        for key in ("scaler", "clf", "reg"):
            if key in model and hasattr(model[key], "feature_names_in_"):
                cols = list(model[key].feature_names_in_)
                break
        if cols is None:
            # Fallback: use n_features_in_ to detect width, but we can't know names.
            width = None
            for key in ("scaler", "clf", "reg"):
                if key in model and hasattr(model[key], "n_features_in_"):
                    width = int(model[key].n_features_in_)
                    break
            if width is None:
                logger.warning("Cannot introspect bundled model; using X as-is")
                return X
            if X.shape[1] == width:
                return X
            # Try to trim X to exactly `width` numeric cols by dropping extras
            # that the bundle likely didn't see (added city_* dummies, etc.)
            raise ValueError(
                f"Bundled model expects {width} features, got {X.shape[1]}. "
                "The bundle has no feature_names_in_, so the correct mapping "
                "is ambiguous. Re-train the model with feature_names preserved."
            )

    elif hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
        cols = list(model.estimator.feature_names_in_)
    elif hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_):
        first = model.calibrated_classifiers_[0].estimator
        if hasattr(first, "feature_names_in_"):
            cols = list(first.feature_names_in_)

    if cols is None:
        logger.warning("Cannot introspect model features; using X as-is")
        return X

    # Add missing columns as zeros, drop extras, reorder
    out = pd.DataFrame(index=X.index)
    for c in cols:
        out[c] = X[c] if c in X.columns else 0.0
    missing = set(cols) - set(X.columns)
    extra = set(X.columns) - set(cols)
    if missing:
        logger.info("  Added %d zero-filled missing columns: %s...",
                    len(missing), sorted(list(missing))[:5])
    if extra:
        logger.info("  Dropped %d extra columns", len(extra))
    return out


# ============================================================================
# Main entry point
# ============================================================================

def predict_wildfire_risk_30day(
    forecast_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    save: bool = True,
    output_name: str = "wildfire_risk_forecast",
) -> pd.DataFrame:
    """Apply the trained wildfire classifier + regressor to the 30-day forecast.

    Output columns::

        City                  str
        anchor_date           datetime
        forecast_date         datetime
        horizon_days          int
        fire_probability      float in [0, 1]   (calibrated classifier prob)
        risk_category         str               ("low" / "moderate" / "high" / "very_high")
        expected_fire_count   float >= 0        (regressor prediction)
    """
    forecast_path = forecast_path or (PROCESSED_DIR / "weather_forecast.csv")
    history_path  = history_path  or (INTERIM_DIR   / "weather_daily_clean.csv")
    models_dir    = models_dir    or (MODELS_DIR    / "wildfire")

    logger.info("=" * 72)
    logger.info("PHASE 4.3 - 30-day wildfire risk prediction")
    logger.info("=" * 72)

    # --- Load inputs ---
    forecast = pd.read_csv(forecast_path, parse_dates=["anchor_date", "forecast_date"])
    if forecast["forecast_date"].dt.tz is not None:
        forecast["forecast_date"] = forecast["forecast_date"].dt.tz_localize(None)
    if forecast["anchor_date"].dt.tz is not None:
        forecast["anchor_date"] = forecast["anchor_date"].dt.tz_localize(None)

    history = pd.read_csv(history_path, parse_dates=["date"])
    if history["date"].dt.tz is not None:
        history["date"] = history["date"].dt.tz_localize(None)

    ndvi = pd.read_csv(INTERIM_DIR / "ndvi.csv")
    roads = pd.read_csv(INTERIM_DIR / "roads.csv")
    population = pd.read_csv(INTERIM_DIR / "population.csv")
    lightning = pd.read_csv(INTERIM_DIR / "lightning.csv")
    cities_all = pd.read_csv(INTERIM_DIR / "cities_reference.csv")
    cities = cities_all[cities_all["City"].isin(history["City"].unique())]

    # --- Build features for the 30-day window ---
    feats = build_forecast_feature_frame(
        forecast=forecast, history=history, ndvi=ndvi,
        roads=roads, population=population, lightning=lightning,
        cities=cities,
    )

    # --- Load best classifier + regressor ---
    best = json.loads((models_dir / "best.json").read_text())
    clf_path = Path(best["classifier"]["path"])
    reg_path = Path(best["regressor"]["path"])
    logger.info("Loading classifier: %s", clf_path.name)
    logger.info("Loading regressor : %s", reg_path.name)
    clf_model = joblib.load(clf_path)
    reg_model = joblib.load(reg_path)

    # Add one-hot City columns (same as training)
    for city in cities["City"]:
        feats[f"city_{city}"] = (feats["City"] == city).astype(np.int8)

    X = feats.drop(columns=["City", "date"])

    # --- Classifier ---
    X_clf = _align_columns_to_model(X, clf_model)
    if isinstance(clf_model, dict):  # bundled logistic
        Xs = clf_model["scaler"].transform(X_clf.fillna(0.0).values)
        probs = clf_model["clf"].predict_proba(Xs)[:, 1]
    else:
        probs = clf_model.predict_proba(X_clf)[:, 1]

    # --- Regressor ---
    X_reg = _align_columns_to_model(X, reg_model)
    if isinstance(reg_model, dict):  # bundled ridge
        Xs = reg_model["scaler"].transform(X_reg.fillna(0.0).values)
        counts = np.expm1(reg_model["reg"].predict(Xs))
    else:
        counts = np.expm1(reg_model.predict(X_reg))
    counts = np.clip(counts, 0, None)

    # --- Assemble output ---
    def _risk_label(p: float) -> str:
        if not np.isfinite(p):   return "unknown"
        if p < 0.10:             return "low"
        if p < 0.25:             return "moderate"
        if p < 0.50:             return "high"
        return "very_high"

    out = pd.DataFrame({
        "City": feats["City"].values,
        "forecast_date": feats["date"].values,
        "fire_probability": probs,
        "expected_fire_count": counts,
    })
    out["risk_category"] = out["fire_probability"].apply(_risk_label)
    out["anchor_date"] = forecast["anchor_date"].iloc[0]
    out["horizon_days"] = (out["forecast_date"] - out["anchor_date"]).dt.days
    out = out[["City", "anchor_date", "forecast_date", "horizon_days",
               "fire_probability", "risk_category", "expected_fire_count"]]
    out = out.sort_values(["City", "forecast_date"]).reset_index(drop=True)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        p = PROCESSED_DIR / f"{output_name}.csv"
        out.to_csv(p, index=False)
        logger.info("Saved -> %s (%.2f KB)", p, p.stat().st_size / 1024)

    logger.info("Risk forecast ready: %d rows, %d cities, horizons %d..%d",
                len(out), out["City"].nunique(),
                int(out["horizon_days"].min()), int(out["horizon_days"].max()))
    logger.info("=" * 72)
    return out


if __name__ == "__main__":
    predict_wildfire_risk_30day()
