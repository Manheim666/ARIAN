"""
Weather forecasting (Phase 2 - Step 4).

Produces the 30-day forward forecast the downstream Climate and Wildfire
phases will consume.

Strategy
--------
We train one model **per (target, horizon)** for a sparse set of horizons
``{1, 3, 7, 14, 30}`` and **piecewise-linearly interpolate** the predictions
across the intermediate horizons. This is the production-standard
"direct multi-horizon" approach: every prediction uses exclusively observed
features (no recursive feedback of model error), keeping the 30-day horizon
well-calibrated.

Two public entry points:

* :func:`train_multi_horizon_models` — fits the sparse horizon ladder for
  every target and persists models to ``models/weather/<target>/h{H}/<algo>.joblib``.
* :func:`make_30day_forecast` — loads those models, emits a tidy
  ``data/processed/weather_forecast.csv`` with columns
  ``City, date, target, horizon_days, y_pred``.

Design principles
-----------------
* **Deterministic.** Given the same feature matrix and seed, outputs are
  bitwise identical.
* **Defensive.** If a horizon's model is missing, we fall back to the nearest
  trained horizon (with a warning) rather than crash.
* **Feature-date-anchored.** The "forecast date" is stated as the anchor day
  (the last observed day) + horizon, so downstream joins are unambiguous.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import FORECAST_TARGETS, MODELS_DIR, PROCESSED_DIR
from src.utils.logging_utils import get_logger
from src.weather.features import DAILY_TARGET_COLUMNS
from src.weather.train import (
    prepare_training_frames,
    train_hgbr,
    train_ridge,
    evaluate_regression,
)

logger = get_logger(__name__)

# Which horizons we actually train models for
DEFAULT_HORIZON_ANCHORS: Tuple[int, ...] = (1, 3, 7, 14, 30)
# Full range we report in the output dataset
DEFAULT_OUTPUT_HORIZONS: Tuple[int, ...] = tuple(range(1, 31))


# ============================================================================
# 1. Multi-horizon training
# ============================================================================

def _model_path(target: str, horizon: int, algo: str) -> Path:
    """Canonical on-disk location for a (target, horizon, algo) artefact."""
    return MODELS_DIR / "weather" / target / f"h{horizon}" / f"{algo}.joblib"


def train_multi_horizon_models(
    features_path: Optional[Path] = None,
    targets: Optional[List[str]] = None,
    horizons: Sequence[int] = DEFAULT_HORIZON_ANCHORS,
    algo: str = "hgbr",
    test_start: str = "2025-01-01",
) -> pd.DataFrame:
    """Train one ``algo`` per (target, horizon). Returns a metrics DataFrame.

    Sparse horizon ladder reduces compute from 5 targets x 30 horizons = 150
    models to 5 x 5 = 25, while still letting the final forecast track the
    actual 30-day accuracy curve (typically monotone for these targets).
    """
    features_path = features_path or (PROCESSED_DIR / "weather_features.csv")
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} - run notebook 03 first")

    logger.info("=" * 72)
    logger.info("PHASE 2.4 - Multi-horizon model training")
    logger.info("Horizons: %s   Algo: %s", list(horizons), algo)
    logger.info("=" * 72)
    feats = pd.read_csv(features_path, parse_dates=["date"])
    if feats["date"].dt.tz is not None:
        feats["date"] = feats["date"].dt.tz_localize(None)

    targets = targets or FORECAST_TARGETS
    trainers = {"hgbr": train_hgbr, "ridge": train_ridge}
    if algo not in trainers:
        raise ValueError(f"Unsupported algo {algo!r}; choose from {list(trainers)}")

    rows: List[Dict[str, Any]] = []
    for target in targets:
        circular = target == "wind_direction_10m"
        logger.info("--- %s (circular=%s) ---", target, circular)
        for h in horizons:
            try:
                tf = prepare_training_frames(feats, target=target,
                                             test_start=test_start, horizon=h)
                model, pred = trainers[algo](tf)
                m = evaluate_regression(tf.y_test.values, pred, circular=circular)
                path = _model_path(target, h, algo)
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path)
                rows.append({"target": target, "horizon": h, "algo": algo, **m,
                             "model_path": str(path)})
                logger.info("  h=%-2d  MAE=%.3f  RMSE=%.3f",
                            h, m.get("MAE", np.nan), m.get("RMSE", np.nan))
            except Exception as exc:  # noqa: BLE001
                logger.error("  h=%d failed: %s", h, exc)

    summary = pd.DataFrame(rows)
    out = MODELS_DIR / "weather" / "multi_horizon_summary.csv"
    summary.to_csv(out, index=False)
    logger.info("Saved summary -> %s", out)
    return summary


# ============================================================================
# 2. Anchor-day feature slice
# ============================================================================

def get_anchor_features(
    features_path: Optional[Path] = None,
    anchor_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Return the feature row for every city at ``anchor_date``.

    If ``anchor_date`` is ``None``, use the last available observation date
    in the feature frame.
    """
    features_path = features_path or (PROCESSED_DIR / "weather_features.csv")
    feats = pd.read_csv(features_path, parse_dates=["date"])
    if feats["date"].dt.tz is not None:
        feats["date"] = feats["date"].dt.tz_localize(None)

    if anchor_date is None:
        anchor = feats["date"].max()
    else:
        anchor = pd.Timestamp(anchor_date)

    anchor_df = feats[feats["date"] == anchor].copy()
    if anchor_df.empty:
        raise ValueError(f"No rows at anchor_date={anchor.date()}")
    logger.info("Anchor date: %s (%d cities)", anchor.date(), len(anchor_df))
    return anchor_df, anchor


# ============================================================================
# 3. 30-day forecast assembly
# ============================================================================

def _load_horizon_ladder(
    target: str,
    horizons: Sequence[int],
    algo: str,
) -> Dict[int, Any]:
    """Load trained models for this target at every horizon in ``horizons``."""
    out: Dict[int, Any] = {}
    for h in horizons:
        p = _model_path(target, h, algo)
        if p.exists():
            out[h] = joblib.load(p)
        else:
            logger.warning("  model missing: %s -> fallback will be used", p)
    if not out:
        raise RuntimeError(f"No trained models for {target} / {algo}")
    return out


def _predict_at_horizon(
    ladder: Dict[int, Any],
    X_anchor: pd.DataFrame,
    target_horizon: int,
) -> np.ndarray:
    """Predict at ``target_horizon``, interpolating between neighbouring anchors."""
    anchors = sorted(ladder.keys())
    if target_horizon in ladder:
        return ladder[target_horizon].predict(X_anchor)

    # Before the first anchor -> use the first model
    if target_horizon < anchors[0]:
        return ladder[anchors[0]].predict(X_anchor)
    # After the last anchor -> use the last model
    if target_horizon > anchors[-1]:
        return ladder[anchors[-1]].predict(X_anchor)

    # Piecewise-linear between neighbouring anchors
    lower = max(a for a in anchors if a <= target_horizon)
    upper = min(a for a in anchors if a >= target_horizon)
    w = (target_horizon - lower) / (upper - lower)
    p_lo = ladder[lower].predict(X_anchor)
    p_hi = ladder[upper].predict(X_anchor)
    return (1.0 - w) * p_lo + w * p_hi


def make_30day_forecast(
    features_path: Optional[Path] = None,
    anchor_date: Optional[str] = None,
    targets: Optional[List[str]] = None,
    horizons: Sequence[int] = DEFAULT_OUTPUT_HORIZONS,
    trained_horizons: Sequence[int] = DEFAULT_HORIZON_ANCHORS,
    algo: str = "hgbr",
    save: bool = True,
    output_name: str = "weather_forecast",
) -> pd.DataFrame:
    """Emit the 30-day forecast dataset.

    Output schema::

        City           str
        anchor_date    datetime64[ns]    # last observed day
        forecast_date  datetime64[ns]    # anchor_date + horizon_days
        horizon_days   int
        target         str               # e.g. 'temperature_2m'
        y_pred         float

    All forecasts for a given anchor date and city share the same 30 rows
    per target, making downstream pivots trivial.
    """
    logger.info("=" * 72)
    logger.info("PHASE 2.4 - Building 30-day forecast dataset")
    logger.info("=" * 72)

    targets = targets or FORECAST_TARGETS
    anchor_df, anchor = get_anchor_features(features_path=features_path,
                                             anchor_date=anchor_date)
    cities = anchor_df["City"].tolist()

    all_rows: List[Dict[str, Any]] = []
    for target in targets:
        logger.info("--- %s ---", target)
        try:
            ladder = _load_horizon_ladder(target, trained_horizons, algo)
        except RuntimeError as exc:
            logger.error("Skipping %s: %s", target, exc)
            continue

        # Use the feature columns the first model in the ladder was trained on
        sample_model = next(iter(ladder.values()))
        if hasattr(sample_model, "feature_names_in_"):
            feat_cols = list(sample_model.feature_names_in_)
            X_anchor = anchor_df.reindex(columns=feat_cols)
        else:
            # Fall back to the numeric columns from anchor_df (minus meta)
            drop = {"City", "date"} | {c for c in anchor_df.columns if c.endswith("_outlier")}
            feat_cols = [c for c in anchor_df.columns
                         if c not in drop and pd.api.types.is_numeric_dtype(anchor_df[c])]
            X_anchor = anchor_df[feat_cols]

        for h in horizons:
            preds = _predict_at_horizon(ladder, X_anchor, h)
            forecast_date = anchor + pd.Timedelta(days=int(h))
            for city, p in zip(cities, preds):
                all_rows.append({
                    "City": city,
                    "anchor_date": anchor,
                    "forecast_date": forecast_date,
                    "horizon_days": int(h),
                    "target": target,
                    "y_pred": float(p),
                })

    forecast = pd.DataFrame(all_rows).sort_values(
        ["City", "target", "horizon_days"]
    ).reset_index(drop=True)
    logger.info("Forecast: %d rows (%d cities x %d horizons x %d targets)",
                len(forecast), len(cities), len(horizons), len(targets))

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / f"{output_name}.csv"
        forecast.to_csv(out, index=False)
        logger.info("Saved -> %s (%.2f MB)", out, out.stat().st_size / 1024 / 1024)

    return forecast


# ============================================================================
# 4. Wide-format pivot (convenience for downstream consumers)
# ============================================================================

def forecast_to_wide(forecast: pd.DataFrame) -> pd.DataFrame:
    """Pivot (City, forecast_date, target) -> one column per target.

    The result has one row per (City, forecast_date), ready for joins against
    other daily datasets.
    """
    wide = (
        forecast
        .pivot_table(index=["City", "anchor_date", "forecast_date", "horizon_days"],
                     columns="target", values="y_pred")
        .reset_index()
    )
    wide.columns.name = None
    return wide


if __name__ == "__main__":
    # Full pipeline: train the ladder, then produce the output dataset
    train_multi_horizon_models()
    make_30day_forecast()
