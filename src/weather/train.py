"""
Weather model training (Phase 2 - Step 3).

Trains and compares a ladder of regressors for each forecast target:

    Persistence  -> lower bound (y_pred[t] = y[t-1]); must be beaten
    Ridge        -> linear, regularised baseline
    HGBR         -> HistGradientBoostingRegressor (sklearn's fast GBM; XGBoost-class)
    XGBoost      -> if the `xgboost` package is installed
    SARIMA       -> if `statsmodels` is installed (classical seasonal AR-I-MA)
    Prophet      -> if `prophet` is installed (additive trend + seasonality)

Design
------
* **Pooled cross-city models** by default (all 5 cities share one model per
  target). City dummies are in the feature set so the model can specialise
  where it needs to. This is both more data-efficient and gives a single
  deployable artefact per target.

* **Strictly time-ordered splits**: `TimeSeriesSplit` for CV, one fixed
  forward holdout (default ``2025-01-01``) for final test metrics. No shuffle.

* **Per-target orchestration**: :func:`train_weather_models` iterates over
  `FORECAST_TARGETS`, trains every available model, scores it, keeps the
  best by test RMSE, and pickles it to ``models/weather/<target>/<algo>.joblib``.

* **Circular-aware metric for wind direction** so we do not reward a model
  that is 350° off from truth as if it were 350° wrong.

Public entry point: :func:`train_weather_models`.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    FORECAST_TARGETS,
    MODELS_DIR,
    PROCESSED_DIR,
)
from src.utils.logging_utils import get_logger
from src.weather.features import (
    DAILY_TARGET_COLUMNS,
    feature_columns,
    time_train_test_split,
)

logger = get_logger(__name__)


# ============================================================================
# Metrics
# ============================================================================

def _circular_abs_error_deg(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Absolute error on the 0-360 circle (e.g. 350 vs 10 is 20, not 340)."""
    diff = np.abs(np.asarray(y_true) - np.asarray(y_pred)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    circular: bool = False,
) -> Dict[str, float]:
    """Return MAE, RMSE, R^2 (and MAPE when appropriate)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "n": 0}

    if circular:
        err = _circular_abs_error_deg(y_true, y_pred)
        return {
            "MAE": float(err.mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": np.nan,  # R^2 not meaningful on a circle
            "n": int(len(y_true)),
            "circular_MAE_deg": float(err.mean()),
        }

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    out = {"MAE": float(mae), "RMSE": rmse, "R2": float(r2), "n": int(len(y_true))}

    # MAPE only if no targets are near zero (rain can have many zeros -> useless)
    denom = np.abs(y_true)
    if (denom > 1e-3).all():
        out["MAPE_pct"] = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)
    return out


# ============================================================================
# Data preparation
# ============================================================================

@dataclass
class TrainingFrames:
    """Everything a single-target training run needs.

    Target is shifted forward by ``horizon`` days per city, so the features
    observed at row ``t`` are used to predict the target at ``t + horizon``.
    This is the ONLY way to use same-day derived features (min/max/humidity/
    dew point / etc.) without leaking — they are observed before ``t + horizon``
    is realised.
    """

    target: str               # raw target key, e.g. "temperature_2m"
    target_col: str           # original daily column, e.g. "temperature_2m_mean"
    horizon: int              # forecast horizon in days
    predictors: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series        # y at t + horizon
    y_test: pd.Series
    train_meta: pd.DataFrame  # City + date of the FEATURE row (t)
    test_meta: pd.DataFrame


def prepare_training_frames(
    feats: pd.DataFrame,
    target: str,
    test_start: str = "2025-01-01",
    horizon: int = 1,
    warmup_drop: int = 365,
    group_col: str = "City",
    date_col: str = "date",
) -> TrainingFrames:
    """Build the (X, y) quartet for one target at the requested forecast horizon.

    Pipeline:
      1. Drop the first ``warmup_drop`` rows per city (lag-365 warm-up).
      2. Create ``y_shift = target.shift(-horizon)`` per city — this is the
         value we want to predict from features observed at ``t``.
      3. Drop rows where ``y_shift`` is NaN (the last ``horizon`` rows of
         each city cannot be trained on).
      4. Split by ``test_start``. The split applies to the FEATURE-row date
         (``t``), so the test target dates are ``test_start + horizon`` onward.
      5. Exclude the raw target column itself from predictors (obvious
         leakage) and the wind-direction target when that is what we forecast.
    """
    if target not in DAILY_TARGET_COLUMNS:
        raise ValueError(f"Unknown target {target!r}. Choose from {list(DAILY_TARGET_COLUMNS)}")
    target_col = DAILY_TARGET_COLUMNS[target]
    if target_col not in feats.columns:
        raise ValueError(f"Target column {target_col!r} not in feature frame")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    # 1. Sort + drop warmup per city (pd.concat avoids the fragmentation penalty
    #    from groupby().apply())
    df = feats.sort_values([group_col, date_col]).reset_index(drop=True)
    pieces: List[pd.DataFrame] = []
    for _, g in df.groupby(group_col, sort=False):
        if len(g) > warmup_drop:
            pieces.append(g.iloc[warmup_drop:])
    trimmed = pd.concat(pieces, ignore_index=True)

    # 2. Shift target forward by horizon within each city. We defragment and
    #    assign in one go to avoid pandas' fragmentation penalty.
    shifted_col = f"_y_shift_h{horizon}"
    shifted = trimmed.groupby(group_col, sort=False)[target_col].shift(-horizon)
    trimmed = trimmed.copy()  # defragment
    trimmed[shifted_col] = shifted.values

    # 3. Drop rows with NaN shifted target (last `horizon` rows per city)
    before = len(trimmed)
    trimmed = trimmed.dropna(subset=[shifted_col]).reset_index(drop=True)
    logger.info("Horizon=%d shift dropped %d rows (last %d per city)",
                horizon, before - len(trimmed), horizon)

    # 4. Predictors: every numeric column except the SHIFTED target and any
    #    meta/outlier columns. The raw target_col IS today's observed value
    #    and is a legitimate predictor of tomorrow. It is specifically what
    #    the persistence baseline uses.
    meta = {group_col, date_col, shifted_col}
    outlier_cols = {c for c in trimmed.columns if c.endswith("_outlier")}
    # The OTHER daily target columns are also fine as predictors — they are
    # observed at day t just like temperature_2m_mean is. Only exclude the
    # shifted target (obvious) and the raw target when forecasting direction
    # (circular leakage through sin/cos + wind_u/v).
    all_predictors = [
        c for c in trimmed.columns
        if c not in meta | outlier_cols and pd.api.types.is_numeric_dtype(trimmed[c])
    ]
    if target == "wind_direction_10m":
        # Strip direction and its trivial trigonometric reparameterisations;
        # keep the vector components (wind_u, wind_v) since they carry speed too.
        forbidden = {"wind_direction_10m", "wind_dir_sin", "wind_dir_cos"}
        all_predictors = [c for c in all_predictors if c not in forbidden]

    logger.info("Predictor set: %d columns (target col %s is %s in predictors)",
                len(all_predictors), target_col,
                "included" if target_col in all_predictors else "excluded")

    # 5. Time split — by feature-row date
    train, test = time_train_test_split(trimmed, test_start=test_start, date_col=date_col)

    tf = TrainingFrames(
        target=target,
        target_col=target_col,
        horizon=horizon,
        predictors=all_predictors,
        X_train=train[all_predictors].copy(),
        X_test=test[all_predictors].copy(),
        y_train=train[shifted_col].copy(),
        y_test=test[shifted_col].copy(),
        train_meta=train[[group_col, date_col]].copy(),
        test_meta=test[[group_col, date_col]].copy(),
    )
    logger.info("Prepared %s h=%d: X_train=%s, X_test=%s",
                target, horizon, tf.X_train.shape, tf.X_test.shape)
    return tf


# ============================================================================
# Individual model trainers
# ============================================================================

def train_persistence(tf: TrainingFrames) -> Tuple[Any, np.ndarray]:
    """Naive persistence: predict y[t+h] = y[t] (today's observed value).

    The feature-row at date ``t`` still contains ``target_col`` (the observed
    value today), which is the right persistence forecast for ``t + horizon``.
    This is the hurdle every trained model must clear.
    """
    col = tf.target_col
    if col not in tf.X_test.columns:
        raise RuntimeError(
            f"Persistence baseline needs {col} in features "
            f"(was it stripped because target == wind_direction_10m?)"
        )

    class _Persistence:
        def __init__(self, c: str) -> None:
            self.col = c
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return X[self.col].values
        def __repr__(self) -> str:
            return f"Persistence(col={self.col!r})"

    model = _Persistence(col)
    return model, model.predict(tf.X_test)


def train_ridge(tf: TrainingFrames, alpha: float = 1.0) -> Tuple[Any, np.ndarray]:
    """Ridge regression with a fixed alpha (simple, explainable)."""
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(tf.X_train.fillna(0.0), tf.y_train)
    pred = model.predict(tf.X_test.fillna(0.0))
    return model, pred


def train_hgbr(
    tf: TrainingFrames,
    max_iter: int = 200,
    learning_rate: float = 0.08,
    max_depth: Optional[int] = 6,
    l2_regularization: float = 0.1,
    random_state: int = 0,
) -> Tuple[Any, np.ndarray]:
    """sklearn's HistGradientBoostingRegressor - fast, handles NaNs natively.

    Defaults tuned for ~500-feature tabular data on a single CPU: smaller
    depth, higher learning rate, 200 boosting iterations with early stopping.
    """
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=random_state,
    )
    model.fit(tf.X_train, tf.y_train)
    pred = model.predict(tf.X_test)
    return model, pred


def train_xgboost(
    tf: TrainingFrames,
    n_estimators: int = 800,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    random_state: int = 0,
) -> Tuple[Any, np.ndarray]:
    """XGBoost — only used when the library is installed on the host.

    Raises :class:`RuntimeError` otherwise; the orchestrator skips gracefully.
    """
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("xgboost is not installed.") from exc

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(tf.X_train, tf.y_train, verbose=False)
    pred = model.predict(tf.X_test)
    return model, pred


def train_sarima_per_city(
    feats: pd.DataFrame,
    target_col: str,
    test_start: str,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
    group_col: str = "City",
    date_col: str = "date",
) -> Tuple[Any, pd.Series]:
    """Fit a SARIMA per city and produce a concatenated test prediction.

    Returns a dict-of-fitted-models (keyed by city) and a Series of predictions
    aligned to the test index across all cities.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("statsmodels is not installed.") from exc

    ts = pd.Timestamp(test_start)
    models: Dict[str, Any] = {}
    preds: List[pd.Series] = []
    for city, g in feats.sort_values(date_col).groupby(group_col, sort=False):
        g = g.set_index(date_col)
        y_train = g.loc[g.index < ts, target_col]
        y_test  = g.loc[g.index >= ts, target_col]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred = res.forecast(steps=len(y_test))
        pred.index = y_test.index
        pred.name = f"sarima_{city}"
        preds.append(pred.to_frame(name="y_pred").assign(**{group_col: city}).reset_index())
        models[city] = res
        logger.info("  SARIMA fit %s -> aic=%.1f", city, res.aic)

    pred_df = pd.concat(preds, ignore_index=True)
    return models, pred_df


def train_prophet_per_city(
    feats: pd.DataFrame,
    target_col: str,
    test_start: str,
    group_col: str = "City",
    date_col: str = "date",
) -> Tuple[Any, pd.DataFrame]:
    """Fit Prophet per city (requires the ``prophet`` package)."""
    try:
        from prophet import Prophet  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("prophet is not installed.") from exc

    ts = pd.Timestamp(test_start)
    models: Dict[str, Any] = {}
    preds: List[pd.DataFrame] = []
    for city, g in feats.sort_values(date_col).groupby(group_col, sort=False):
        train_df = g[g[date_col] < ts][[date_col, target_col]].rename(
            columns={date_col: "ds", target_col: "y"}
        )
        test_df = g[g[date_col] >= ts][[date_col]].rename(columns={date_col: "ds"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(train_df)
        fc = m.predict(test_df)[["ds", "yhat"]].rename(columns={"ds": date_col, "yhat": "y_pred"})
        fc[group_col] = city
        preds.append(fc)
        models[city] = m
        logger.info("  Prophet fit %s", city)

    return models, pd.concat(preds, ignore_index=True)


# ============================================================================
# Time-series CV
# ============================================================================

def time_series_cv(
    estimator_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "neg_root_mean_squared_error",
) -> Dict[str, List[float]]:
    """Manual TSCV so we can control NaN handling and callable factories."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes = [], []
    for fold, (tr, va) in enumerate(tscv.split(X), start=1):
        est = estimator_factory()
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xva, yva = X.iloc[va], y.iloc[va]
        if hasattr(est, "fit"):
            # HistGradientBoosting handles NaNs natively; others may not
            try:
                est.fit(Xtr, ytr)
                p = est.predict(Xva)
            except ValueError:
                est.fit(Xtr.fillna(0.0), ytr)
                p = est.predict(Xva.fillna(0.0))
        else:
            p = est.predict(Xva)
        rmses.append(float(np.sqrt(mean_squared_error(yva, p))))
        maes.append(float(mean_absolute_error(yva, p)))
        logger.debug("  fold %d: RMSE=%.3f  MAE=%.3f", fold, rmses[-1], maes[-1])
    return {"rmse": rmses, "mae": maes}


# ============================================================================
# Result bookkeeping
# ============================================================================

@dataclass
class ModelResult:
    """Scored, persisted training result for one (target, algo) pair."""
    target: str
    algo: str
    metrics: Dict[str, float]
    cv_metrics: Optional[Dict[str, List[float]]] = None
    model_path: Optional[Path] = None
    params: Dict[str, Any] = field(default_factory=dict)


def save_model(obj: Any, target: str, algo: str) -> Path:
    out_dir = MODELS_DIR / "weather" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{algo}.joblib"
    joblib.dump(obj, path)
    logger.info("  Saved %s/%s -> %s (%.2f MB)",
                target, algo, path, path.stat().st_size / 1024 / 1024)
    return path


# ============================================================================
# Per-target orchestrator
# ============================================================================

def train_single_target(
    tf: TrainingFrames,
    algos: Optional[List[str]] = None,
    circular: bool = False,
    run_cv: bool = True,
) -> Dict[str, ModelResult]:
    """Fit every requested algo for a single target, score on the holdout."""
    all_algos = algos or ["persistence", "ridge", "hgbr", "xgboost"]
    results: Dict[str, ModelResult] = {}

    dispatch: Dict[str, Callable[[TrainingFrames], Tuple[Any, np.ndarray]]] = {
        "persistence": train_persistence,
        "ridge":       train_ridge,
        "hgbr":        train_hgbr,
        "xgboost":     train_xgboost,
    }

    for algo in all_algos:
        if algo not in dispatch:
            logger.warning("  Unknown algo %s, skipping", algo)
            continue
        try:
            logger.info("  training %-12s ...", algo)
            model, pred = dispatch[algo](tf)
            m = evaluate_regression(tf.y_test.values, pred, circular=circular)

            cv_scores = None
            if run_cv and algo not in ("persistence", "xgboost"):
                try:
                    factory_map = {
                        "ridge": lambda: Ridge(alpha=1.0, random_state=0),
                        "hgbr":  lambda: HistGradientBoostingRegressor(
                            max_iter=150, learning_rate=0.08, max_depth=6,
                            l2_regularization=0.1, early_stopping=True,
                            random_state=0),
                    }
                    if algo in factory_map:
                        cv_scores = time_series_cv(factory_map[algo], tf.X_train, tf.y_train, n_splits=4)
                except Exception as cv_exc:  # noqa: BLE001
                    logger.warning("  CV failed for %s: %s", algo, cv_exc)

            model_path = save_model(model, tf.target, algo) if algo != "persistence" else None
            results[algo] = ModelResult(
                target=tf.target, algo=algo, metrics=m,
                cv_metrics=cv_scores, model_path=model_path,
            )
            mae_val = m.get("MAE", float("nan"))
            rmse_val = m.get("RMSE", float("nan"))
            logger.info("  %s MAE=%.3f RMSE=%.3f",
                        algo, mae_val, rmse_val)
        except RuntimeError as exc:
            logger.info("  %s skipped: %s", algo, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("  %s failed: %s", algo, exc)

    return results


# ============================================================================
# Top-level orchestrator
# ============================================================================

def train_weather_models(
    features_path: Optional[Path] = None,
    targets: Optional[List[str]] = None,
    algos: Optional[List[str]] = None,
    test_start: str = "2025-01-01",
    horizon: int = 1,
    run_cv: bool = True,
) -> pd.DataFrame:
    """Train every (target, algo) combination at one forecast horizon.

    Persists best model per target to ``models/weather/<target>/<algo>.joblib``
    and writes ``models/weather/summary.csv`` + ``models/weather/best.json``.
    """
    features_path = features_path or (PROCESSED_DIR / "weather_features.csv")
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} - run notebook 03 first")

    logger.info("=" * 72)
    logger.info("PHASE 2.3 - Weather Modeling (horizon=%d day)", horizon)
    logger.info("Input: %s", features_path)
    logger.info("=" * 72)
    feats = pd.read_csv(features_path, parse_dates=["date"])
    if feats["date"].dt.tz is not None:
        feats["date"] = feats["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    targets = targets or FORECAST_TARGETS
    algos = algos or ["persistence", "ridge", "hgbr", "xgboost"]

    summary_rows: List[Dict[str, Any]] = []
    best_per_target: Dict[str, Dict[str, Any]] = {}

    for target in targets:
        circular = (target == "wind_direction_10m")
        logger.info("--- target: %s (circular=%s, h=%d) ---", target, circular, horizon)
        try:
            tf = prepare_training_frames(
                feats, target=target, test_start=test_start, horizon=horizon
            )
        except ValueError as exc:
            logger.error("Skipping %s: %s", target, exc)
            continue

        # For a pure circular target we can't use scalar persistence meaningfully
        algos_this = [a for a in algos if not (circular and a == "persistence")]
        results = train_single_target(tf, algos=algos_this, circular=circular, run_cv=run_cv)

        for algo, r in results.items():
            row = {"target": target, "algo": algo, "horizon": horizon, **r.metrics}
            if r.cv_metrics:
                row["cv_rmse_mean"] = float(np.mean(r.cv_metrics["rmse"]))
                row["cv_rmse_std"]  = float(np.std(r.cv_metrics["rmse"]))
            summary_rows.append(row)

        if results:
            candidates = {a: r for a, r in results.items() if a != "persistence"}
            if candidates:
                best_algo = min(candidates, key=lambda a: candidates[a].metrics.get("RMSE", np.inf))
                best_per_target[target] = {
                    "algo": best_algo,
                    "horizon": horizon,
                    "metrics": results[best_algo].metrics,
                    "model_path": str(results[best_algo].model_path),
                }
                logger.info("  >>> best for %s: %s (RMSE=%.3f)",
                            target, best_algo, results[best_algo].metrics["RMSE"])

    summary = pd.DataFrame(summary_rows)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "weather").mkdir(parents=True, exist_ok=True)
    summary.to_csv(MODELS_DIR / "weather" / "summary.csv", index=False)
    with open(MODELS_DIR / "weather" / "best.json", "w") as f:
        json.dump(best_per_target, f, indent=2, default=str)

    logger.info("=" * 72)
    logger.info("Modeling complete. Summary saved to %s", MODELS_DIR / "weather" / "summary.csv")
    logger.info("=" * 72)
    return summary


if __name__ == "__main__":
    train_weather_models()
