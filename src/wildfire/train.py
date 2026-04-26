"""
Wildfire model training (Phase 4 - Step 2).

Trains two complementary models on the feature matrix from Phase 4.1:

    CLASSIFIER    target = fire_occurred  (binary)
      baseline:   prior rate (class-balance floor) + Logistic Regression
      main:       HistGradientBoostingClassifier (HGBC)
      optional:   RandomForest, XGBoost (lazy-imported)
      metric:     ROC-AUC, PR-AUC, F1 at the optimal threshold

    REGRESSOR     target = fire_count     (non-negative integer)
      baseline:   mean (constant) + Ridge
      main:       HistGradientBoostingRegressor with log1p target
      metric:     MAE, RMSE, Spearman rho

Design
------
* **Time-ordered train/test split.** Test = last full year of FIRMS coverage
  (2024). Training uses everything before.
* **Pooled cross-city models** (City as a one-hot in the feature set) so we
  can learn shared fire dynamics while still allowing the model to
  specialise per city.
* **Class imbalance** handled by ``class_weight="balanced"`` for logistic /
  HGBC's ``class_weight`` and by calibrated thresholding after training.
* **Probability calibration** via `CalibratedClassifierCV` on the HGBC output
  so that `predict_proba` is a legitimate risk number, not a raw score.

Public entry points
-------------------
- :func:`train_fire_classifier`
- :func:`train_fire_count_regressor`
- :func:`train_wildfire_models` (orchestrator -- trains and persists both)
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr  # type: ignore

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import MODELS_DIR, PROCESSED_DIR
from src.utils.logging_utils import get_logger
from src.wildfire.features import TARGET_COLUMNS, predictor_columns

logger = get_logger(__name__)


# ============================================================================
# Data preparation
# ============================================================================

@dataclass
class TrainingSplit:
    target: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictors: List[str]
    train_meta: pd.DataFrame   # City + date
    test_meta: pd.DataFrame


def prepare_split(
    df: pd.DataFrame,
    target: str,
    test_start: str = "2024-01-01",
    group_col: str = "City",
    date_col: str = "date",
) -> TrainingSplit:
    """Time-based split. Rows with NaN target are dropped.

    Also builds City one-hot columns so pooled models can specialise per city.
    """
    if target not in df.columns:
        raise ValueError(f"Target {target!r} not in frame")

    work = df.dropna(subset=[target]).copy()
    work["date"] = pd.to_datetime(work["date"])
    if work["date"].dt.tz is not None:
        work["date"] = work["date"].dt.tz_localize(None)

    # One-hot the city if not already present
    dummy_prefix = "city_"
    if not any(c.startswith(dummy_prefix) for c in work.columns):
        dummies = pd.get_dummies(work[group_col], prefix="city", dtype=np.int8)
        work = pd.concat([work, dummies], axis=1)

    preds = predictor_columns(work) + [c for c in work.columns if c.startswith(dummy_prefix)]
    preds = sorted(set(preds))
    # drop non-numeric defensively
    preds = [c for c in preds if pd.api.types.is_numeric_dtype(work[c])]

    ts = pd.Timestamp(test_start)
    train = work[work["date"] < ts]
    test  = work[work["date"] >= ts]

    split = TrainingSplit(
        target=target,
        X_train=train[preds].copy(),
        X_test=test[preds].copy(),
        y_train=train[target].copy(),
        y_test=test[target].copy(),
        predictors=preds,
        train_meta=train[[group_col, date_col]].copy(),
        test_meta=test[[group_col, date_col]].copy(),
    )
    logger.info("Split %s @ %s: train=%s test=%s (%d predictors)",
                target, test_start, split.X_train.shape, split.X_test.shape, len(preds))
    if target == "fire_occurred":
        logger.info("  class balance: train %.3f, test %.3f",
                    split.y_train.mean(), split.y_test.mean())
    return split


# ============================================================================
# Classifier
# ============================================================================

def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Sweep thresholds and return (best_threshold, best_f1)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve drops the last element of thresholds
    thr = np.append(thr, 1.0)
    # Avoid division by zero
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def evaluate_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"ROC_AUC": np.nan, "PR_AUC": np.nan,
                "best_threshold": np.nan, "F1_at_best": np.nan,
                "n": int(len(y_true))}
    roc = roc_auc_score(y_true, y_score)
    pr  = average_precision_score(y_true, y_score)
    thr, f1 = _best_f1_threshold(y_true, y_score)
    return {
        "ROC_AUC": float(roc), "PR_AUC": float(pr),
        "best_threshold": float(thr), "F1_at_best": float(f1),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def train_fire_classifier(
    split: TrainingSplit,
    algo: str = "hgbc",
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """Fit a classifier; return ``(model, test_scores, metrics)``.

    ``test_scores`` are calibrated probabilities in [0, 1].
    """
    X_tr = split.X_train
    X_te = split.X_test
    y_tr = split.y_train.astype(int)
    y_te = split.y_test.astype(int)

    if algo == "baseline":
        model = DummyClassifier(strategy="stratified", random_state=0)
        model.fit(X_tr.fillna(0.0), y_tr)
        scores = model.predict_proba(X_te.fillna(0.0))[:, 1]

    elif algo == "logistic":
        # Needs scaling + imputation
        Xtr = X_tr.fillna(0.0)
        Xte = X_te.fillna(0.0)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        base = LogisticRegression(max_iter=300, class_weight="balanced",
                                  solver="liblinear")
        base.fit(Xtr_s, y_tr)
        scores = base.predict_proba(Xte_s)[:, 1]
        model = {"scaler": scaler, "clf": base}   # bundle for persistence

    elif algo == "hgbc":
        # HGBC handles NaNs natively. Wrap with isotonic calibration so
        # predict_proba gives calibrated risks.
        base = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.08,
            max_iter=250,
            max_depth=6,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            class_weight="balanced",
            random_state=0,
        )
        # CalibratedClassifierCV refits on TimeSeries-like splits internally,
        # which is fine for tabular non-time-sensitive calibration
        model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        scores = model.predict_proba(X_te)[:, 1]

    elif algo == "xgboost":
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("xgboost not installed") from exc
        pos = max(int(y_tr.sum()), 1)
        neg = max(int((y_tr == 0).sum()), 1)
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=neg / pos, eval_metric="logloss",
            tree_method="hist", random_state=0, n_jobs=-1,
        )
        model.fit(X_tr, y_tr, verbose=False)
        scores = model.predict_proba(X_te)[:, 1]
    else:
        raise ValueError(f"Unknown algo {algo!r}")

    metrics = evaluate_classifier(y_te.values, scores)
    logger.info("  %s  ROC_AUC=%.3f  PR_AUC=%.3f  F1=%.3f @ thr=%.2f",
                algo, metrics["ROC_AUC"], metrics["PR_AUC"],
                metrics["F1_at_best"], metrics["best_threshold"])
    return model, scores, metrics


# ============================================================================
# Regressor (fire count)
# ============================================================================

def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "spearman": np.nan, "n": 0}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) >= 3 else np.nan
    return {"MAE": float(mae), "RMSE": rmse,
            "spearman": float(rho) if np.isfinite(rho) else np.nan,
            "n": int(len(y_true))}


def train_fire_count_regressor(
    split: TrainingSplit,
    algo: str = "hgbr_log",
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """Fit a count regressor; ``hgbr_log`` applies log1p to the target.

    Returns ``(model, test_predictions, metrics)``. Predictions are already
    transformed back to the original (count) scale.
    """
    X_tr = split.X_train
    X_te = split.X_test
    y_tr = split.y_train.astype(float)
    y_te = split.y_test.astype(float)

    if algo == "baseline":
        model = DummyRegressor(strategy="mean")
        model.fit(X_tr.fillna(0.0), y_tr)
        preds = model.predict(X_te.fillna(0.0))

    elif algo == "ridge":
        Xtr = X_tr.fillna(0.0)
        Xte = X_te.fillna(0.0)
        scaler = StandardScaler().fit(Xtr)
        base = Ridge(alpha=1.0, random_state=0)
        base.fit(scaler.transform(Xtr), np.log1p(y_tr))
        preds = np.expm1(base.predict(scaler.transform(Xte)))
        preds = np.clip(preds, 0, None)
        model = {"scaler": scaler, "reg": base}

    elif algo == "hgbr_log":
        base = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.08, max_iter=300,
            max_depth=6, l2_regularization=0.1,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, random_state=0,
        )
        base.fit(X_tr, np.log1p(y_tr))
        preds = np.expm1(base.predict(X_te))
        preds = np.clip(preds, 0, None)
        model = base

    else:
        raise ValueError(f"Unknown algo {algo!r}")

    metrics = evaluate_regressor(y_te.values, preds)
    logger.info("  %s  MAE=%.3f  RMSE=%.3f  spearman=%.3f",
                algo, metrics["MAE"], metrics["RMSE"], metrics["spearman"])
    return model, preds, metrics


# ============================================================================
# Orchestrator
# ============================================================================

def save_model(obj: Any, subpath: str) -> Path:
    p = MODELS_DIR / "wildfire" / subpath
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)
    logger.info("  Saved -> %s (%.2f MB)", p, p.stat().st_size / 1024 / 1024)
    return p


def train_wildfire_models(
    features_path: Optional[Path] = None,
    test_start: str = "2024-01-01",
    classifier_algos: Optional[List[str]] = None,
    regressor_algos: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """End-to-end: train and compare every classifier and regressor algo.

    Persists best model per task to ``models/wildfire/{classifier,regressor}/<algo>.joblib``
    and writes ``models/wildfire/summary.csv``, ``best.json``.
    """
    features_path = features_path or (PROCESSED_DIR / "wildfire_features.csv")
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} -- run notebook 07 first")

    classifier_algos = classifier_algos or ["baseline", "logistic", "hgbc", "xgboost"]
    regressor_algos  = regressor_algos  or ["baseline", "ridge", "hgbr_log"]

    logger.info("=" * 72)
    logger.info("PHASE 4.2 - Wildfire modeling")
    logger.info("Test split: rows from %s onward", test_start)
    logger.info("=" * 72)

    df = pd.read_csv(features_path, parse_dates=["date"])

    # ---------- Classifier ----------
    logger.info("--- CLASSIFICATION: fire_occurred ---")
    split_clf = prepare_split(df, "fire_occurred", test_start=test_start)
    cls_rows: List[Dict] = []
    best_cls = {"algo": None, "metric": -np.inf, "path": None}
    for a in classifier_algos:
        try:
            model, scores, m = train_fire_classifier(split_clf, algo=a)
            path = save_model(model, f"classifier/{a}.joblib")
            cls_rows.append({"task": "classification", "algo": a,
                             "model_path": str(path), **m})
            if a != "baseline" and m.get("PR_AUC", -1) > best_cls["metric"]:
                best_cls = {"algo": a, "metric": m["PR_AUC"], "path": str(path),
                            "metrics": m}
        except RuntimeError as exc:
            logger.info("  %s skipped: %s", a, exc)

    # ---------- Regressor ----------
    logger.info("--- REGRESSION: fire_count ---")
    split_reg = prepare_split(df, "fire_count", test_start=test_start)
    reg_rows: List[Dict] = []
    best_reg = {"algo": None, "metric": np.inf, "path": None}
    for a in regressor_algos:
        try:
            model, preds, m = train_fire_count_regressor(split_reg, algo=a)
            path = save_model(model, f"regressor/{a}.joblib")
            reg_rows.append({"task": "regression", "algo": a,
                             "model_path": str(path), **m})
            if a != "baseline" and m.get("MAE", np.inf) < best_reg["metric"]:
                best_reg = {"algo": a, "metric": m["MAE"], "path": str(path),
                            "metrics": m}
        except Exception as exc:  # noqa: BLE001
            logger.warning("  %s failed: %s", a, exc)

    # ---------- Persist summary ----------
    summary = pd.DataFrame(cls_rows + reg_rows)
    (MODELS_DIR / "wildfire").mkdir(parents=True, exist_ok=True)
    summary.to_csv(MODELS_DIR / "wildfire" / "summary.csv", index=False)

    best = {"classifier": best_cls, "regressor": best_reg}
    with open(MODELS_DIR / "wildfire" / "best.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    logger.info("=" * 72)
    logger.info("Complete. Best classifier = %s, best regressor = %s",
                best_cls.get("algo"), best_reg.get("algo"))
    logger.info("=" * 72)

    return {"summary": summary, "best": pd.DataFrame([best_cls, best_reg])}


if __name__ == "__main__":
    train_wildfire_models()
