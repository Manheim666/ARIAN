"""
ARIAN Wildfire Prediction — Modeling Utilities
================================================
Model factories, training helpers, hyperparameter search utilities.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

SEED = 42


def _gpu_available() -> bool:
    """Return True if a CUDA GPU is usable for tree-based boosters."""
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


USE_GPU = _gpu_available()


# ═══════════════════════════════════════════════════════════════════════════
# Weather Model Factory
# ═══════════════════════════════════════════════════════════════════════════

def get_weather_models():
    """Return dict of {name: model} for weather forecasting comparison."""
    models = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_split=5,
            random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=15, min_samples_split=5,
            random_state=SEED, n_jobs=-1),
        "HistGBR": HistGradientBoostingRegressor(
            max_iter=500, max_depth=8, learning_rate=0.05,
            random_state=SEED),
    }

    try:
        import xgboost as xgb
        xgb_params = dict(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbosity=0)
        if USE_GPU:
            xgb_params["tree_method"] = "gpu_hist"
        models["XGBoost"] = xgb.XGBRegressor(**xgb_params)
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        lgb_params = dict(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1)
        if USE_GPU:
            lgb_params["device"] = "gpu"
        models["LightGBM"] = lgb.LGBMRegressor(**lgb_params)
    except ImportError:
        pass

    try:
        import catboost as cb
        cb_params = dict(
            iterations=500, depth=8, learning_rate=0.05,
            random_seed=SEED, verbose=0)
        if USE_GPU:
            cb_params["task_type"] = "GPU"
        models["CatBoost"] = cb.CatBoostRegressor(**cb_params)
    except ImportError:
        pass

    return models


# ═══════════════════════════════════════════════════════════════════════════
# Fire Detection Model Factory
# ═══════════════════════════════════════════════════════════════════════════

def get_fire_models(imbalance_ratio=10.0):
    """Return dict of {name: (model, imbalance_strategy)} for fire detection."""
    models = {}

    # 1. LogisticRegression baseline
    models["LogReg_balanced"] = (
        LogisticRegression(class_weight="balanced", max_iter=1000,
                           random_state=SEED, n_jobs=-1),
        "class_weight=balanced")

    # 2. RandomForest balanced
    models["RF_balanced"] = (
        RandomForestClassifier(n_estimators=300, max_depth=20,
                               min_samples_split=5, class_weight="balanced",
                               random_state=SEED, n_jobs=-1),
        "class_weight=balanced")

    # 3. ExtraTrees balanced
    models["ET_balanced"] = (
        ExtraTreesClassifier(n_estimators=300, max_depth=20,
                             min_samples_split=5, class_weight="balanced",
                             random_state=SEED, n_jobs=-1),
        "class_weight=balanced")

    # 4. HistGradientBoosting
    models["HistGBC"] = (
        HistGradientBoostingClassifier(
            max_iter=500, max_depth=8, learning_rate=0.05,
            class_weight="balanced", random_state=SEED),
        "class_weight=balanced")

    # 5. XGBoost cost-sensitive
    try:
        import xgboost as xgb
        xgb_params = dict(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=imbalance_ratio,
            eval_metric="aucpr", random_state=SEED,
            use_label_encoder=False, n_jobs=-1)
        if USE_GPU:
            xgb_params["tree_method"] = "gpu_hist"
        models["XGB_cost"] = (
            xgb.XGBClassifier(**xgb_params),
            f"scale_pos_weight={imbalance_ratio:.1f}" + (", gpu" if USE_GPU else ""))
    except ImportError:
        pass

    # 6. LightGBM
    try:
        import lightgbm as lgb
        lgb_params = dict(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=SEED, n_jobs=-1, verbose=-1)
        if USE_GPU:
            lgb_params["device"] = "gpu"
        models["LGB_unbalance"] = (
            lgb.LGBMClassifier(**lgb_params),
            "is_unbalance=True" + (", device=gpu" if USE_GPU else ""))
    except ImportError:
        pass

    # 7. CatBoost
    try:
        import catboost as cb
        cb_params = dict(
            iterations=500, depth=8, learning_rate=0.05,
            auto_class_weights="Balanced", eval_metric="F1",
            random_seed=SEED, verbose=0)
        if USE_GPU:
            cb_params["task_type"] = "GPU"
        models["CB_balanced"] = (
            cb.CatBoostClassifier(**cb_params),
            "auto_class_weights=Balanced" + (", GPU" if USE_GPU else ""))
    except ImportError:
        pass

    # 8. BalancedRandomForest (imblearn)
    try:
        from imblearn.ensemble import BalancedRandomForestClassifier
        models["BalancedRF"] = (
            BalancedRandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=5,
                random_state=SEED, n_jobs=-1),
            "balanced_subsample")
    except ImportError:
        pass

    # 9. EasyEnsemble (imblearn)
    try:
        from imblearn.ensemble import EasyEnsembleClassifier
        models["EasyEnsemble"] = (
            EasyEnsembleClassifier(
                n_estimators=20, random_state=SEED, n_jobs=-1),
            "easy_ensemble")
    except ImportError:
        pass

    return models


def calibrate_model(model, X_val, y_val, method="isotonic"):
    """Return a CalibratedClassifierCV wrapper."""
    cal = CalibratedClassifierCV(model, method=method, cv="prefit")
    cal.fit(X_val, y_val)
    return cal
