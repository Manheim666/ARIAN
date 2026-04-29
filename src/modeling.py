from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelResult:
    name: str
    metrics: dict


def _bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.quantile(values, alpha / 2))
    hi = float(np.quantile(values, 1 - alpha / 2))
    return lo, hi


def train_regression_next_day_temperature(df: pd.DataFrame, figures_dir=None) -> ModelResult:
    """Regression model: predict next-day Temperature_C using lag/rolling + seasonality."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer

    d = df.dropna(subset=["Temperature_C", "lag_1", "lag_7", "rolling_mean_7"]).copy()
    d = d.sort_values(["City", "Date"]) if "City" in d.columns else d

    # target: next day temp per city
    if "City" in d.columns:
        d["y"] = d.groupby("City")["Temperature_C"].shift(-1)
    else:
        d["y"] = d["Temperature_C"].shift(-1)
    d = d.dropna(subset=["y"]) 

    cat = [c for c in ["City"] if c in d.columns]
    num = [c for c in ["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7", "month", "day_of_year", "Precipitation_mm", "Wind_Speed_kmh"] if c in d.columns]

    X = d[cat + num]
    y = d["y"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
            (
                "num",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]),
                num,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    # Bootstrap CI for RMSE
    rng = np.random.default_rng(7)
    idx = np.arange(len(y_test))
    rmses = []
    for _ in range(200):
        b = rng.choice(idx, size=len(idx), replace=True)
        rmses.append(float(np.sqrt(mean_squared_error(y_test.iloc[b], pred[b]))))
    rmse_ci = _bootstrap_ci(np.asarray(rmses))

    if figures_dir is not None:
        try:
            from pathlib import Path

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            Path(figures_dir).mkdir(parents=True, exist_ok=True)
            resid = (y_test.to_numpy() - pred).astype(float)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(pred, resid, alpha=0.25)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title("Regression residuals (next-day temp)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual")
            fig.tight_layout()
            fig.savefig(Path(figures_dir) / "regression_residuals.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    return ModelResult(
        "Regression: Ridge next-day temperature",
        {"rmse": rmse, "rmse_ci95": rmse_ci, "n_test": int(len(y_test))},
    )


def train_classification_rain(df: pd.DataFrame) -> ModelResult:
    """Classification model: rain vs no rain."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer

    if "Precipitation_mm" not in df.columns:
        return ModelResult("Classification: Logistic rain/no-rain", {"skipped": True, "reason": "Missing Precipitation_mm"})

    d = df.dropna(subset=["Precipitation_mm", "lag_1", "rolling_mean_7"]).copy()
    d["y"] = (d["Precipitation_mm"].astype(float) > 0.0).astype(int)

    cat = [c for c in ["City"] if c in d.columns]
    num = [c for c in ["Temperature_C", "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7", "month", "day_of_year", "Wind_Speed_kmh"] if c in d.columns]

    X = d[cat + num]
    y = d["y"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
            (
                "num",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]),
                num,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("model", LogisticRegression(max_iter=200, n_jobs=None)),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred))

    rng = np.random.default_rng(7)
    idx = np.arange(len(y_test))
    accs = []
    f1s = []
    yt = y_test.to_numpy()
    for _ in range(200):
        b = rng.choice(idx, size=len(idx), replace=True)
        accs.append(float(accuracy_score(yt[b], pred[b])))
        f1s.append(float(f1_score(yt[b], pred[b], zero_division=0)))
    acc_ci = _bootstrap_ci(np.asarray(accs))
    f1_ci = _bootstrap_ci(np.asarray(f1s))

    return ModelResult(
        "Classification: Logistic rain/no-rain",
        {"accuracy": acc, "accuracy_ci95": acc_ci, "f1": f1, "f1_ci95": f1_ci, "n_test": int(len(y_test))},
    )
