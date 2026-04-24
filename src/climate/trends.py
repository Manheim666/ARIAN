"""
Climate analysis (Phase 3).

Compares the weather record (and the next-30-day forecast) against historical
climatology to answer the project's climate-level questions:

    - Is this year warmer than last year?
    - Is rainfall increasing or decreasing?
    - Is the upcoming month hotter than "normal"?
    - Are seasons shifting?

Design principles
-----------------
* **Honest baselines.** We treat "climatology" as the multi-year average of
  the same calendar day (smoothed) and publish both its mean and standard
  deviation so anomaly estimates carry uncertainty.
* **Small-sample caveats.** With only ~5 full years, the SD of the
  climatology is itself noisy. Trend tests use Mann-Kendall + Theil-Sen
  (robust to noise and small N) rather than OLS + t-test.
* **No hallucinated signals.** Every comparison returns both an effect size
  and a qualitative label ("hotter / near normal / cooler") with explicit
  thresholds so the downstream notebook never invents a narrative.

Public API
----------
- :func:`daily_climatology` - mean/std per day-of-year per city
- :func:`monthly_climatology` - mean/std per calendar month per city
- :func:`annual_summary` - yearly aggregates for trend tests
- :func:`mann_kendall_trend`, :func:`theil_sen_slope` - robust trend stats
- :func:`compare_window_to_baseline` - anomaly of a window vs climatology
- :func:`forecast_anomalies` - attaches climatology anomaly to each forecast row
- :func:`detect_seasonal_shift` - phase-shift of seasonal-cycle peaks
- :func:`run_climate_analysis` - orchestrator that produces report artefacts
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# 1. Climatology baselines
# ============================================================================

def daily_climatology(
    df: pd.DataFrame,
    variable: str,
    group_col: str = "City",
    date_col: str = "date",
    smooth_window: int = 7,
) -> pd.DataFrame:
    """Mean and std per (city, day-of-year), smoothed over a ``smooth_window``.

    Parameters
    ----------
    smooth_window
        Centered rolling window (days) applied to the DOY climatology to
        dampen sampling noise from having only 5 years of history. 7-day is
        the meteorological default for daily climatology.

    Returns
    -------
    DataFrame
        Columns: ``City``, ``doy``, ``clim_mean``, ``clim_std``, ``n_years``.
    """
    if variable not in df.columns:
        raise ValueError(f"{variable} not in frame")

    work = df[[group_col, date_col, variable]].copy()
    work["doy"] = work[date_col].dt.dayofyear
    # Calendar-year drop for reproducibility
    work["year"] = work[date_col].dt.year

    stats = (
        work.groupby([group_col, "doy"])
            .agg(clim_mean=(variable, "mean"),
                 clim_std=(variable, "std"),
                 n_years=(variable, "count"))
            .reset_index()
    )

    if smooth_window and smooth_window > 1:
        stats = stats.sort_values([group_col, "doy"]).reset_index(drop=True)
        smoothed = (
            stats.groupby(group_col)[["clim_mean", "clim_std"]]
                 .transform(lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean())
        )
        stats[["clim_mean", "clim_std"]] = smoothed.values

    logger.info("Daily climatology for %s: %d city-days (smoothed %dd)",
                variable, len(stats), smooth_window)
    return stats


def monthly_climatology(
    df: pd.DataFrame,
    variable: str,
    group_col: str = "City",
    date_col: str = "date",
    agg: str = "mean",
) -> pd.DataFrame:
    """Monthly climatology aggregated first within-year, then across years.

    For accumulated variables (rain, precipitation) pass ``agg='sum'`` so the
    within-year aggregate is a monthly *total*; for intensive variables
    (temperature) pass ``'mean'``.
    """
    if agg not in {"mean", "sum"}:
        raise ValueError("agg must be 'mean' or 'sum'")
    if variable not in df.columns:
        raise ValueError(f"{variable} not in frame")

    work = df[[group_col, date_col, variable]].copy()
    work["year"] = work[date_col].dt.year
    work["month"] = work[date_col].dt.month

    # Within-year monthly aggregate, then climatology across years
    within = work.groupby([group_col, "year", "month"])[variable].agg(agg).reset_index()
    climo = (
        within.groupby([group_col, "month"])[variable]
              .agg(clim_mean="mean", clim_std="std", n_years="count")
              .reset_index()
    )
    logger.info("Monthly climatology for %s: %d city-months", variable, len(climo))
    return climo


def annual_summary(
    df: pd.DataFrame,
    variable: str,
    group_col: str = "City",
    date_col: str = "date",
    agg: str = "mean",
    min_days: int = 300,
) -> pd.DataFrame:
    """One row per (city, year) with ``variable`` aggregated over the year.

    Excludes years with fewer than ``min_days`` observations so partial years
    don't bias the annual trend.
    """
    work = df[[group_col, date_col, variable]].copy()
    work["year"] = work[date_col].dt.year
    counts = work.groupby([group_col, "year"])[variable].count()
    valid = counts[counts >= min_days].index
    work = work.set_index([group_col, "year"]).loc[valid].reset_index()

    summary = work.groupby([group_col, "year"])[variable].agg(agg).reset_index()
    summary = summary.rename(columns={variable: f"{variable}_{agg}"})
    logger.info("Annual %s (%s): %d city-years after filtering",
                variable, agg, len(summary))
    return summary


# ============================================================================
# 2. Robust trend statistics
# ============================================================================

def mann_kendall_trend(y: Sequence[float]) -> Dict[str, float]:
    """Non-parametric trend test (Mann-Kendall).

    Returns
    -------
    dict with keys:
      ``S`` (sum of sign differences), ``z`` (standardised statistic),
      ``p_two_sided``, ``direction`` ("increasing" / "decreasing" / "no trend"),
      ``n``.

    Notes
    -----
    Variance correction for ties is included. For small samples (n<10)
    results should be treated as suggestive, not definitive.
    """
    y = np.asarray([v for v in y if np.isfinite(v)], dtype=float)
    n = len(y)
    if n < 3:
        return {"S": 0.0, "z": 0.0, "p_two_sided": 1.0, "direction": "no trend", "n": int(n)}

    # S statistic
    S = 0
    for i in range(n - 1):
        S += int(np.sum(np.sign(y[i + 1:] - y[i])))

    # Variance with tie correction
    _, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0

    if var_s <= 0:
        return {"S": float(S), "z": 0.0, "p_two_sided": 1.0, "direction": "no trend", "n": int(n)}

    if S > 0:
        z = (S - 1) / np.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # Two-sided p-value from standard normal
    from math import erfc
    p = erfc(abs(z) / np.sqrt(2))

    if p < 0.05 and S > 0:
        direction = "increasing"
    elif p < 0.05 and S < 0:
        direction = "decreasing"
    else:
        direction = "no trend"

    return {"S": float(S), "z": float(z), "p_two_sided": float(p),
            "direction": direction, "n": int(n)}


def theil_sen_slope(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Theil-Sen median slope — robust alternative to OLS.

    Slope is the median of all pairwise (y_j - y_i) / (x_j - x_i).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return {"slope": np.nan, "intercept": np.nan, "n": int(n)}

    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    if not slopes:
        return {"slope": np.nan, "intercept": np.nan, "n": int(n)}
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return {"slope": slope, "intercept": intercept, "n": int(n)}


# ============================================================================
# 3. Window-vs-baseline comparison
# ============================================================================

@dataclass
class AnomalyResult:
    """Result of comparing a window to a historical baseline."""
    variable: str
    window_label: str
    window_value: float
    baseline_mean: float
    baseline_std: float
    anomaly: float            # window_value - baseline_mean
    z_score: float            # anomaly / baseline_std
    classification: str       # "much above" / "above" / "normal" / "below" / "much below"


_CLASSIFICATION_THRESHOLDS = [
    (1.5, "much above"),
    (0.5, "above"),
    (-0.5, "normal"),
    (-1.5, "below"),
]


def classify_anomaly(z: float) -> str:
    """Map a z-score to a plain-English label."""
    if not np.isfinite(z):
        return "unknown"
    if z >= 1.5:
        return "much above"
    if z >= 0.5:
        return "above"
    if z >= -0.5:
        return "normal"
    if z >= -1.5:
        return "below"
    return "much below"


def compare_window_to_baseline(
    window_values: Sequence[float],
    baseline_mean: float,
    baseline_std: float,
    variable: str,
    window_label: str,
) -> AnomalyResult:
    """Aggregate a window's values (mean) and contrast to the climatology."""
    x = np.asarray(window_values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return AnomalyResult(variable, window_label, np.nan, baseline_mean, baseline_std,
                             np.nan, np.nan, "unknown")
    window = float(x.mean())
    anomaly = window - baseline_mean
    z = anomaly / baseline_std if baseline_std and baseline_std > 0 else np.nan
    return AnomalyResult(
        variable=variable, window_label=window_label,
        window_value=window, baseline_mean=baseline_mean, baseline_std=baseline_std,
        anomaly=anomaly, z_score=z, classification=classify_anomaly(z),
    )


# ============================================================================
# 4. Forecast anomalies
# ============================================================================

def forecast_anomalies(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    target_to_hist_col: Dict[str, str],
    smooth_window: int = 7,
    group_col: str = "City",
) -> pd.DataFrame:
    """Attach climatology mean/std and z-score to every forecast row.

    Parameters
    ----------
    forecast
        Long-format frame with columns ``City, forecast_date, target, y_pred``.
    history
        Daily history frame used to build the daily climatology.
    target_to_hist_col
        Maps forecast target names to the corresponding daily-history column,
        e.g. ``{"temperature_2m": "temperature_2m_mean",
                "wind_speed_10m": "wind_speed_10m_mean",
                "rain": "rain_sum", ...}``.
    """
    enriched = []
    for target, hist_col in target_to_hist_col.items():
        if hist_col not in history.columns:
            logger.warning("History col %s missing; skipping %s", hist_col, target)
            continue
        clim = daily_climatology(history, hist_col, group_col=group_col,
                                 smooth_window=smooth_window)
        sub = forecast[forecast["target"] == target].copy()
        sub["doy"] = sub["forecast_date"].dt.dayofyear
        merged = sub.merge(clim, on=[group_col, "doy"], how="left")
        merged["anomaly"] = merged["y_pred"] - merged["clim_mean"]
        merged["z_score"] = merged["anomaly"] / merged["clim_std"].replace(0, np.nan)
        merged["classification"] = merged["z_score"].apply(classify_anomaly)
        enriched.append(merged)

    if not enriched:
        return pd.DataFrame()

    out = pd.concat(enriched, ignore_index=True)
    logger.info("Forecast anomalies: %d rows across %d targets",
                len(out), len(enriched))
    return out


# ============================================================================
# 5. Seasonal-shift detection
# ============================================================================

def detect_seasonal_shift(
    history: pd.DataFrame,
    variable: str,
    group_col: str = "City",
    date_col: str = "date",
) -> pd.DataFrame:
    """Day-of-year of the annual peak per (city, year). A monotone shift in
    this series across years is evidence of seasonal migration.
    """
    if variable not in history.columns:
        raise ValueError(f"{variable} not in frame")

    work = history[[group_col, date_col, variable]].copy()
    work["year"] = work[date_col].dt.year
    work["doy"] = work[date_col].dt.dayofyear

    # Smooth within year before finding argmax, so a single anomaly doesn't dominate
    rows: List[Dict] = []
    for (city, year), g in work.groupby([group_col, "year"]):
        if len(g) < 200:   # partial years excluded
            continue
        s = g.set_index("doy")[variable].sort_index()
        s_smooth = s.rolling(15, center=True, min_periods=5).mean()
        peak_doy = int(s_smooth.idxmax()) if s_smooth.notna().any() else None
        trough_doy = int(s_smooth.idxmin()) if s_smooth.notna().any() else None
        rows.append({"City": city, "year": year,
                     "peak_doy": peak_doy, "trough_doy": trough_doy,
                     "peak_value": float(s_smooth.max()),
                     "trough_value": float(s_smooth.min())})
    out = pd.DataFrame(rows)
    logger.info("Seasonal peaks for %s: %d city-years", variable, len(out))
    return out


# ============================================================================
# 6. Orchestrator
# ============================================================================

# Map forecast target keys -> daily-history columns + sensible aggregation
TARGET_AGG_MAP: Dict[str, Tuple[str, str]] = {
    "temperature_2m":     ("temperature_2m_mean",  "mean"),
    "wind_speed_10m":     ("wind_speed_10m_mean",  "mean"),
    "wind_direction_10m": ("wind_direction_10m",   "mean"),
    "rain":               ("rain_sum",             "sum"),
    "precipitation":      ("precipitation_sum",    "sum"),
}


def run_climate_analysis(
    history_path: Optional[Path] = None,
    forecast_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Orchestrate every climate comparison and persist a bundle of reports.

    Outputs (to ``output_dir``):
        * climatology_daily.csv
        * climatology_monthly.csv
        * annual_trends.csv
        * seasonal_peaks.csv
        * forecast_anomalies.csv
        * headline_answers.csv   (the 5 project questions, answered)
    """
    history_path = history_path or (INTERIM_DIR / "weather_daily_clean.csv")
    forecast_path = forecast_path or (PROCESSED_DIR / "weather_forecast.csv")
    output_dir = output_dir or (REPORTS_DIR / "climate")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("PHASE 3 - Climate analysis")
    logger.info("=" * 72)

    history = pd.read_csv(history_path, parse_dates=["date"])
    if history["date"].dt.tz is not None:
        history["date"] = history["date"].dt.tz_localize(None)

    forecast = pd.read_csv(forecast_path, parse_dates=["anchor_date", "forecast_date"])
    if forecast["forecast_date"].dt.tz is not None:
        forecast["forecast_date"] = forecast["forecast_date"].dt.tz_localize(None)

    outputs: Dict[str, pd.DataFrame] = {}

    # --- 1. Climatologies -------------------------------------------------
    daily_frames = []
    monthly_frames = []
    for tgt, (col, agg) in TARGET_AGG_MAP.items():
        if col not in history.columns:
            continue
        d = daily_climatology(history, col).assign(variable=col)
        m = monthly_climatology(history, col, agg=agg).assign(variable=col, agg=agg)
        daily_frames.append(d); monthly_frames.append(m)

    daily_clim = pd.concat(daily_frames, ignore_index=True)
    monthly_clim = pd.concat(monthly_frames, ignore_index=True)
    daily_clim.to_csv(output_dir / "climatology_daily.csv", index=False)
    monthly_clim.to_csv(output_dir / "climatology_monthly.csv", index=False)
    outputs["climatology_daily"] = daily_clim
    outputs["climatology_monthly"] = monthly_clim

    # --- 2. Annual trends -------------------------------------------------
    trend_rows: List[Dict] = []
    for tgt, (col, agg) in TARGET_AGG_MAP.items():
        if col not in history.columns:
            continue
        annual = annual_summary(history, col, agg=agg)
        val_col = f"{col}_{agg}"
        for city, g in annual.groupby("City"):
            mk = mann_kendall_trend(g[val_col].values)
            ts = theil_sen_slope(g["year"].values, g[val_col].values)
            trend_rows.append({
                "City": city, "variable": col, "agg": agg,
                "years": int(g["year"].nunique()),
                "min_year": int(g["year"].min()), "max_year": int(g["year"].max()),
                "mean": float(g[val_col].mean()),
                "mk_z": mk["z"], "mk_p": mk["p_two_sided"], "mk_direction": mk["direction"],
                "ts_slope_per_year": ts["slope"], "ts_intercept": ts["intercept"],
            })
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(output_dir / "annual_trends.csv", index=False)
    outputs["annual_trends"] = trends

    # --- 3. Seasonal shift of temperature peaks ---------------------------
    seasonal = detect_seasonal_shift(history, "temperature_2m_mean")
    seasonal.to_csv(output_dir / "seasonal_peaks.csv", index=False)
    outputs["seasonal_peaks"] = seasonal

    # --- 4. Forecast-vs-climatology anomalies -----------------------------
    fc_anom = forecast_anomalies(
        forecast, history,
        target_to_hist_col={k: v[0] for k, v in TARGET_AGG_MAP.items()
                             if v[0] in history.columns},
    )
    fc_anom.to_csv(output_dir / "forecast_anomalies.csv", index=False)
    outputs["forecast_anomalies"] = fc_anom

    # --- 5. Headline answers ----------------------------------------------
    headlines = _summarise_headline_answers(history, forecast, trends, fc_anom)
    headlines.to_csv(output_dir / "headline_answers.csv", index=False)
    outputs["headline_answers"] = headlines

    logger.info("Climate analysis complete; %d report files in %s",
                len(outputs), output_dir)
    return outputs


def _summarise_headline_answers(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    trends: pd.DataFrame,
    fc_anom: pd.DataFrame,
) -> pd.DataFrame:
    """Answer the five project questions using the artefacts we just built.

    Returns a tidy DataFrame with columns: ``question, city, finding, details``.
    """
    rows: List[Dict] = []

    # Q1: Will summer be hotter?  -> use forecast-window anomaly + annual trend
    for city in sorted(forecast["City"].unique()):
        fc_city = fc_anom[(fc_anom["City"] == city) &
                          (fc_anom["target"] == "temperature_2m")]
        if fc_city.empty:
            continue
        mean_z = fc_city["z_score"].mean()
        label = classify_anomaly(mean_z)
        rows.append({
            "question": "Will the next 30 days be hotter than normal?",
            "city": city,
            "finding": label,
            "details": f"mean z={mean_z:+.2f} across {len(fc_city)} forecast days",
        })

    # Q2: Is the annual mean temperature increasing?
    temp_trends = trends[trends["variable"] == "temperature_2m_mean"]
    for _, r in temp_trends.iterrows():
        rows.append({
            "question": "Is annual mean temperature rising?",
            "city": r["City"],
            "finding": r["mk_direction"],
            "details": f"Theil-Sen slope = {r['ts_slope_per_year']:+.3f} degC/yr over {r['years']} yrs, MK p={r['mk_p']:.3f}",
        })

    # Q3: Is annual rainfall changing?
    rain_trends = trends[trends["variable"] == "rain_sum"]
    for _, r in rain_trends.iterrows():
        rows.append({
            "question": "Is annual rainfall changing?",
            "city": r["City"],
            "finding": r["mk_direction"],
            "details": f"Theil-Sen slope = {r['ts_slope_per_year']:+.2f} mm/yr over {r['years']} yrs, MK p={r['mk_p']:.3f}",
        })

    # Q4: Will the next 30 days be wetter or drier than normal?
    for city in sorted(forecast["City"].unique()):
        fc_city = fc_anom[(fc_anom["City"] == city) &
                          (fc_anom["target"] == "precipitation")]
        if fc_city.empty:
            continue
        mean_z = fc_city["z_score"].mean()
        # For precipitation, +z means MORE rain than normal (i.e. WETTER).
        # classify_anomaly maps z directly to "much above"/"above"/"normal"/...
        # We relabel those to wetter/drier for clarity.
        raw_label = classify_anomaly(mean_z)
        verbal_map = {
            "much above": "much wetter than normal",
            "above": "wetter than normal",
            "normal": "near normal",
            "below": "drier than normal",
            "much below": "much drier than normal",
            "unknown": "unknown",
        }
        rows.append({
            "question": "Will the next 30 days be wetter or drier than normal?",
            "city": city,
            "finding": verbal_map.get(raw_label, raw_label),
            "details": f"precipitation mean z={mean_z:+.2f} over {len(fc_city)} forecast days",
        })

    # Q5: Is wind getting stronger? (agricultural/fire relevance)
    wind_trends = trends[trends["variable"] == "wind_speed_10m_mean"]
    for _, r in wind_trends.iterrows():
        rows.append({
            "question": "Is mean wind speed changing?",
            "city": r["City"],
            "finding": r["mk_direction"],
            "details": f"Theil-Sen slope = {r['ts_slope_per_year']:+.3f} m/s/yr over {r['years']} yrs, MK p={r['mk_p']:.3f}",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_climate_analysis()
