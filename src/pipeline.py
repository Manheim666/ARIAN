from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .analysis import test_recent_vs_past, test_temp_vs_wildfire, test_temperature_trend
from .cleaning import clean_daily_weather
from .config import CITIES, PipelineConfig, detect_project_root, project_paths
from .database import DuckDBConfig, DuckDBDatabase
from .eda import generate_eda_figures
from .features import add_features_daily
from .ingestion import IngestionPaths, fetch_forecast_hourly, fetch_historical_hourly, load_firms_csvs, normalise_fires_daily
from .logging_utils import setup_logging
from .modeling import train_classification_rain, train_regression_next_day_temperature
from .quality_checks import run_quality_checks_daily, write_data_quality_report


def _aggregate_hourly_to_daily(weather_hourly: pd.DataFrame) -> pd.DataFrame:
    w = weather_hourly.copy()
    w["Timestamp"] = pd.to_datetime(w["Timestamp"], utc=True)
    # Standardize daily keys to naive midnight timestamps for consistent joins
    w["Date"] = w["Timestamp"].dt.floor("D").dt.tz_convert(None)

    agg = {
        "Temperature_C": ["mean", "max", "min"],
        "Precipitation_mm": ["sum"],
        "Wind_Speed_kmh": ["mean", "max"],
    }

    keep = [c for c in ["City", "Latitude", "Longitude", "Date", "Temperature_C", "Precipitation_mm", "Wind_Speed_kmh"] if c in w.columns]
    w = w[keep]

    g = w.groupby(["City", "Latitude", "Longitude", "Date"], as_index=False).agg(
        avg_temperature_c=("Temperature_C", "mean"),
        max_temperature_c=("Temperature_C", "max"),
        min_temperature_c=("Temperature_C", "min"),
        total_precip_mm=("Precipitation_mm", "sum"),
        avg_wind_speed_kmh=("Wind_Speed_kmh", "mean"),
        max_wind_speed_kmh=("Wind_Speed_kmh", "max"),
    )

    g = g.rename(
        columns={
            "avg_temperature_c": "Temperature_C",
            "total_precip_mm": "Precipitation_mm",
            "avg_wind_speed_kmh": "Wind_Speed_kmh",
            "max_temperature_c": "Temp_Max_C",
            "min_temperature_c": "Temp_Min_C",
            "max_wind_speed_kmh": "Wind_Max_kmh",
        }
    )
    return g


def run() -> int:
    cfg = PipelineConfig()
    logger = setup_logging(logging.INFO)

    root = detect_project_root()
    paths = project_paths(root)

    logger.info("ARIAN pipeline starting")
    logger.info(f"Project root: {paths['root']}")

    duckdb_path = paths["data"] / "arian.duckdb"
    db = DuckDBDatabase(DuckDBConfig(path=duckdb_path))
    db.init_schema()

    ingestion_paths = IngestionPaths(raw_dir=paths["raw"], processed_dir=paths["processed"], firms_dir=paths["firms"])

    # Incremental ingestion state
    last_ingested = db.get_state("weather_last_timestamp_utc")
    logger.info(f"Last ingestion timestamp (UTC): {last_ingested}")
    incremental_start: str | None = None
    if last_ingested:
        try:
            # Start from next day (archive API is day-granularity). Safe + idempotent due to dedupe.
            ts = pd.to_datetime(last_ingested, utc=True)
            incremental_start = (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            incremental_start = None

    cache_dir = Path.home() / ".arian_http_cache"

    # 1) Ingest weather hourly (incremental)
    try:
        weather_hourly = fetch_historical_hourly(
            ingestion_paths, cfg, cache_dir=cache_dir, incremental_start=incremental_start
        )
        if not weather_hourly.empty:
            last_ts = pd.to_datetime(weather_hourly["Timestamp"], utc=True).max()
            db.set_state("weather_last_timestamp_utc", last_ts.isoformat())
        logger.info(f"Weather hourly rows: {len(weather_hourly):,}")
    except Exception as exc:
        logger.error(f"Weather ingestion failed: {exc}")
        return 2

    # 2) Forecast (30 days)
    try:
        forecast_hourly = fetch_forecast_hourly(ingestion_paths, cfg, cache_dir=cache_dir)
        logger.info(f"Forecast hourly rows: {len(forecast_hourly):,} (horizon {cfg.forecast_horizon_days}d)")
    except Exception as exc:
        logger.error(f"Forecast ingestion failed: {exc}")
        forecast_hourly = pd.DataFrame()

    # 3) FIRMS ingestion + normalize to daily city labels
    fires_raw = load_firms_csvs(ingestion_paths)
    logger.info(f"FIRMS detections loaded: {len(fires_raw):,}")

    fires_daily = normalise_fires_daily(fires_raw, cfg)
    if not fires_daily.empty:
        fires_daily.to_parquet(ingestion_paths.fires_daily_parquet, index=False)
        logger.info(f"Daily fire rows: {len(fires_daily):,}")
    else:
        logger.info("Daily fire rows: 0")

    # 4) Build daily master from hourly (for modeling + stats)
    daily_weather = _aggregate_hourly_to_daily(weather_hourly)

    # Join fires
    if not fires_daily.empty:
        fires_daily = fires_daily.copy()
        fires_daily["Date"] = pd.to_datetime(fires_daily["Date"]).dt.floor("D")
        d = daily_weather.merge(fires_daily, on=["City", "Date"], how="left")
        d["Fire_Occurred"] = d["Fire_Occurred"].fillna(0)
        d["fire_count"] = d.get("fire_count", 0).fillna(0)
    else:
        d = daily_weather
        d["Fire_Occurred"] = 0
        d["fire_count"] = 0

    # 5) Cleaning
    cleaned, decisions = clean_daily_weather(d.rename(columns={"Temp_Max_C": "Temp_Max_C", "Temp_Min_C": "Temp_Min_C"}))

    # 6) Feature engineering
    featured = add_features_daily(cleaned)

    # 7) Quality checks + report
    quality_summary, quality_issues = run_quality_checks_daily(featured)
    report_path = paths["reports"] / "data_quality_report.md"
    write_data_quality_report(report_path, quality_summary, quality_issues, decisions=decisions)
    logger.info(
        "Data quality summary: "
        f"null={quality_summary.null_violations}, dupes={quality_summary.duplicate_violations}, range={quality_summary.range_violations}"
    )
    logger.info(f"Wrote report: {report_path}")

    # 8) EDA figures
    figs = generate_eda_figures(featured, paths["reports"] / "figures")
    logger.info(f"EDA figures saved: {len(figs)}")

    # 9) Database load (dim/fact)
    # Build dim tables
    city_rows = []
    for i, (city, (lat, lon)) in enumerate(sorted(__import__("src.config", fromlist=["CITIES"]).CITIES.items()), start=1):
        city_rows.append({"city_id": i, "city_name": city, "latitude": float(lat), "longitude": float(lon)})
    city_id_map = {r["city_name"]: r["city_id"] for r in city_rows}

    date_df = featured[["Date"]].drop_duplicates().copy()
    date_df["Date"] = pd.to_datetime(date_df["Date"]).dt.date
    date_df = date_df.sort_values("Date")
    date_rows = []
    for i, dt in enumerate(date_df["Date"].tolist(), start=1):
        ts = pd.to_datetime(dt)
        date_rows.append(
            {
                "date_id": i,
                "date": dt,
                "year": int(ts.year),
                "month": int(ts.month),
                "day": int(ts.day),
                "day_of_year": int(ts.dayofyear),
            }
        )
    date_id_map = {r["date"]: r["date_id"] for r in date_rows}

    db.upsert_dim_city(city_rows)
    db.upsert_dim_date(date_rows)

    # fact_weather
    fw_rows = []
    for _, r in featured.iterrows():
        dt = pd.to_datetime(r["Date"]).date()
        fw_rows.append(
            {
                "city_id": int(city_id_map[r["City"]]),
                "date_id": int(date_id_map[dt]),
                "avg_temperature_c": float(r["Temperature_C"]) if pd.notna(r.get("Temperature_C")) else None,
                "max_temperature_c": float(r.get("Temp_Max_C")) if pd.notna(r.get("Temp_Max_C")) else None,
                "min_temperature_c": float(r.get("Temp_Min_C")) if pd.notna(r.get("Temp_Min_C")) else None,
                "total_precip_mm": float(r.get("Precipitation_mm")) if pd.notna(r.get("Precipitation_mm")) else None,
                "avg_wind_speed_kmh": float(r.get("Wind_Speed_kmh")) if pd.notna(r.get("Wind_Speed_kmh")) else None,
                "max_wind_speed_kmh": float(r.get("Wind_Max_kmh")) if pd.notna(r.get("Wind_Max_kmh")) else None,
            }
        )
    db.upsert_fact_weather(fw_rows)

    # fact_fire
    if "Fire_Occurred" in featured.columns:
        ff_rows = []
        for _, r in featured.iterrows():
            dt = pd.to_datetime(r["Date"]).date()
            ff_rows.append(
                {
                    "city_id": int(city_id_map[r["City"]]),
                    "date_id": int(date_id_map[dt]),
                    "fire_count": int(r.get("fire_count")) if pd.notna(r.get("fire_count")) else None,
                    "mean_brightness": float(r.get("mean_brightness")) if pd.notna(r.get("mean_brightness")) else None,
                    "max_frp": float(r.get("max_frp")) if pd.notna(r.get("max_frp")) else None,
                    "fire_occurred": bool(int(r.get("Fire_Occurred"))) if pd.notna(r.get("Fire_Occurred")) else None,
                }
            )
        db.upsert_fact_fire(ff_rows)

    v = db.validation_queries()
    logger.info(f"DuckDB validation: {v}")

    # 10) Hypothesis tests
    hyp_a = test_temperature_trend(featured)
    hyp_b = test_recent_vs_past(featured)
    hyp_c = test_temp_vs_wildfire(featured)
    logger.info(f"Hypothesis A: {hyp_a.interpretation} (p={hyp_a.p_value:.3g})")
    logger.info(f"Hypothesis B: {hyp_b.interpretation} (p={hyp_b.p_value:.3g})")
    logger.info(f"Hypothesis C: {hyp_c.interpretation} (p={hyp_c.p_value:.3g})")

    hyp_report = paths["reports"] / "hypothesis_report.md"
    hyp_lines = [
        "# Hypothesis Test Report\n\n",
    ]
    for res in [hyp_a, hyp_b, hyp_c]:
        hyp_lines.append(f"## {res.name}\n")
        hyp_lines.append(f"- Method: {res.method}\n")
        hyp_lines.append(f"- Statistic: {res.statistic}\n")
        hyp_lines.append(f"- p-value: {res.p_value}\n")
        hyp_lines.append(f"- Effect size: {res.effect_size}\n")
        hyp_lines.append("- Assumptions:\n")
        for a in res.assumptions:
            hyp_lines.append(f"  - {a}\n")
        hyp_lines.append(f"- Interpretation: {res.interpretation}\n\n")
    hyp_report.write_text("".join(hyp_lines), encoding="utf-8")
    logger.info(f"Wrote hypothesis report: {hyp_report}")

    # 11) Modeling
    reg_res = train_regression_next_day_temperature(featured, figures_dir=paths["reports"] / "figures")
    clf_res = train_classification_rain(featured)
    logger.info(f"Model 1: {reg_res.name} metrics={reg_res.metrics}")
    logger.info(f"Model 2: {clf_res.name} metrics={clf_res.metrics}")

    model_report = paths["reports"] / "model_report.md"
    model_lines = [
        "# Modeling Report\n\n",
        f"## {reg_res.name}\n",
        f"- Metrics: {reg_res.metrics}\n\n",
        f"## {clf_res.name}\n",
        f"- Metrics: {clf_res.metrics}\n",
    ]
    model_report.write_text("".join(model_lines), encoding="utf-8")
    logger.info(f"Wrote model report: {model_report}")

    logger.info("Trust evaluation: see reports/data_quality_report.md")
    logger.info("ARIAN pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
