from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


import pandas as pd

try:
    import duckdb
except Exception as exc:  # pragma: no cover
    duckdb = None
    _DUCKDB_IMPORT_ERROR = exc
else:
    _DUCKDB_IMPORT_ERROR = None


@dataclass(frozen=True)
class DuckDBConfig:
    path: Path


class DuckDBDatabase:
    def __init__(self, cfg: DuckDBConfig):
        if duckdb is None:
            raise RuntimeError(
                "duckdb is required but not installed. "
                f"Original error: {_DUCKDB_IMPORT_ERROR}"
            )
        self.cfg = cfg

    def connect(self):
        return duckdb.connect(str(self.cfg.path))

    def init_schema(self) -> None:
        with self.connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS dim_city (
                    city_id INTEGER PRIMARY KEY,
                    city_name VARCHAR NOT NULL UNIQUE,
                    latitude DOUBLE NOT NULL,
                    longitude DOUBLE NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dim_date (
                    date_id INTEGER PRIMARY KEY,
                    date DATE NOT NULL UNIQUE,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    day INTEGER NOT NULL,
                    day_of_year INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fact_weather (
                    city_id INTEGER NOT NULL,
                    date_id INTEGER NOT NULL,
                    avg_temperature_c DOUBLE,
                    max_temperature_c DOUBLE,
                    min_temperature_c DOUBLE,
                    total_precip_mm DOUBLE,
                    avg_wind_speed_kmh DOUBLE,
                    max_wind_speed_kmh DOUBLE,
                    PRIMARY KEY (city_id, date_id)
                );

                CREATE TABLE IF NOT EXISTS fact_fire (
                    city_id INTEGER NOT NULL,
                    date_id INTEGER NOT NULL,
                    fire_count INTEGER,
                    mean_brightness DOUBLE,
                    max_frp DOUBLE,
                    fire_occurred BOOLEAN,
                    PRIMARY KEY (city_id, date_id)
                );

                CREATE TABLE IF NOT EXISTS pipeline_state (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR
                );
                """
            )

    def upsert_dim_city(self, rows: Iterable[dict[str, Any]]) -> None:
        df_rows = pd.DataFrame(list(rows))
        if df_rows.empty:
            return
        with self.connect() as con:
            con.execute(
                """
                CREATE TEMP TABLE _city_stage AS
                SELECT * FROM (SELECT 1 AS city_id, ''::VARCHAR AS city_name, 0.0::DOUBLE AS latitude, 0.0::DOUBLE AS longitude) WHERE 1=0;
                """
            )
            con.register("_city_rows", df_rows)
            con.execute(
                """
                INSERT INTO _city_stage
                SELECT city_id, city_name, latitude, longitude FROM _city_rows;
                """
            )
            con.execute(
                """
                INSERT INTO dim_city AS t
                SELECT * FROM _city_stage
                ON CONFLICT (city_id) DO UPDATE SET
                    city_name = excluded.city_name,
                    latitude = excluded.latitude,
                    longitude = excluded.longitude;
                """
            )

    def upsert_dim_date(self, rows: Iterable[dict[str, Any]]) -> None:
        df_rows = pd.DataFrame(list(rows))
        if df_rows.empty:
            return
        with self.connect() as con:
            con.execute(
                """
                CREATE TEMP TABLE _date_stage AS
                SELECT * FROM (SELECT 1 AS date_id, DATE '2000-01-01' AS date, 2000 AS year, 1 AS month, 1 AS day, 1 AS day_of_year) WHERE 1=0;
                """
            )
            con.register("_date_rows", df_rows)
            con.execute("INSERT INTO _date_stage SELECT * FROM _date_rows;")
            con.execute(
                """
                INSERT INTO dim_date AS t
                SELECT * FROM _date_stage
                ON CONFLICT (date_id) DO UPDATE SET
                    date = excluded.date,
                    year = excluded.year,
                    month = excluded.month,
                    day = excluded.day,
                    day_of_year = excluded.day_of_year;
                """
            )

    def upsert_fact_weather(self, rows: Iterable[dict[str, Any]]) -> None:
        df_rows = pd.DataFrame(list(rows))
        if df_rows.empty:
            return
        with self.connect() as con:
            con.execute(
                """
                CREATE TEMP TABLE _weather_stage AS
                SELECT * FROM (
                    SELECT 1::INTEGER AS city_id, 1::INTEGER AS date_id,
                           NULL::DOUBLE AS avg_temperature_c, NULL::DOUBLE AS max_temperature_c, NULL::DOUBLE AS min_temperature_c,
                           NULL::DOUBLE AS total_precip_mm,
                           NULL::DOUBLE AS avg_wind_speed_kmh, NULL::DOUBLE AS max_wind_speed_kmh
                ) WHERE 1=0;
                """
            )
            con.register("_weather_rows", df_rows)
            con.execute("INSERT INTO _weather_stage SELECT * FROM _weather_rows;")
            con.execute(
                """
                INSERT INTO fact_weather AS t
                SELECT * FROM _weather_stage
                ON CONFLICT (city_id, date_id) DO UPDATE SET
                    avg_temperature_c = excluded.avg_temperature_c,
                    max_temperature_c = excluded.max_temperature_c,
                    min_temperature_c = excluded.min_temperature_c,
                    total_precip_mm = excluded.total_precip_mm,
                    avg_wind_speed_kmh = excluded.avg_wind_speed_kmh,
                    max_wind_speed_kmh = excluded.max_wind_speed_kmh;
                """
            )

    def upsert_fact_fire(self, rows: Iterable[dict[str, Any]]) -> None:
        df_rows = pd.DataFrame(list(rows))
        if df_rows.empty:
            return
        with self.connect() as con:
            con.execute(
                """
                CREATE TEMP TABLE _fire_stage AS
                SELECT * FROM (
                    SELECT 1::INTEGER AS city_id, 1::INTEGER AS date_id,
                           NULL::INTEGER AS fire_count,
                           NULL::DOUBLE AS mean_brightness,
                           NULL::DOUBLE AS max_frp,
                           NULL::BOOLEAN AS fire_occurred
                ) WHERE 1=0;
                """
            )
            con.register("_fire_rows", df_rows)
            con.execute("INSERT INTO _fire_stage SELECT * FROM _fire_rows;")
            con.execute(
                """
                INSERT INTO fact_fire AS t
                SELECT * FROM _fire_stage
                ON CONFLICT (city_id, date_id) DO UPDATE SET
                    fire_count = excluded.fire_count,
                    mean_brightness = excluded.mean_brightness,
                    max_frp = excluded.max_frp,
                    fire_occurred = excluded.fire_occurred;
                """
            )

    def get_state(self, key: str) -> str | None:
        with self.connect() as con:
            res = con.execute("SELECT value FROM pipeline_state WHERE key = ?", [key]).fetchone()
            return None if res is None else res[0]

    def set_state(self, key: str, value: str) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO pipeline_state AS t(key, value)
                VALUES (?, ?)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value;
                """,
                [key, value],
            )

    def validation_queries(self) -> dict[str, Any]:
        with self.connect() as con:
            out: dict[str, Any] = {}
            out["city_count"] = con.execute("SELECT COUNT(*) FROM dim_city").fetchone()[0]
            out["date_count"] = con.execute("SELECT COUNT(*) FROM dim_date").fetchone()[0]
            out["weather_rows"] = con.execute("SELECT COUNT(*) FROM fact_weather").fetchone()[0]
            out["fire_rows"] = con.execute("SELECT COUNT(*) FROM fact_fire").fetchone()[0]
            out["weather_pk_dupes"] = con.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT city_id, date_id, COUNT(*) c
                    FROM fact_weather
                    GROUP BY 1,2
                    HAVING c > 1
                )
                """
            ).fetchone()[0]
            out["fire_pk_dupes"] = con.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT city_id, date_id, COUNT(*) c
                    FROM fact_fire
                    GROUP BY 1,2
                    HAVING c > 1
                )
                """
            ).fetchone()[0]
            return out
