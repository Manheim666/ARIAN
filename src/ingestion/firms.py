"""
src/ingestion/firms.py
──────────────────────
NASA FIRMS active-fire ingestor.

FIRMS gotchas addressed here:
  1. Three sensors (MODIS, VIIRS-SNPP, VIIRS-NOAA20) report overlapping
     detections — if left raw, the same fire is counted 3×. We dedupe
     spatiotemporally (500 m × 30 min window).
  2. Confidence semantics differ by sensor:
        MODIS   → 0–100 integer
        VIIRS   → {'l','n','h'} string
     We unify to a 0–100 scale.
  3. CRS: FIRMS ships WGS84 lon/lat → we reproject to `crs_working` (UTM 38N).
  4. Column casing is inconsistent across archive vintages ('acq_date'
     vs 'ACQ_DATE') — we normalise to lowercase snake_case.

This loader produces bronze-layer GeoDataFrames, one per year.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.ingestion.base import BaseIngestor
from src.utils.geo import ensure_crs, points_from_latlon, validate_within_bbox

log = logging.getLogger(__name__)


_VIIRS_CONF_MAP = {"l": 25, "n": 60, "h": 90}


class FirmsIngestor(BaseIngestor):
    source_name = "firms"

    def discover(self) -> list[str]:
        """Each CSV in the FIRMS dir becomes one partition (keyed by file stem)."""
        firms_dir = self.settings.abs(self.settings.firms_dir)
        if not firms_dir.exists():
            log.warning("FIRMS dir not found: %s", firms_dir)
            return []
        files = sorted(firms_dir.glob("*.csv"))
        return [f.stem for f in files]

    def _ingest_one(self, partition_key: str) -> gpd.GeoDataFrame:
        csv_path = self.settings.abs(self.settings.firms_dir) / f"{partition_key}.csv"

        # Read with dtype hints — FIRMS CSVs are 500k+ rows per year
        df = pd.read_csv(
            csv_path,
            dtype={
                "latitude": "float32",
                "longitude": "float32",
                "brightness": "float32",
                "bright_t31": "float32",
                "bright_ti4": "float32",
                "bright_ti5": "float32",
                "frp": "float32",
                "scan": "float32",
                "track": "float32",
                "satellite": "category",
                "instrument": "category",
                "confidence": "object",       # mixed across sensors
                "version": "category",
                "daynight": "category",
                "type": "Int8",
            },
            parse_dates=["acq_date"],
            low_memory=False,
        )

        # ── normalise column names ─────────────────────────────────────
        df.columns = [c.lower().strip() for c in df.columns]

        # ── unify confidence to 0–100 int ───────────────────────────────
        conf = df["confidence"].astype(str).str.strip().str.lower()
        numeric = pd.to_numeric(conf, errors="coerce")
        # VIIRS rows where it's l/n/h
        fallback = conf.map(_VIIRS_CONF_MAP)
        df["confidence_int"] = numeric.fillna(fallback).astype("Int16")

        # ── derive timestamp ───────────────────────────────────────────
        # acq_time is HHMM as int (0-2359) in UTC
        hhmm = df["acq_time"].astype(str).str.zfill(4)
        df["acq_datetime_utc"] = pd.to_datetime(
            df["acq_date"].dt.strftime("%Y-%m-%d") + " " + hhmm.str[:2] + ":" + hhmm.str[2:],
            utc=True,
            errors="coerce",
        )

        # ── filter by confidence ───────────────────────────────────────
        before = len(df)
        df = df[df["confidence_int"] >= self.settings.firms_confidence_min]
        log.info(
            "FIRMS %s: %d / %d rows retained after confidence ≥ %d",
            partition_key, len(df), before, self.settings.firms_confidence_min,
        )

        # ── dedupe across sensors ──────────────────────────────────────
        df = self._spatiotemporal_dedupe(df)

        # ── filter to AZ bbox (tight) ──────────────────────────────────
        min_lon, min_lat, max_lon, max_lat = self.settings.bbox_wgs84
        df = df[
            df["longitude"].between(min_lon, max_lon) &
            df["latitude"].between(min_lat, max_lat)
        ].reset_index(drop=True)

        if df.empty:
            log.warning("FIRMS %s: 0 rows after AZ bbox filter", partition_key)
            return gpd.GeoDataFrame(
                columns=["acq_datetime_utc", "frp", "confidence_int", "geometry"],
                geometry="geometry",
                crs=self.settings.crs_working,
            )

        # ── build GeoDataFrame in WGS84, then reproject ────────────────
        geometry = points_from_latlon(df["latitude"], df["longitude"], self.settings.crs_wgs84)
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.settings.crs_wgs84)

        # sanity check: nothing should lie outside AZ
        validate_within_bbox(gdf, self.settings.bbox_wgs84, tolerance_pct=0.01)

        gdf = ensure_crs(gdf, self.settings.crs_working)

        keep_cols = [
            "acq_datetime_utc", "acq_date", "acq_time",
            "latitude", "longitude",
            "brightness", "bright_t31", "bright_ti4", "bright_ti5",
            "frp", "confidence_int", "satellite", "instrument",
            "daynight", "type", "geometry",
        ]
        keep_cols = [c for c in keep_cols if c in gdf.columns]
        return gdf[keep_cols]

    # ── dedupe: same fire seen by multiple sensors within 500m & 30min
    @staticmethod
    def _spatiotemporal_dedupe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # coarse gridding trick: floor lat/lon to ~500 m, floor time to 30 min.
        # Dedup key is the tuple; within a key keep the highest-confidence row.
        # This is O(N log N), avoids a full spatial-join.
        lat_bucket = np.floor(df["latitude"] * 222.0).astype("int32")   # 1°≈111km → 500m = 1/222°
        lon_bucket = np.floor(df["longitude"] * 222.0).astype("int32")
        time_bucket = (df["acq_datetime_utc"].astype("int64") // (30 * 60 * 10**9)).astype("int64")
        df = df.assign(_k=lat_bucket.astype(str) + "_" +
                          lon_bucket.astype(str) + "_" +
                          time_bucket.astype(str))
        df = df.sort_values("confidence_int", ascending=False)
        df = df.drop_duplicates("_k", keep="first").drop(columns="_k")
        return df.sort_values("acq_datetime_utc").reset_index(drop=True)
