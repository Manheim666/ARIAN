"""
src/grid/builder.py
───────────────────
Builds the canonical 5 km grid of Azerbaijan in EPSG:32638 and clips it to
the country boundary (read from the forest KMZ's country shape — or a fallback
bbox if the KMZ doesn't contain a country polygon).

The grid is the *spine* of the entire feature store. Every dynamic feature
(weather, NDVI, fire count, lightning) gets joined onto it by `cell_id`.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from src.config.settings import Settings, get_settings
from src.utils.geo import build_regular_grid, ensure_crs, reproject_bbox
from src.utils.io import write_geoparquet

log = logging.getLogger(__name__)


class GridBuilder:
    """
    Single-responsibility: produce `grid_cells.parquet`, a GeoParquet of
    ~3 500 square cells covering AZ, each with a stable cell_id, lat/lon
    centroid (WGS84), and metric centroid (UTM 38N).
    """

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    # ── public ─────────────────────────────────────────────────────────
    def build(self, mask_to_country: bool = True) -> gpd.GeoDataFrame:
        # 1. project bbox to metric CRS
        bbox_m = reproject_bbox(
            self.s.bbox_wgs84,
            self.s.crs_wgs84,
            self.s.crs_working,
        )
        log.info("Projected AZ bbox (EPSG:%s): %s", self.s.crs_working, bbox_m)

        # 2. build the raw grid
        grid = build_regular_grid(
            bbox_projected=bbox_m,
            cell_size_m=self.s.grid_resolution_m,
            crs=self.s.crs_working,
        )
        log.info("Raw grid: %d cells", len(grid))

        # 3. clip to country boundary (optional — strongly recommended)
        if mask_to_country:
            country = self._load_country_polygon()
            if country is not None:
                country = ensure_crs(country, self.s.crs_working)
                grid = gpd.sjoin(
                    grid, country[["geometry"]],
                    how="inner", predicate="intersects",
                ).drop(columns=["index_right"])
                grid = grid.reset_index(drop=True)
                grid["cell_id"] = grid.index.astype("int64")
                log.info("Clipped to country: %d cells", len(grid))

        # 4. enrich with WGS84 centroid (needed for Open-Meteo lat/lon calls)
        centroids_wgs = grid.geometry.centroid.to_crs(self.s.crs_wgs84)
        grid["lat"] = centroids_wgs.y.astype("float32")
        grid["lon"] = centroids_wgs.x.astype("float32")

        return grid

    def save(self, grid: gpd.GeoDataFrame) -> Path:
        out = self.s.abs(self.s.silver_dir) / "grid" / "grid_cells.parquet"
        write_geoparquet(grid, out)
        log.info("Wrote grid: %s (%d cells)", out, len(grid))
        return out

    # ── helpers ────────────────────────────────────────────────────────
    def _load_country_polygon(self) -> gpd.GeoDataFrame | None:
        """
        Read the country outline from the KMZ forest file. KMZ is a zipped
        KML — geopandas/fiona reads it natively if the kml driver is enabled.

        If the KMZ contains multiple polygons (forest patches, not country
        border), we dissolve to a single multipolygon and return the convex
        hull as a fallback country shape. For production, replace this with
        the official GADM-level-0 AZ polygon.
        """
        kmz_path = self.s.abs(self.s.forest_kmz)
        if not kmz_path.exists():
            log.warning("Forest KMZ not found — building grid from bbox only")
            # fallback: bbox polygon
            min_lon, min_lat, max_lon, max_lat = self.s.bbox_wgs84
            poly = box(min_lon, min_lat, max_lon, max_lat)
            return gpd.GeoDataFrame({"geometry": [poly]}, crs=self.s.crs_wgs84)

        try:
            # Extract the KML inside the KMZ to read
            with zipfile.ZipFile(kmz_path) as z:
                kml_name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
                if kml_name is None:
                    raise RuntimeError(f"No .kml inside {kmz_path}")
                kml_bytes = z.read(kml_name)

            # write KML to a temp buffer pyogrio can read
            tmp = kmz_path.with_suffix(".kml.tmp")
            tmp.write_bytes(kml_bytes)
            try:
                gdf = gpd.read_file(tmp, engine="pyogrio")
            finally:
                tmp.unlink(missing_ok=True)

            if gdf.crs is None:
                gdf.set_crs(self.s.crs_wgs84, inplace=True)
            # dissolve to one multi-polygon representing the country extent
            dissolved = gdf.dissolve().convex_hull
            return gpd.GeoDataFrame({"geometry": dissolved}, crs=gdf.crs)
        except Exception as exc:  # noqa: BLE001
            log.warning("KMZ load failed (%s) — using bbox fallback", exc)
            min_lon, min_lat, max_lon, max_lat = self.s.bbox_wgs84
            poly = box(min_lon, min_lat, max_lon, max_lat)
            return gpd.GeoDataFrame({"geometry": [poly]}, crs=self.s.crs_wgs84)
