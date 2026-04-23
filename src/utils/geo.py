"""
src/utils/geo.py
────────────────
Low-level geospatial primitives used everywhere downstream.

Philosophy
──────────
* Never reproject on the fly inside hot loops. Reproject once at ingestion
  boundary, store everything in `crs_working`.
* `shapely` 2.x vectorised ops only; no Python-level iteration over geometries.
* Every function is pure — no file I/O, no globals. Makes unit testing trivial.
"""
from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import numpy as np
import pyproj
import shapely
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry


# ─────────────────────────────────────────────────────────────────────────
#  CRS transforms
# ─────────────────────────────────────────────────────────────────────────
def make_transformer(src_crs: str, dst_crs: str) -> pyproj.Transformer:
    """
    Build an *always_xy* transformer (lon/lat order, not lat/lon).
    Cached inside pyproj under the hood.
    """
    return pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def reproject_bbox(
    bbox: tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float, float, float]:
    """
    Reproject a bounding box by transforming its 4 corners AND densifying
    each edge — naive 4-corner transform under-estimates the true extent
    for curved projections.
    """
    tr = make_transformer(src_crs, dst_crs)
    min_lon, min_lat, max_lon, max_lat = bbox

    # Densify: 25 points per edge — cheap, numerically safe
    lons = np.concatenate([
        np.linspace(min_lon, max_lon, 25),
        np.full(25, max_lon),
        np.linspace(max_lon, min_lon, 25),
        np.full(25, min_lon),
    ])
    lats = np.concatenate([
        np.full(25, min_lat),
        np.linspace(min_lat, max_lat, 25),
        np.full(25, max_lat),
        np.linspace(max_lat, min_lat, 25),
    ])
    xs, ys = tr.transform(lons, lats)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


# ─────────────────────────────────────────────────────────────────────────
#  GeoDataFrame helpers
# ─────────────────────────────────────────────────────────────────────────
def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """
    Reproject `gdf` to `target_crs` iff needed. No-op when already correct.
    Raises if `gdf.crs` is None (most common foot-gun).
    """
    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame has no CRS declared. Refusing to reproject a "
            "CRS-less frame — set gdf.crs = 'EPSG:4326' (or whatever is "
            "correct for the source) before calling ensure_crs."
        )
    if gdf.crs.to_string() == target_crs:
        return gdf
    return gdf.to_crs(target_crs)


def points_from_latlon(
    lats: Iterable[float],
    lons: Iterable[float],
    crs: str = "EPSG:4326",
) -> gpd.GeoSeries:
    """Vectorised point creation. O(N), no Python loop."""
    lats = np.asarray(lats, dtype="float64")
    lons = np.asarray(lons, dtype="float64")
    return gpd.GeoSeries(shapely.points(lons, lats), crs=crs)


# ─────────────────────────────────────────────────────────────────────────
#  Grid generation
# ─────────────────────────────────────────────────────────────────────────
def build_regular_grid(
    bbox_projected: tuple[float, float, float, float],
    cell_size_m: int,
    crs: str,
) -> gpd.GeoDataFrame:
    """
    Generate a regular axis-aligned grid of square polygons covering `bbox`.
    `bbox` must already be in a *metric* CRS (UTM, etc.) — passing lon/lat
    here yields degrees-wide cells which is almost certainly a bug.

    Returns columns: cell_id (int), geometry, x_idx, y_idx, centroid_x,
    centroid_y, area_m2.
    """
    xmin, ymin, xmax, ymax = bbox_projected

    # snap bbox outward to cell boundaries so indexing is deterministic
    xmin = np.floor(xmin / cell_size_m) * cell_size_m
    ymin = np.floor(ymin / cell_size_m) * cell_size_m
    xmax = np.ceil(xmax / cell_size_m) * cell_size_m
    ymax = np.ceil(ymax / cell_size_m) * cell_size_m

    xs = np.arange(xmin, xmax, cell_size_m)
    ys = np.arange(ymin, ymax, cell_size_m)

    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    xv = xv.ravel()
    yv = yv.ravel()

    # shapely 2.x vectorised box construction via polygons()
    polys = shapely.polygons(
        np.stack([
            np.column_stack([xv,               yv]),
            np.column_stack([xv + cell_size_m, yv]),
            np.column_stack([xv + cell_size_m, yv + cell_size_m]),
            np.column_stack([xv,               yv + cell_size_m]),
            np.column_stack([xv,               yv]),
        ], axis=1)
    )

    gdf = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(len(polys), dtype="int64"),
            "x_idx": ((xv - xmin) / cell_size_m).astype("int32"),
            "y_idx": ((yv - ymin) / cell_size_m).astype("int32"),
            "centroid_x": xv + cell_size_m / 2,
            "centroid_y": yv + cell_size_m / 2,
            "area_m2":   cell_size_m * cell_size_m,
            "geometry":  polys,
        },
        crs=crs,
    )
    return gdf


# ─────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────
def validate_within_bbox(
    gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    tolerance_pct: float = 0.05,
) -> None:
    """
    Raise if more than `tolerance_pct` of geometries fall outside the bbox.
    Use after every ingestion step — catches misaligned CRS declarations.
    """
    minx, miny, maxx, maxy = bbox
    bounds = gdf.geometry.bounds
    outside = (
        (bounds["minx"] < minx) | (bounds["miny"] < miny) |
        (bounds["maxx"] > maxx) | (bounds["maxy"] > maxy)
    )
    pct = outside.mean()
    if pct > tolerance_pct:
        raise ValueError(
            f"{pct:.1%} of geometries fall outside expected bbox {bbox}. "
            f"Likely CRS mismatch. First offending row index: "
            f"{outside.idxmax()}"
        )
