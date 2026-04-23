"""
tests/test_smoke.py
───────────────────
Phase-1 smoke: imports resolve, config loads, no syntax/circular-import bugs.
Run:  pytest -xvs tests/test_smoke.py

Real data-integration tests live in tests/integration/ and require live
network + ≥200 MB of sample files — not run in CI by default.
"""
from __future__ import annotations


def test_settings_load():
    from src.config.settings import get_settings
    s = get_settings()
    assert s.grid_resolution_m == 5_000
    assert s.crs_working == "EPSG:32638"
    assert len(s.openmeteo_hourly_vars) >= 10
    assert "vapour_pressure_deficit" in s.openmeteo_hourly_vars


def test_imports():
    # just importing proves no circular imports or syntax errors
    from src.ingestion.openmeteo import OpenMeteoClient, compute_vpd, compute_kbdi_iter
    from src.ingestion.firms import FirmsIngestor
    from src.grid.builder import GridBuilder
    from src.utils.geo import build_regular_grid, reproject_bbox
    assert OpenMeteoClient is not None
    assert FirmsIngestor.source_name == "firms"


def test_vpd_math():
    """Spot-check VPD: at 25 °C / 50 % RH it should be ≈ 1.58 kPa."""
    import pandas as pd
    from src.ingestion.openmeteo import compute_vpd
    vpd = compute_vpd(pd.Series([25.0]), pd.Series([50.0]))
    assert abs(vpd.iloc[0] - 1.58) < 0.05


def test_bbox_reprojection():
    from src.utils.geo import reproject_bbox
    bbox_m = reproject_bbox((44.77, 38.39, 50.38, 41.91), "EPSG:4326", "EPSG:32638")
    xmin, ymin, xmax, ymax = bbox_m
    # width/height of AZ in meters — sanity bounds
    assert 400_000 < (xmax - xmin) < 700_000
    assert 300_000 < (ymax - ymin) < 500_000


def test_grid_generation_tiny():
    """Build a small toy grid to confirm geometry construction works."""
    from src.utils.geo import build_regular_grid
    gdf = build_regular_grid(
        bbox_projected=(0, 0, 20_000, 10_000),
        cell_size_m=5_000,
        crs="EPSG:32638",
    )
    assert len(gdf) == 4 * 2   # 4 × 2 cells
    assert (gdf["area_m2"] == 25_000_000).all()
    assert gdf["cell_id"].is_unique
