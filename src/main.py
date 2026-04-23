"""
src/main.py
───────────
Phase-1 orchestration entry point. Not a production scheduler — the
Prefect flow is minimal on purpose so the pipeline can be run locally with
`python -m src.main` before we wire it to Prefect Cloud.

Run order (enforced):

    1. Build grid (GridBuilder)           ← spine for everything else
    2. Ingest FIRMS fires (FirmsIngestor)
    3. Ingest Open-Meteo history for every grid cell (async, batched)
    4. Ingest Open-Meteo seasonal forecast for 1-month horizon
    5. Write a top-level summary manifest

Other ingestors (NDVI, population, roads, landcover, lightning) follow the
same BaseIngestor pattern and are trivially added — they're deferred to
Phase-2's EDA once the grid + fires + weather spine is validated end-to-end.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config.settings import get_settings
from src.grid.builder import GridBuilder
from src.ingestion.firms import FirmsIngestor
from src.ingestion.openmeteo import OpenMeteoClient
from src.utils.io import write_parquet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("phase1")


# ─────────────────────────────────────────────────────────────────────────
async def run_openmeteo_history(grid: gpd.GeoDataFrame, start: str, end: str) -> pd.DataFrame:
    s = get_settings()
    locations = pd.DataFrame({
        "cell_id": grid["cell_id"].to_numpy(),
        "lat":     grid["lat"].to_numpy(),
        "lon":     grid["lon"].to_numpy(),
    })

    async with OpenMeteoClient(s) as om:
        # split the grid into chunks so progress is visible + checkpoint-safe
        CHUNK = 200
        frames: list[pd.DataFrame] = []
        for i in range(0, len(locations), CHUNK):
            chunk = locations.iloc[i:i + CHUNK]
            log.info("Open-Meteo history chunk %d–%d", i, i + len(chunk))
            df = await om.fetch_archive_daily_batch(
                chunk, start=start, end=end, concurrency=8,
            )
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


async def run_openmeteo_seasonal(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    s = get_settings()
    sample = grid.sample(min(len(grid), 500), random_state=42)  # seasonal is expensive — subsample
    locations = pd.DataFrame({
        "cell_id": sample["cell_id"].to_numpy(),
        "lat":     sample["lat"].to_numpy(),
        "lon":     sample["lon"].to_numpy(),
    })

    async with OpenMeteoClient(s) as om:
        frames: list[pd.DataFrame] = []
        sem = asyncio.Semaphore(4)

        async def _one(row):
            async with sem:
                df = await om.fetch_seasonal(row.lat, row.lon, months=1)
                df["cell_id"] = row.cell_id
                return df

        for row in locations.itertuples(index=False):
            try:
                frames.append(await _one(row))
            except Exception as exc:  # noqa: BLE001
                log.warning("seasonal fetch failed cell=%s: %s", row.cell_id, exc)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────
async def main(
    hist_start: str = "2020-01-01",
    hist_end: str | None = None,
) -> None:
    s = get_settings()
    hist_end = hist_end or str(date.today() - timedelta(days=6))   # ERA5 has ~5-day lag

    # 1. Grid ---------------------------------------------------------------
    gb = GridBuilder(s)
    grid = gb.build(mask_to_country=True)
    gb.save(grid)

    # 2. FIRMS -------------------------------------------------------------
    fi = FirmsIngestor(s)
    fi.run()

    # 3. Open-Meteo history -------------------------------------------------
    hist = await run_openmeteo_history(grid, hist_start, hist_end)
    if not hist.empty:
        out = s.abs(s.bronze_dir) / "openmeteo_history" / f"{hist_start}_{hist_end}.parquet"
        write_parquet(hist, out)
        log.info("Open-Meteo history: %d rows → %s", len(hist), out)

    # 4. Seasonal forecast --------------------------------------------------
    seas = await run_openmeteo_seasonal(grid)
    if not seas.empty:
        out = s.abs(s.bronze_dir) / "openmeteo_seasonal" / f"issued_{date.today()}.parquet"
        write_parquet(seas, out)
        log.info("Open-Meteo seasonal: %d rows → %s", len(seas), out)

    log.info("Phase-1 ingestion complete.")


if __name__ == "__main__":
    asyncio.run(main())
