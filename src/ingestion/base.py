"""
src/ingestion/base.py
─────────────────────
Abstract base for every data source. Contracts:

    1. `discover()`  → list of partition keys this source can yield
    2. `ingest(partition_key)` → produces a (Geo)DataFrame + writes bronze parquet
    3. Writes a manifest row on success (idempotency + audit)
    4. Is safe to re-run: already-ingested partitions are skipped unless
       `force=True`.

No ingestor performs feature engineering — that is the silver layer's job.
Each ingestor is *only* responsible for: fetch → normalize schema → project
into `crs_working` → write bronze.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.config.settings import Settings, get_settings
from src.utils.io import already_ingested, record_manifest

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Uniform return shape so callers can build summary tables."""
    source: str
    partition_key: str
    rows: int
    path: Path
    skipped: bool = False


class BaseIngestor(ABC):
    """
    Subclass contract:
        - `source_name`: short snake_case id used for paths + manifests
        - `discover()`: enumerate partitions (e.g. list of dates, years, files)
        - `_ingest_one(partition_key)`: return a DataFrame (or GeoDataFrame)
    """

    source_name: str = ""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.source_name:
            raise ValueError(f"{type(self).__name__} must set class attr `source_name`")

    # ── public API ──────────────────────────────────────────────────────
    @abstractmethod
    def discover(self) -> list[str]:
        """Return list of partition keys (e.g. ['2020', '2021', ...])."""

    @abstractmethod
    def _ingest_one(self, partition_key: str) -> pd.DataFrame | gpd.GeoDataFrame:
        """Fetch + normalize one partition. Must set CRS if geo."""

    def run(self, force: bool = False) -> list[IngestResult]:
        """Run the full source. Idempotent unless force=True."""
        partitions = self.discover()
        log.info("%s: %d partitions discovered", self.source_name, len(partitions))

        results: list[IngestResult] = []
        for pk in partitions:
            if not force and already_ingested(
                self.settings.abs(self.settings.manifest_dir), self.source_name, pk
            ):
                log.info("%s [%s] already ingested — skipping", self.source_name, pk)
                results.append(IngestResult(
                    source=self.source_name, partition_key=pk, rows=0,
                    path=self._bronze_path(pk), skipped=True,
                ))
                continue

            try:
                df = self._ingest_one(pk)
            except Exception as exc:  # noqa: BLE001
                # fail loud on a single partition but DON'T abort the whole flow
                log.exception("%s [%s] FAILED: %s", self.source_name, pk, exc)
                continue

            out = self._bronze_path(pk)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._write(df, out)

            record_manifest(
                self.settings.abs(self.settings.manifest_dir),
                self.source_name,
                {
                    "partition_key": pk,
                    "rows": len(df),
                    "path": str(out),
                    "is_geo": isinstance(df, gpd.GeoDataFrame),
                },
            )
            log.info("%s [%s]: %d rows → %s", self.source_name, pk, len(df), out)
            results.append(IngestResult(
                source=self.source_name, partition_key=pk,
                rows=len(df), path=out,
            ))
        return results

    # ── helpers overridable by subclasses ──────────────────────────────
    def _bronze_path(self, partition_key: str) -> Path:
        return self.settings.abs(self.settings.bronze_dir) / self.source_name / f"{partition_key}.parquet"

    def _write(self, df: pd.DataFrame | gpd.GeoDataFrame, path: Path) -> None:
        if isinstance(df, gpd.GeoDataFrame):
            df.to_parquet(path, compression="zstd", index=False)
        else:
            df.to_parquet(path, compression="zstd", index=False)
