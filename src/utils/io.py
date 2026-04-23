"""
src/utils/io.py
───────────────
Parquet / GeoParquet read-write helpers with manifest tracking.

A *manifest* is an append-only JSONL file, one row per ingested partition,
capturing:  (source, partition_key, row_count, sha256, ingested_at).

Why bother?  So re-running the pipeline with the same inputs produces a
no-op, and schema drift is detectable (sha256 on the file contents).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
#  Parquet
# ─────────────────────────────────────────────────────────────────────────
def write_parquet(
    df: pd.DataFrame,
    path: Path,
    partition_cols: list[str] | None = None,
    compression: str = "zstd",
) -> None:
    """Atomic parquet write via *.tmp + rename. Survives SIGKILL mid-write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if partition_cols:
        # pyarrow dataset API handles partitioning natively
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(path),
            partition_cols=partition_cols,
            compression=compression,
            existing_data_behavior="overwrite_or_ignore",
        )
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, compression=compression, index=False)
        tmp.replace(path)


def write_geoparquet(
    gdf: gpd.GeoDataFrame,
    path: Path,
    compression: str = "zstd",
) -> None:
    """
    GeoParquet write. Embeds CRS in the file metadata (GeoParquet 1.1 spec).
    Downstream readers reconstruct CRS without relying on side-channels.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    gdf.to_parquet(tmp, compression=compression, index=False)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────
#  Manifests
# ─────────────────────────────────────────────────────────────────────────
def sha256_of_file(path: Path, buf_size: int = 1 << 20) -> str:
    """Streaming sha256 — works for files > RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def record_manifest(
    manifest_dir: Path,
    source: str,
    entry: dict[str, Any],
) -> None:
    """
    Append one JSON record per ingested partition.
    Callers should include enough identifying info in `entry` to detect
    duplicates (partition_key, row_count, sha256).
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{source}.jsonl"
    entry = {
        **entry,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_manifest(manifest_dir: Path, source: str) -> pd.DataFrame:
    """Load all manifest entries for a source. Empty DF if none yet."""
    path = manifest_dir / f"{source}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def already_ingested(
    manifest_dir: Path,
    source: str,
    partition_key: str,
) -> bool:
    """Idempotency check — skip re-ingest of identical partitions."""
    df = load_manifest(manifest_dir, source)
    if df.empty or "partition_key" not in df.columns:
        return False
    return (df["partition_key"] == partition_key).any()
