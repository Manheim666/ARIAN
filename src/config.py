from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    history_start: str = "2012-01-20"
    forecast_horizon_days: int = 30
    fire_buffer_km: float = 20.0
    polite_delay_s: float = 1.0

    @property
    def history_end(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


CITIES: dict[str, tuple[float, float]] = {
    "Baku": (40.409, 49.867),
    "Sumqayit": (40.59, 49.669),
    "Ganja": (40.6828, 46.3606),
    "Mingachevir": (40.764, 47.06),
    "Shirvan": (39.932, 48.93),
    "Lankaran": (38.752, 48.848),
    "Shaki": (41.198, 47.169),
    "Nakhchivan": (39.209, 45.412),
    "Yevlakh": (40.618, 47.15),
    "Quba": (41.361, 48.526),
    "Khachmaz": (41.464, 48.806),
    "Gabala": (40.998, 47.847),
    "Shamakhi": (40.63, 48.641),
    "Jalilabad": (39.209, 48.299),
    "Zaqatala": (41.63, 46.643),
    "Barda": (40.3744, 47.1266),
}


def detect_project_root() -> Path:
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "data").is_dir() and (cand / "notebooks").is_dir():
            return cand
    return here


def project_paths(root: Path) -> dict[str, Path]:
    data = root / "data"
    raw = data / "raw"
    processed = data / "processed"
    reference = data / "reference"
    outputs = root / "outputs"
    models = root / "models"
    reports = root / "reports"

    firms = raw / "firms"

    for p in (raw, processed, reference, outputs, models, reports, firms):
        p.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "data": data,
        "raw": raw,
        "processed": processed,
        "reference": reference,
        "outputs": outputs,
        "models": models,
        "reports": reports,
        "firms": firms,
    }
