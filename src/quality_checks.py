from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class QualitySummary:
    null_violations: int
    duplicate_violations: int
    range_violations: int


def run_quality_checks_daily(df: pd.DataFrame) -> tuple[QualitySummary, list[str]]:
    issues: list[str] = []

    null_viol = 0
    duplicate_viol = 0
    range_viol = 0

    # Null thresholds
    critical_cols = [c for c in ["City", "Date", "Temperature_C", "Precipitation_mm"] if c in df.columns]
    for c in critical_cols:
        frac = float(df[c].isna().mean())
        if frac > 0.01:
            null_viol += 1
            issues.append(f"Null fraction too high for `{c}`: {frac:.2%} (threshold 1%).")

    # Duplicates
    if "City" in df.columns and "Date" in df.columns:
        dupes = int(df.duplicated(subset=["City", "Date"]).sum())
        if dupes > 0:
            duplicate_viol += 1
            issues.append(f"Found {dupes} duplicate (City, Date) rows.")

    # Ranges
    if "Temperature_C" in df.columns:
        bad = int(((df["Temperature_C"] < -50) | (df["Temperature_C"] > 60)).sum())
        if bad:
            range_viol += 1
            issues.append(f"Temperature out of expected range [-50, 60] C: {bad} rows.")

    if "Precipitation_mm" in df.columns:
        bad = int((df["Precipitation_mm"] < 0).sum())
        if bad:
            range_viol += 1
            issues.append(f"Negative precipitation values: {bad} rows.")

    return QualitySummary(null_viol, duplicate_viol, range_viol), issues


def write_data_quality_report(path: Path, summary: QualitySummary, issues: list[str], decisions: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Data Quality Report\n")
    lines.append("\n## Summary\n")
    lines.append(f"- Null violations: {summary.null_violations}\n")
    lines.append(f"- Duplicate violations: {summary.duplicate_violations}\n")
    lines.append(f"- Range violations: {summary.range_violations}\n")

    lines.append("\n## Issues\n")
    if not issues:
        lines.append("No issues detected.\n")
    else:
        for i in issues:
            lines.append(f"- {i}\n")

    if decisions is not None:
        lines.append("\n## Cleaning decisions\n")
        for section, payload in decisions.items():
            lines.append(f"\n### {section}\n")
            if isinstance(payload, dict):
                for k, v in payload.items():
                    lines.append(f"- {k}: {v}\n")
            else:
                lines.append(f"- {payload}\n")

    lines.append("\n## Trust evaluation — Can we trust this data?\n")
    lines.append(
        "This pipeline is suitable for analytics and modeling, but the data has known limitations. "
        "Key risks and mitigations are documented below.\n"
    )
    lines.append("\n### Weather (Open-Meteo / ERA5-family reanalysis)\n")
    lines.append("- Reanalysis data is model-based, not a physical station sensor. It can smooth extremes and under-represent local microclimates.\n")
    lines.append("- API-derived timestamps and aggregation can introduce boundary effects (UTC day boundaries vs local time).\n")
    lines.append("\n### Missing data + interpolation bias\n")
    lines.append("- Limited interpolation/forward-fill is applied for small gaps only. This can bias variance downward and may weaken extreme-event signals.\n")
    lines.append("- Large gaps are intentionally not filled; downstream models should treat remaining missingness carefully.\n")
    lines.append("\n### Seasonality + distribution shift\n")
    lines.append("- Strong annual seasonality exists in temperature/rain/wind; naive train/test splits can leak seasonality.\n")
    lines.append("- Climate trends imply non-stationarity; statistical tests assume stable measurement process over time, which may not fully hold.\n")
    lines.append("\n### Wildfire detections (NASA FIRMS)\n")
    lines.append("- FIRMS records hotspots detected by satellites; it is not a full burned-area ground truth.\n")
    lines.append("- Detection probability varies with overpass timing, cloud cover, sensor characteristics, and fire size/intensity.\n")
    lines.append("- City-level labeling uses a fixed buffer around centroids; fires near boundaries may be misattributed.\n")
    lines.append("\n### Bottom line\n")
    lines.append("- For *trend analysis and relative risk scoring*, the data is reasonably trustworthy with the above caveats.\n")
    lines.append("- For *absolute fire incidence forecasting*, FIRMS detection bias and labeling geometry are the biggest limitations.\n")

    path.write_text("".join(lines), encoding="utf-8")
