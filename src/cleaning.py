from __future__ import annotations

import pandas as pd


def clean_daily_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Production-grade cleaning for daily city-level weather.

    Returns cleaned df and a decisions dict suitable for reporting.
    """

    decisions: dict = {"missing": {}, "outliers": {}, "consistency": {}}

    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"]).dt.floor("D")

    numeric_cols = [c for c in out.columns if out[c].dtype.kind in "if" and c not in {"Latitude", "Longitude"}]

    # Consistency checks / physical clipping
    if "Precipitation_mm" in out.columns:
        neg = (out["Precipitation_mm"] < 0).sum()
        out["Precipitation_mm"] = out["Precipitation_mm"].clip(lower=0)
        decisions["consistency"]["precipitation_nonnegative_clipped"] = int(neg)

    if "Temperature_C" in out.columns:
        bad = ((out["Temperature_C"] < -50) | (out["Temperature_C"] > 60)).sum()
        out.loc[(out["Temperature_C"] < -50) | (out["Temperature_C"] > 60), "Temperature_C"] = pd.NA
        decisions["consistency"]["temperature_out_of_range_set_null"] = int(bad)

    # Missing values: time-aware interpolation within city (limited), then forward-fill small gaps
    if "City" in out.columns and "Date" in out.columns:
        out = out.sort_values(["City", "Date"])
        for col in numeric_cols:
            before = int(out[col].isna().sum())
            out[col] = out.groupby("City")[col].transform(
                lambda s: s.interpolate(limit=3, limit_direction="both")
            )
            out[col] = out.groupby("City")[col].transform(
                lambda s: s.ffill(limit=2).bfill(limit=2)
            )
            after = int(out[col].isna().sum())
            decisions["missing"][col] = {"nulls_before": before, "nulls_after": after}

    # Outliers: IQR flagging (do not drop; winsorize extreme tails per city)
    if "City" in out.columns:
        for col in [c for c in numeric_cols if c in out.columns]:
            def _winsorize_city(s: pd.Series) -> pd.Series:
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                if pd.isna(iqr) or iqr == 0:
                    return s
                lo = q1 - 1.5 * iqr
                hi = q3 + 1.5 * iqr
                return s.clip(lo, hi)

            before = out[col].copy()
            out[col] = out.groupby("City")[col].transform(_winsorize_city)
            changed = int((before != out[col]).sum(skipna=True))
            decisions["outliers"][col] = {"winsorized_values": changed}

    return out, decisions
