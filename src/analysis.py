from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HypothesisResult:
    name: str
    method: str
    assumptions: list[str]
    statistic: str
    p_value: float
    effect_size: float | None
    interpretation: str


def test_temperature_trend(df: pd.DataFrame) -> HypothesisResult:
    """(A) Linear regression: Temperature_C ~ time."""
    try:
        import statsmodels.api as sm
    except Exception as exc:
        raise RuntimeError("statsmodels is required for hypothesis testing") from exc

    d = df.dropna(subset=["Date", "Temperature_C"]).copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.floor("D")
    d["t"] = (d["Date"] - d["Date"].min()).dt.days.astype(float)

    y = d["Temperature_C"].astype(float)
    X = sm.add_constant(d["t"].astype(float))
    model = sm.OLS(y, X).fit()

    slope = float(model.params["t"])
    p = float(model.pvalues["t"])

    interp = (
        f"Slope={slope:.4f} C/day. "
        + ("Reject H0: warming trend detected." if p < 0.05 else "Fail to reject H0: no statistically significant trend.")
    )

    return HypothesisResult(
        name="Temperature trend over time",
        method="OLS linear regression (statsmodels OLS)",
        assumptions=[
            "Linear relationship between time and mean temperature",
            "Independent errors (approx.)",
            "Homoscedastic residuals (approx.)",
            "Residuals approximately normal for inference",
        ],
        statistic="slope_C_per_day",
        p_value=p,
        effect_size=slope,
        interpretation=interp,
    )


def test_recent_vs_past(df: pd.DataFrame) -> HypothesisResult:
    """(B) t-test: last 5 years vs first 5 years."""
    try:
        from scipy import stats
    except Exception as exc:
        raise RuntimeError("scipy is required for hypothesis testing") from exc

    d = df.dropna(subset=["Date", "Temperature_C"]).copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.floor("D")
    start = d["Date"].min()
    first_end = start + pd.DateOffset(years=5)
    last_start = d["Date"].max() - pd.DateOffset(years=5)

    first = d.loc[d["Date"] < first_end, "Temperature_C"].astype(float)
    last = d.loc[d["Date"] >= last_start, "Temperature_C"].astype(float)

    if len(first) < 30 or len(last) < 30:
        return HypothesisResult(
            name="Recent vs past mean temperature",
            method="Welch's t-test (scipy.stats.ttest_ind, unequal variances)",
            assumptions=[
                "Samples are independent (approx.)",
                "Temperatures are approximately normally distributed or sample sizes large enough for CLT",
                "Unequal variances allowed (Welch)",
            ],
            statistic="cohens_d",
            p_value=float("nan"),
            effect_size=None,
            interpretation="Insufficient data for 5y vs 5y comparison.",
        )

    tstat, p = stats.ttest_ind(last, first, equal_var=False, nan_policy="omit")
    diff = float(last.mean() - first.mean())

    # Cohen's d (using pooled std as an approximation)
    s1, s2 = float(first.std(ddof=1)), float(last.std(ddof=1))
    pooled = np.sqrt((s1**2 + s2**2) / 2) if (s1 > 0 and s2 > 0) else np.nan
    d_eff = float(diff / pooled) if pooled and not np.isnan(pooled) else None

    interp = (
        f"Mean(last5y)-Mean(first5y)={diff:.3f} C. "
        + ("Reject H0: means differ." if p < 0.05 else "Fail to reject H0: no significant mean difference.")
    )

    return HypothesisResult(
        name="Recent vs past mean temperature",
        method="Welch's t-test (scipy.stats.ttest_ind, unequal variances)",
        assumptions=[
            "Samples are independent (approx.)",
            "Temperatures are approximately normally distributed or sample sizes large enough for CLT",
            "Unequal variances allowed (Welch)",
        ],
        statistic="cohens_d",
        p_value=float(p),
        effect_size=d_eff,
        interpretation=interp,
    )


def test_temp_vs_wildfire(df: pd.DataFrame) -> HypothesisResult:
    """(C) Relationship: Temperature_C vs Fire_Occurred or fire_count."""
    try:
        from scipy import stats
    except Exception as exc:
        raise RuntimeError("scipy is required for hypothesis testing") from exc

    d = df.copy()
    if "Fire_Occurred" in d.columns:
        ycol = "Fire_Occurred"
    elif "fire_count" in d.columns:
        ycol = "fire_count"
    else:
        return HypothesisResult(
            name="Temperature vs wildfire activity",
            method="Pearson correlation (scipy.stats.pearsonr)",
            assumptions=[
                "Linear association (Pearson)",
                "No extreme outliers dominating correlation",
                "Paired observations are representative",
            ],
            statistic="pearson_r",
            p_value=float("nan"),
            effect_size=None,
            interpretation="No wildfire column found (expected Fire_Occurred or fire_count).",
        )

    d = d.dropna(subset=["Temperature_C", ycol])
    if len(d) < 100:
        return HypothesisResult(
            name="Temperature vs wildfire activity",
            method="Pearson correlation (scipy.stats.pearsonr)",
            assumptions=[
                "Linear association (Pearson)",
                "No extreme outliers dominating correlation",
                "Paired observations are representative",
            ],
            statistic="pearson_r",
            p_value=float("nan"),
            effect_size=None,
            interpretation="Insufficient paired observations.",
        )

    r, p = stats.pearsonr(d["Temperature_C"].astype(float), d[ycol].astype(float))
    interp = (
        f"Pearson r={r:.3f}. "
        + ("Reject H0: association detected." if p < 0.05 else "Fail to reject H0: no significant association.")
    )

    return HypothesisResult(
        name="Temperature vs wildfire activity",
        method="Pearson correlation (scipy.stats.pearsonr)",
        assumptions=[
            "Linear association (Pearson)",
            "No extreme outliers dominating correlation",
            "Paired observations are representative",
        ],
        statistic="pearson_r",
        p_value=float(p),
        effect_size=float(r),
        interpretation=interp,
    )
