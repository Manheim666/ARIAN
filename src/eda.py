from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_eda_figures(df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return []

    out_paths: list[Path] = []

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.floor("D")

    if "City" in d.columns and "Temperature_C" in d.columns:
        sample_cities = list(d["City"].dropna().unique())[:6]
        fig, ax = plt.subplots(figsize=(12, 5))
        for c in sample_cities:
            sub = d[d["City"] == c].sort_values("Date")
            ax.plot(sub["Date"], sub["Temperature_C"], label=c, alpha=0.8)
        ax.set_title("Temperature time series (sample cities)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature_C")
        ax.legend(loc="best", ncols=3, fontsize=8)
        p = figures_dir / "temperature_timeseries_sample.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)

        if "Fire_Occurred" in d.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            sub = d.dropna(subset=["Temperature_C", "Fire_Occurred"]).copy()
            sub["Fire_Occurred"] = sub["Fire_Occurred"].astype(int)
            sns.boxplot(data=sub, x="Fire_Occurred", y="Temperature_C", ax=ax)
            ax.set_title("Temperature on fire days vs non-fire days (all cities)")
            ax.set_xlabel("Fire_Occurred")
            p = figures_dir / "temp_fireday_boxplot.png"
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out_paths.append(p)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(d["Temperature_C"].dropna().astype(float), kde=True, ax=ax)
        ax.set_title("Temperature distribution")
        p = figures_dir / "temperature_distribution.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)

    if "City" in d.columns and "Precipitation_mm" in d.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        city_means = d.groupby("City")["Precipitation_mm"].mean(numeric_only=True).sort_values(ascending=False)
        city_means.plot(kind="bar", ax=ax)
        ax.set_title("Mean precipitation by city")
        ax.set_ylabel("Precipitation_mm")
        p = figures_dir / "mean_precip_by_city.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)

    if "Temperature_C" in d.columns and ("fire_count" in d.columns or "Fire_Occurred" in d.columns):
        ycol = "fire_count" if "fire_count" in d.columns else "Fire_Occurred"
        fig, ax = plt.subplots(figsize=(6, 5))
        sub = d.dropna(subset=["Temperature_C", ycol]).copy()
        sub = sub.sample(min(len(sub), 5000), random_state=7)
        sns.scatterplot(data=sub, x="Temperature_C", y=ycol, alpha=0.3, ax=ax)
        ax.set_title(f"Temperature vs {ycol} (sample)")
        p = figures_dir / f"temp_vs_{ycol}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)

    return out_paths
