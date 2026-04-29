# Data Quality Report

## Summary
- Null violations: 0
- Duplicate violations: 0
- Range violations: 0

## Issues
No issues detected.

## Cleaning decisions

### missing
- Temperature_C: {'nulls_before': 0, 'nulls_after': 0}
- Temp_Max_C: {'nulls_before': 0, 'nulls_after': 0}
- Temp_Min_C: {'nulls_before': 0, 'nulls_after': 0}
- Precipitation_mm: {'nulls_before': 0, 'nulls_after': 0}
- Wind_Speed_kmh: {'nulls_before': 0, 'nulls_after': 0}
- Wind_Max_kmh: {'nulls_before': 0, 'nulls_after': 0}
- fire_count: {'nulls_before': 0, 'nulls_after': 0}
- mean_brightness: {'nulls_before': 74363, 'nulls_after': 43717}
- max_frp: {'nulls_before': 74363, 'nulls_after': 43717}
- Fire_Occurred: {'nulls_before': 0, 'nulls_after': 0}

### outliers
- Temperature_C: {'winsorized_values': 0}
- Temp_Max_C: {'winsorized_values': 0}
- Temp_Min_C: {'winsorized_values': 5}
- Precipitation_mm: {'winsorized_values': 0}
- Wind_Speed_kmh: {'winsorized_values': 2605}
- Wind_Max_kmh: {'winsorized_values': 1950}
- fire_count: {'winsorized_values': 444}
- mean_brightness: {'winsorized_values': 46025}
- max_frp: {'winsorized_values': 47322}
- Fire_Occurred: {'winsorized_values': 0}

### consistency
- precipitation_nonnegative_clipped: 0
- temperature_out_of_range_set_null: 0

## Trust evaluation — Can we trust this data?
This pipeline is suitable for analytics and modeling, but the data has known limitations. Key risks and mitigations are documented below.

### Weather (Open-Meteo / ERA5-family reanalysis)
- Reanalysis data is model-based, not a physical station sensor. It can smooth extremes and under-represent local microclimates.
- API-derived timestamps and aggregation can introduce boundary effects (UTC day boundaries vs local time).

### Missing data + interpolation bias
- Limited interpolation/forward-fill is applied for small gaps only. This can bias variance downward and may weaken extreme-event signals.
- Large gaps are intentionally not filled; downstream models should treat remaining missingness carefully.

### Seasonality + distribution shift
- Strong annual seasonality exists in temperature/rain/wind; naive train/test splits can leak seasonality.
- Climate trends imply non-stationarity; statistical tests assume stable measurement process over time, which may not fully hold.

### Wildfire detections (NASA FIRMS)
- FIRMS records hotspots detected by satellites; it is not a full burned-area ground truth.
- Detection probability varies with overpass timing, cloud cover, sensor characteristics, and fire size/intensity.
- City-level labeling uses a fixed buffer around centroids; fires near boundaries may be misattributed.

### Bottom line
- For *trend analysis and relative risk scoring*, the data is reasonably trustworthy with the above caveats.
- For *absolute fire incidence forecasting*, FIRMS detection bias and labeling geometry are the biggest limitations.
