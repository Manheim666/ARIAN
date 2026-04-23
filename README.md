# Azerbaijan Wildfire Detection — ML Pipeline README
# Team name: ARIAN

## Project Overview

Binary classification pipeline that predicts daily wildfire occurrence across 15 Azerbaijani cities, using NASA satellite fire data fused with Open-Meteo weather records. The pipeline ends with a 7-day risk forecast dashboard.

---

## 1. Dataset History Length

**~2 year (2024 to present)**

- Weather data: fetched from **March 31, 2024 → today** via Open-Meteo Archive API
- Fire event data: NASA FIRMS VIIRS-SNPP **2024** (Azerbaijan, vegetation fires only — `type == 0`)

---

## 2. Dataset Granularity

**Daily**

Each row = one (city, date) pair with daily weather aggregates. Hourly data is also fetched but used only for 7-day forecast feature preparation — not model training.

**Coverage:** 15 cities across Azerbaijan:
Baku, Ganja, Lankaran, Guba, Zaqatala, Nakhchivan, Sheki, Shirvan, Mingachevir, Khachmaz, Goychay, Shamkir, Sabirabad, Imishli, Shamakhi

---

## 3. Initial Version of Features

### Numerical Features (18)

| Feature | Description |
|---|---|
| `temperature_2m_max` | Daily max temperature (°C) |
| `temperature_2m_mean` | Daily mean temperature (°C) |
| `temperature_2m_min` | Daily min temperature (°C) |
| `apparent_temperature_mean` | Feels-like mean temperature |
| `wind_speed_10m_max` | Max wind speed at 10m (km/h) |
| `wind_gusts_10m_max` | Max wind gusts at 10m (km/h) |
| `precipitation_sum` | Total daily precipitation (mm) |
| `temp_max_lag1` | Yesterday's max temperature |
| `temp_mean_lag1` | Yesterday's mean temperature |
| `precip_lag1` | Yesterday's precipitation |
| `wind_lag1` | Yesterday's max wind speed |
| `temp_roll3` | 3-day rolling avg max temperature |
| `precip_roll3` | 3-day rolling avg precipitation |
| `wind_roll3` | 3-day rolling avg wind speed |
| `temp_roll7` | 7-day rolling avg max temperature |
| `precip_roll7` | 7-day rolling avg precipitation |
| `dry_streak` | Consecutive days with precipitation ≤ 1 mm |
| `month` | Calendar month (1–12) |

### Categorical Features (2)

| Feature | Values |
|---|---|
| `location` | 15 Azerbaijani cities |
| `season` | spring / summer / autumn / winter |

---

## 4. Target Variable

**`fire_occurred`** — Binary (0 = no fire, 1 = fire occurred)

Derived by spatial join (0.5° buffer) between daily weather rows and NASA FIRMS vegetation fire records. The dataset is heavily imbalanced; **SMOTE** is applied to the training set only to address this.

---

## 5. Prediction Horizon

**< 1 month — 7-day ahead forecast**

The trained model is applied to a 7-day Open-Meteo forecast to produce a daily fire risk probability and categorical risk level (Very Low / Low / Medium / High / Very High) for each of the 15 cities.

---

## Pipeline Structure

| Section | Description |
|---|---|
| 0 | Imports & logging setup |
| 1 | Data collection — Open-Meteo weather + NASA FIRMS fire data |
| 2 | Spatial join — assign fires to nearest city |
| 3 | Feature engineering — lag, rolling, dry streak, season |
| 4 | EDA — class imbalance, seasonal patterns, correlations |
| 5 | Outlier detection — IsolationForest, LOF, OneClassSVM |
| 6 | Preprocessing — temporal split (70/15/15), scaling, SMOTE, PCA |
| 7 | Model training — Logistic Regression, Decision Tree, Random Forest, XGBoost |
| 8 | Hyperparameter tuning — GridSearchCV / RandomizedSearchCV |
| 9 | Final test-set evaluation — F1, ROC-AUC, confusion matrices |
| 10 | 7-day risk forecast dashboard — heatmap + peak risk bar chart |

---

## Evaluation Metric

**Primary: F1-score** (recall weighted over precision — a missed fire is worse than a false alarm)

Secondary: ROC-AUC

---

## Dependencies

```
pandas numpy requests requests-cache retry-requests openmeteo-requests
geopandas folium plotly shapely fiona
matplotlib seaborn scipy
scikit-learn imbalanced-learn xgboost
```

Install:
```bash
pip install pandas numpy requests requests-cache retry-requests openmeteo-requests geopandas folium plotly shapely fiona matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost
```

---

## Data Sources

- **Drive link:** https://drive.google.com/drive/folders/1k8BDFHhVXrcpNhFYbuBb2A75xsNAu6UF?usp=drive_link
