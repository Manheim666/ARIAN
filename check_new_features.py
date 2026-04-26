
import pandas as pd
from src.weather.features import build_weather_features

# Load features
df = build_weather_features(save=False)

# Check the new features
new_features = ['temperature_range', 'heating_degree_days', 'cooling_degree_days']
print(f'DataFrame shape: {df.shape}')
print(f'New features present: {all(f in df.columns for f in new_features)}')

# Show correlation with original temperature features
temp_cols = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean']
all_temp_cols = temp_cols + new_features
correlation = df[all_temp_cols].corr()
print('\nCorrelation matrix:')
print(correlation.round(3))

# Check by city
print('\nAverage by city:')
print(df.groupby('City')[new_features].mean().round(2))
