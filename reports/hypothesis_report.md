# Hypothesis Test Report

## Temperature trend over time
- Method: OLS linear regression (statsmodels OLS)
- Statistic: slope_C_per_day
- p-value: 0.0015072349059546608
- Effect size: 6.70418254526423e-05
- Assumptions:
  - Linear relationship between time and mean temperature
  - Independent errors (approx.)
  - Homoscedastic residuals (approx.)
  - Residuals approximately normal for inference
- Interpretation: Slope=0.0001 C/day. Reject H0: warming trend detected.

## Recent vs past mean temperature
- Method: Welch's t-test (scipy.stats.ttest_ind, unequal variances)
- Statistic: cohens_d
- p-value: 1.732373197697988e-28
- Effect size: 0.09162350869457654
- Assumptions:
  - Samples are independent (approx.)
  - Temperatures are approximately normally distributed or sample sizes large enough for CLT
  - Unequal variances allowed (Welch)
- Interpretation: Mean(last5y)-Mean(first5y)=0.847 C. Reject H0: means differ.

## Temperature vs wildfire activity
- Method: Pearson correlation (scipy.stats.pearsonr)
- Statistic: pearson_r
- p-value: 0.0
- Effect size: 0.19391834533286867
- Assumptions:
  - Linear association (Pearson)
  - No extreme outliers dominating correlation
  - Paired observations are representative
- Interpretation: Pearson r=0.194. Reject H0: association detected.

