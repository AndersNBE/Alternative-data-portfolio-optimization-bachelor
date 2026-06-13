# Elastic Net Panel Forecasting Research Note

## Model

`elastic_net_panel` is a pooled country-month return forecasting model with
country fixed effects and regularized linear coefficients. The implementation
uses an expanding-window design: for each prediction month, the model is fit
only on panel rows with `date` strictly before the prediction date. The
challenge-round default feature set is intentionally compact:

- `gnc_lag_1` through `gnc_lag_3`
- `return_lag_1`, `return_lag_2`
- `rolling_vol_3`

The current-month `gnc` column is excluded by default to avoid look-ahead risk.
It can be enabled with `{"include_current_gnc": true}` only if the caller can
document that the GNC observation is available before the target return is
realized. Country fixed effects are encoded inside the sklearn pipeline, so the
encoder is fit on the training window only and unseen countries at prediction
time are handled safely.

The estimator defaults to a conservative Elastic Net with time-ordered
cross-validation inside the training window. Config can switch to Ridge or
Lasso. The raw model forecast is then shrunk toward a historical-mean forecast:

`forecast = historical_mean + weight * clipped(raw_model - historical_mean)`

The default `weight` ramps from zero after the minimum 36 training months to
0.25 after 120 training months, and the model deviation is clipped at 8
percentage points per month. If the training window is too small, the fit fails,
or predictions are non-finite, the model fully falls back to a historical mean
forecast using the country mean when at least 36 country observations exist and
otherwise the pooled global mean.

## Research Basis

1. Gu, Kelly, and Xiu (2020), "Empirical Asset Pricing via Machine Learning,"
   Review of Financial Studies, DOI:
   [10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009). The paper shows
   that machine-learning methods can improve risk-premium measurement when many
   return predictors are available, and identifies signals related to momentum,
   liquidity, and volatility as important. This supports using a pooled
   predictive model with return lags and volatility controls, while keeping the
   specification transparent for a small project panel.

2. De Mol, Giannone, and Reichlin (2008), "Forecasting using a large number of
   predictors: Is Bayesian shrinkage a valid alternative to principal
   components?", Journal of Econometrics, DOI:
   [10.1016/j.jeconom.2008.08.011](https://doi.org/10.1016/j.jeconom.2008.08.011).
   Their results motivate shrinkage in high-dimensional forecasting panels,
   where correlated predictors and limited time-series length make unrestricted
   OLS unstable. Elastic Net, Ridge, and Lasso are frequentist analogues that
   stabilize coefficient estimates and can handle many related lag features.

3. Welch and Goyal (2008), "A Comprehensive Look at The Empirical Performance
   of Equity Premium Prediction," Review of Financial Studies, DOI:
   [10.1093/rfs/hhm014](https://doi.org/10.1093/rfs/hhm014). They emphasize
   strict out-of-sample evaluation and show that many equity-premium predictors
   fail to beat a historical average in real-time tests. This motivates the
   expanding-window design, the 36-month minimum before model fitting, and the
   historical-mean fallback baseline.

4. Rapach, Strauss, and Zhou (2010), "Out-of-Sample Equity Premium Prediction:
   Combination Forecasts and Links to the Real Economy," Review of Financial
   Studies, DOI:
   [10.1093/rfs/hhp063](https://doi.org/10.1093/rfs/hhp063). They argue that
   model uncertainty and instability hurt individual predictive regressions, and
   that combining information can reduce forecast volatility. A regularized
   pooled panel model with explicit shrinkage toward a historical mean is a
   related compromise: it combines information across lagged signals and
   countries while dampening noisy deviations from the benchmark forecast.

## Why This Fits The Container GNC Panel

The country panel has more potential predictors than a single-country monthly
history can support cleanly. Pooling countries increases the effective sample
size, while country fixed effects absorb stable country-level return differences
that should not be forced into the GNC slope. The first benchmark round showed
that the unshrunk Elastic Net was best by RMSE only marginally and had negative
return-forecast correlation. The challenge version therefore treats the
historical mean as a strong prior, uses fewer highly overlapping lag features,
and only allows a small regularized signal to move the forecast away from that
benchmark. Elastic Net remains useful because it shrinks related GNC and market
lag predictors jointly; Ridge and Lasso remain available as sensitivity checks.

## Assumptions

- All default features are known before the target return for the prediction
  month is realized.
- The monthly `date` column correctly orders information availability.
- The pooled slope assumption is acceptable: countries may have different
  intercepts, but the default model shares feature slopes across countries.
- Missing predictor values are missing because of short histories or data gaps,
  and median imputation from the training window is a reasonable neutral fill.
- A historical-mean prior is appropriate when predictor instability is high; the
  regularized signal should be treated as an incremental tilt, not a standalone
  return estimate.

## Failure Modes

- If `gnc_lag_*` was constructed after observing the same-month return, the model
  can still leak information despite the expanding-window split.
- Forecasts can be weak if GNC has no stable relation to future country index
  returns, or if the signal changes after geopolitical or market-regime breaks.
- One-hot fixed effects do not estimate reliable country intercepts for countries
  with very short histories; those cases rely heavily on the pooled relationship
  and historical-mean fallback.
- The shrinkage weight may be too conservative if GNC has a genuinely strong
  regime-specific relation to future returns, especially for directional or
  strategy metrics.
- Cross-validation inside small early windows can select overly aggressive or
  overly conservative regularization. The `min_train_months`, `alpha_grid`,
  `forecast_signal_weight`, and `use_cv` config values should be checked in
  robustness runs.
- Linear Elastic Net will miss nonlinear interactions. If sample size grows,
  tree-based or interaction-augmented models can be compared against this model.
