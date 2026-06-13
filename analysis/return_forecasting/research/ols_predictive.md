# OLS Predictive Regression Research Note

## Model choice

This model uses a conservative expanding-window out-of-sample predictive
regression for each country. For a country-month prediction at date `t`, the
estimator only sees rows with `date < t`. The implemented forecast is a robust
combination of univariate OLS forecasts from lagged GNC predictors, shrunk toward
the country's historical mean return. If the training window is too short, a
predictor has too few valid observations, or the regression is ill-conditioned,
the model falls back to the country's historical mean return computed from prior
rows only.

The default feature set was tightened after the first benchmark round:

- `gnc_lag_1` through `gnc_lag_3`

The initial version averaged `gnc_lag_1..6` and `return_lag_1..6`. On the
2020-01 to 2026-04 benchmark it underperformed the historical mean on RMSE, MAE,
correlation, direction, and strategy Sharpe. That result suggests the OLS signal
was too noisy and too active for the monthly sample, so the revised default keeps
the model closer to its intended container-GNC research question and avoids
adding return-lag forecasts that can dominate a small univariate combination.

The implementation rejects non-lagged predictors by default to avoid leakage.
Forecasts use a median combination by default, are shrunk 65% toward the
historical mean (`shrinkage_weight`, default 0.35 on the OLS forecast deviation),
and are clipped to a configurable band around the historical mean
(`forecast_bound_sigma`, default 1.5). These are pragmatic
Campbell-Thompson-style restrictions against unstable OOS forecasts from short
samples.

## Research basis

1. Yu, Hao, Wu, Zhao and Wang (2023), "Eye in outer space: satellite imageries of
   container ports can predict world stock returns", Humanities and Social
   Sciences Communications. DOI: https://doi.org/10.1057/s41599-023-01891-9

   This is the direct methodological anchor for using container-derived port
   information as an alternative real-time economic activity signal. The paper
   estimates predictive regressions with lagged container information and uses
   forecast combinations over univariate regressions. Our implementation adapts
   that idea from daily port-level signals to monthly country-level GNC signals,
   while using stronger shrinkage because the benchmark is monthly and has fewer
   observations.

2. Welch and Goyal (2008), "A Comprehensive Look at The Empirical Performance of
   Equity Premium Prediction", Review of Financial Studies. DOI:
   https://doi.org/10.1093/rfs/hhm014

   Welch and Goyal show that many return predictors look better in-sample than
   they do in real-time out-of-sample evaluation. This motivates the strict
   expanding-window setup and the historical mean fallback benchmark. The model
   is designed so each forecast could have been formed with information available
   before the prediction date.

3. Campbell and Thompson (2008), "Predicting Excess Stock Returns Out of Sample:
   Can Anything Beat the Historical Average?", Review of Financial Studies. DOI:
   https://doi.org/10.1093/rfs/hhm055

   Campbell and Thompson argue that weak economically motivated restrictions can
   improve unstable predictive regressions. Because the project forecasts raw
   country index returns rather than a clean excess market premium, the default
   implementation does not impose a nonnegative forecast restriction. It does,
   however, shrink forecasts toward the historical average and use a configurable
   historical-mean-plus/minus-volatility bound to avoid implausibly large
   forecasts from noisy small-sample OLS estimates.

4. Timmermann (2006), "Forecast Combinations", Handbook of Economic Forecasting.
   DOI: https://doi.org/10.1016/S1574-0706(05)01004-9

   Forecast-combination evidence supports simple combinations when the best
   single predictor is uncertain and individual models are unstable. This fits
   the country GNC setting, where different lags may matter across markets and
   the available monthly sample can be short. The revised implementation uses a
   median rather than a mean combination to reduce the influence of one unstable
   lag regression.

## Round 1 benchmark response

Round 1 showed:

- `ols_predictive`: RMSE 0.080207, MAE 0.042518, corr -0.0313,
  directional 0.5201, Sharpe -0.0267
- `historical_mean`: RMSE 0.079781, MAE 0.042078, corr -0.0086,
  directional 0.5284, Sharpe -0.0382

The correct response is not to add more predictors. The OLS model already lost
to the historical mean on error metrics and did not produce useful correlation.
The revised specification therefore moves closer to the historical mean unless
the GNC lags provide a stable signal in the expanding training window.

## Assumptions

- Lagged country-level GNC captures information about near-term economic
  activity that is not already fully reflected in market prices.
- Monthly country index returns can be modeled with a stable local linear
  relationship over the expanding training window.
- Simple robust forecast combination is preferable to estimating many
  combination weights in a short monthly sample.
- `target_return` is aligned so row `t` is the return to be predicted from
  predictors known before or at the start of that return period.

## Failure modes

- If `gnc_lag_*` columns were constructed after observing the target return
  period, the panel itself can still leak despite the model using only lagged
  column names.
- OLS coefficients can be unstable under structural breaks, changing trade
  patterns, crises, or short country histories.
- Forecast combination can dilute a genuinely useful single lag if other lags
  are noisy; the median combination reduces but does not eliminate this risk.
- The historical-mean fallback, shrinkage, and clipping make the model robust,
  but they can underreact when the true expected return shifts quickly.
- Raw index returns mix local equity moves, currency effects if applicable,
  index composition changes, and country-specific shocks not captured by GNC.
