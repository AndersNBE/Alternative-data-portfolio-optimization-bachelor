# Cleaned Return Forecast h=5

- Target: `cumulative_forward_log_return_month_t_to_t_plus_h_minus_1`
- Rows after feature filter: 11,371
- Eval rows: 7,979
- Countries: 19
- Date range: 2017-02-01 to 2025-12-01

## Metrics

| forecast_horizon_months | model_id | n_predictions | n_countries | date_min | date_max | rmse | mae | bias | corr | directional_accuracy | active_directional_accuracy | predicted_positive_rate | actual_positive_rate | mean_actual_return | mean_predicted_return | strategy_mean_monthly_return | strategy_monthly_vol | strategy_annualized_sharpe | rmse_improvement_vs_historical_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | random_forest_panel | 7979 | 19 | 2020-01-01 | 2025-12-01 | 0.116808 | 0.092249 | -0.028675 | 0.325582 | 0.665622 | 0.665622 | 0.746585 | 0.677904 | 0.044469 | 0.015794 | 0.045465 | 0.118854 | 1.325107 | 0.060489 |
| 5 | distributed_lag | 7979 | 19 | 2020-01-01 | 2025-12-01 | 0.121376 | 0.096036 | -0.033887 | 0.210851 | 0.625768 | 0.625768 | 0.656348 | 0.677904 | 0.044469 | 0.010582 | 0.034291 | 0.122545 | 0.969332 | 0.023749 |
| 5 | elastic_net_panel | 7979 | 19 | 2020-01-01 | 2025-12-01 | 0.123194 | 0.097470 | -0.034002 | 0.135993 | 0.610102 | 0.610102 | 0.654719 | 0.677904 | 0.044469 | 0.010467 | 0.028831 | 0.123944 | 0.805811 | 0.009128 |
| 5 | historical_mean | 7979 | 19 | 2020-01-01 | 2025-12-01 | 0.124328 | 0.098355 | -0.034382 | 0.096521 | 0.585412 | 0.585412 | 0.640306 | 0.677904 | 0.044469 | 0.010088 | 0.021957 | 0.125344 | 0.606811 | 0.000000 |
| 5 | ols_predictive | 7979 | 19 | 2020-01-01 | 2025-12-01 | 0.124383 | 0.098392 | -0.034499 | 0.095921 | 0.587417 | 0.587417 | 0.635543 | 0.677904 | 0.044469 | 0.009971 | 0.022175 | 0.125306 | 0.613042 | -0.000440 |
| 5 | always_positive | 7979 | 19 | 2020-01-01 | 2025-12-01 | 0.962941 | 0.955531 | 0.955531 | nan | 0.677904 | 0.677904 | 1.000000 | 0.677904 | 0.044469 | 1.000000 | 0.044469 | 0.119230 | 1.292014 | -6.745135 |