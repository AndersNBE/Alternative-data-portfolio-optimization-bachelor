# Cleaned Return Forecast h=4

- Target: `cumulative_forward_log_return_month_t_to_t_plus_h_minus_1`
- Rows after feature filter: 11,488
- Eval rows: 8,096
- Countries: 19
- Date range: 2017-02-01 to 2026-01-01

## Metrics

| forecast_horizon_months | model_id | n_predictions | n_countries | date_min | date_max | rmse | mae | bias | corr | directional_accuracy | active_directional_accuracy | predicted_positive_rate | actual_positive_rate | mean_actual_return | mean_predicted_return | strategy_mean_monthly_return | strategy_monthly_vol | strategy_annualized_sharpe | rmse_improvement_vs_historical_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | random_forest_panel | 8096 | 19 | 2020-01-01 | 2026-01-01 | 0.106090 | 0.081946 | -0.021889 | 0.321665 | 0.675025 | 0.675025 | 0.743454 | 0.653903 | 0.035298 | 0.013409 | 0.040014 | 0.107504 | 1.289360 | 0.055633 |
| 4 | distributed_lag | 8096 | 19 | 2020-01-01 | 2026-01-01 | 0.108839 | 0.084638 | -0.025600 | 0.251470 | 0.637969 | 0.637969 | 0.665143 | 0.653903 | 0.035298 | 0.009697 | 0.032353 | 0.110052 | 1.018385 | 0.031158 |
| 4 | elastic_net_panel | 8096 | 19 | 2020-01-01 | 2026-01-01 | 0.111116 | 0.086361 | -0.025716 | 0.144003 | 0.614377 | 0.614377 | 0.685524 | 0.653903 | 0.035298 | 0.009582 | 0.024462 | 0.112070 | 0.756135 | 0.010889 |
| 4 | historical_mean | 8096 | 19 | 2020-01-01 | 2026-01-01 | 0.112340 | 0.087449 | -0.026198 | 0.090512 | 0.585104 | 0.585104 | 0.664155 | 0.653903 | 0.035298 | 0.009099 | 0.018278 | 0.113243 | 0.559120 | 0.000000 |
| 4 | ols_predictive | 8096 | 19 | 2020-01-01 | 2026-01-01 | 0.112374 | 0.087479 | -0.026283 | 0.090095 | 0.584857 | 0.584857 | 0.661932 | 0.653903 | 0.035298 | 0.009015 | 0.018806 | 0.113157 | 0.575726 | -0.000310 |
| 4 | always_positive | 8096 | 19 | 2020-01-01 | 2026-01-01 | 0.970857 | 0.964702 | 0.964702 | nan | 0.653903 | 0.653903 | 1.000000 | 0.653903 | 0.035298 | 1.000000 | 0.035298 | 0.109143 | 1.120319 | -7.642160 |