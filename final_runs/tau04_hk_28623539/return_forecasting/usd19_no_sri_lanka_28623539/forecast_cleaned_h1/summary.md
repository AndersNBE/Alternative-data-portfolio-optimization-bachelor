# Cleaned Return Forecast h=1

- Target: `cumulative_forward_log_return_month_t_to_t_plus_h_minus_1`
- Rows after feature filter: 11,651
- Eval rows: 8,259
- Countries: 19
- Date range: 2017-02-01 to 2026-04-01

## Metrics

| forecast_horizon_months | model_id | n_predictions | n_countries | date_min | date_max | rmse | mae | bias | corr | directional_accuracy | active_directional_accuracy | predicted_positive_rate | actual_positive_rate | mean_actual_return | mean_predicted_return | strategy_mean_monthly_return | strategy_monthly_vol | strategy_annualized_sharpe | rmse_improvement_vs_historical_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | distributed_lag | 8259 | 19 | 2020-01-01 | 2026-04-01 | 0.051791 | 0.037780 | -0.003806 | 0.717063 | 0.828672 | 0.828672 | 0.653106 | 0.591476 | 0.007542 | 0.003737 | 0.033614 | 0.049734 | 2.341333 | 0.139663 |
| 1 | random_forest_panel | 8259 | 19 | 2020-01-01 | 2026-04-01 | 0.053190 | 0.037274 | -0.003372 | 0.605782 | 0.840295 | 0.840295 | 0.635186 | 0.591476 | 0.007542 | 0.004170 | 0.031565 | 0.051059 | 2.141497 | 0.116424 |
| 1 | elastic_net_panel | 8259 | 19 | 2020-01-01 | 2026-04-01 | 0.057628 | 0.042049 | -0.004280 | 0.351111 | 0.711830 | 0.711830 | 0.664608 | 0.591476 | 0.007542 | 0.003263 | 0.018752 | 0.057024 | 1.139125 | 0.042696 |
| 1 | ols_predictive | 8259 | 19 | 2020-01-01 | 2026-04-01 | 0.060198 | 0.044650 | -0.004860 | -0.033777 | 0.559148 | 0.559148 | 0.665698 | 0.591476 | 0.007542 | 0.002683 | 0.001993 | 0.059995 | 0.115054 | 0.000008 |
| 1 | historical_mean | 8259 | 19 | 2020-01-01 | 2026-04-01 | 0.060198 | 0.044652 | -0.004853 | -0.034340 | 0.560601 | 0.560601 | 0.666182 | 0.591476 | 0.007542 | 0.002689 | 0.002038 | 0.059993 | 0.117654 | 0.000000 |
| 1 | always_positive | 8259 | 19 | 2020-01-01 | 2026-04-01 | 0.994243 | 0.992458 | 0.992458 | nan | 0.591476 | 0.591476 | 1.000000 | 0.591476 | 0.007542 | 1.000000 | 0.007542 | 0.059552 | 0.438728 | -15.516123 |