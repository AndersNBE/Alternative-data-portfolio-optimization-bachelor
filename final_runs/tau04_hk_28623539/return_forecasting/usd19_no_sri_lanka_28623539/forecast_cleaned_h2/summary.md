# Cleaned Return Forecast h=2

- Target: `cumulative_forward_log_return_month_t_to_t_plus_h_minus_1`
- Rows after feature filter: 11,625
- Eval rows: 8,233
- Countries: 19
- Date range: 2017-02-01 to 2026-03-01

## Metrics

| forecast_horizon_months | model_id | n_predictions | n_countries | date_min | date_max | rmse | mae | bias | corr | directional_accuracy | active_directional_accuracy | predicted_positive_rate | actual_positive_rate | mean_actual_return | mean_predicted_return | strategy_mean_monthly_return | strategy_monthly_vol | strategy_annualized_sharpe | rmse_improvement_vs_historical_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | distributed_lag | 8233 | 19 | 2020-01-01 | 2026-03-01 | 0.079457 | 0.059310 | -0.009142 | 0.418936 | 0.685412 | 0.685412 | 0.681040 | 0.611320 | 0.015469 | 0.006327 | 0.031029 | 0.079530 | 1.351545 | 0.065064 |
| 2 | random_forest_panel | 8233 | 19 | 2020-01-01 | 2026-03-01 | 0.079678 | 0.059180 | -0.007767 | 0.369876 | 0.676910 | 0.676910 | 0.714320 | 0.611320 | 0.015469 | 0.007702 | 0.030493 | 0.079737 | 1.324721 | 0.062456 |
| 2 | elastic_net_panel | 8233 | 19 | 2020-01-01 | 2026-03-01 | 0.083236 | 0.062185 | -0.009483 | 0.179054 | 0.624195 | 0.624195 | 0.686384 | 0.611320 | 0.015469 | 0.005986 | 0.019256 | 0.083169 | 0.802032 | 0.020597 |
| 2 | historical_mean | 8233 | 19 | 2020-01-01 | 2026-03-01 | 0.084986 | 0.063859 | -0.010033 | 0.022118 | 0.562128 | 0.562128 | 0.679461 | 0.611320 | 0.015469 | 0.005436 | 0.007994 | 0.084994 | 0.325829 | 0.000000 |
| 2 | ols_predictive | 8233 | 19 | 2020-01-01 | 2026-03-01 | 0.084989 | 0.063856 | -0.010013 | 0.021785 | 0.562371 | 0.562371 | 0.678732 | 0.611320 | 0.015469 | 0.005456 | 0.008025 | 0.084991 | 0.327092 | -0.000037 |
| 2 | always_positive | 8233 | 19 | 2020-01-01 | 2026-03-01 | 0.988104 | 0.984531 | 0.984531 | nan | 0.611320 | 0.611320 | 1.000000 | 0.611320 | 0.015469 | 1.000000 | 0.015469 | 0.083956 | 0.638260 | -10.626628 |