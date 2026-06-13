# Cleaned Return Forecast h=3

- Target: `cumulative_forward_log_return_month_t_to_t_plus_h_minus_1`
- Rows after feature filter: 11,574
- Eval rows: 8,182
- Countries: 19
- Date range: 2017-02-01 to 2026-02-01

## Metrics

| forecast_horizon_months | model_id | n_predictions | n_countries | date_min | date_max | rmse | mae | bias | corr | directional_accuracy | active_directional_accuracy | predicted_positive_rate | actual_positive_rate | mean_actual_return | mean_predicted_return | strategy_mean_monthly_return | strategy_monthly_vol | strategy_annualized_sharpe | rmse_improvement_vs_historical_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | random_forest_panel | 8182 | 19 | 2020-01-01 | 2026-02-01 | 0.097132 | 0.073385 | -0.013730 | 0.298572 | 0.632975 | 0.632975 | 0.728306 | 0.610120 | 0.023797 | 0.010067 | 0.032119 | 0.098030 | 1.135005 | 0.047404 |
| 3 | distributed_lag | 8182 | 19 | 2020-01-01 | 2026-02-01 | 0.097900 | 0.074362 | -0.015967 | 0.294151 | 0.633830 | 0.633830 | 0.679052 | 0.610120 | 0.023797 | 0.007830 | 0.030446 | 0.098563 | 1.070054 | 0.039865 |
| 3 | elastic_net_panel | 8182 | 19 | 2020-01-01 | 2026-02-01 | 0.100656 | 0.076491 | -0.016169 | 0.143209 | 0.601076 | 0.601076 | 0.674896 | 0.610120 | 0.023797 | 0.007627 | 0.021738 | 0.100842 | 0.746728 | 0.012840 |
| 3 | historical_mean | 8182 | 19 | 2020-01-01 | 2026-02-01 | 0.101965 | 0.077674 | -0.016699 | 0.065231 | 0.560865 | 0.560865 | 0.660108 | 0.610120 | 0.023797 | 0.007098 | 0.011930 | 0.102466 | 0.403320 | 0.000000 |
| 3 | ols_predictive | 8182 | 19 | 2020-01-01 | 2026-02-01 | 0.101981 | 0.077688 | -0.016726 | 0.064797 | 0.560254 | 0.560254 | 0.658763 | 0.610120 | 0.023797 | 0.007071 | 0.012078 | 0.102449 | 0.408401 | -0.000159 |
| 3 | always_positive | 8182 | 19 | 2020-01-01 | 2026-02-01 | 0.981350 | 0.976203 | 0.976203 | nan | 0.610120 | 0.610120 | 1.000000 | 0.610120 | 0.023797 | 1.000000 | 0.023797 | 0.100376 | 0.821257 | -8.624351 |