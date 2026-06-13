# Cleaned Return Forecast Suite

- Cleaned segmentation model: `03_bce_dice_group_100ep_20260521_204006`
- Threshold: `0.4`
- Return currency: `USD`
- Excluded countries: `Sri Lanka`
- Target: `cumulative_forward_log_return_month_t_to_t_plus_h_minus_1`
- Test start: `2020-01-01`

## Forecast Metrics

| forecast_horizon_months | model_id | n_predictions | rmse | directional_accuracy | actual_positive_rate | predicted_positive_rate | strategy_annualized_sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | distributed_lag | 8259 | 0.051791 | 0.828672 | 0.591476 | 0.653106 | 2.341333 |
| 1 | random_forest_panel | 8259 | 0.053190 | 0.840295 | 0.591476 | 0.635186 | 2.141497 |
| 1 | elastic_net_panel | 8259 | 0.057628 | 0.711830 | 0.591476 | 0.664608 | 1.139125 |
| 1 | ols_predictive | 8259 | 0.060198 | 0.559148 | 0.591476 | 0.665698 | 0.115054 |
| 1 | historical_mean | 8259 | 0.060198 | 0.560601 | 0.591476 | 0.666182 | 0.117654 |
| 1 | always_positive | 8259 | 0.994243 | 0.591476 | 0.591476 | 1.000000 | 0.438728 |
| 2 | distributed_lag | 8233 | 0.079457 | 0.685412 | 0.611320 | 0.681040 | 1.351545 |
| 2 | random_forest_panel | 8233 | 0.079678 | 0.676910 | 0.611320 | 0.714320 | 1.324721 |
| 2 | elastic_net_panel | 8233 | 0.083236 | 0.624195 | 0.611320 | 0.686384 | 0.802032 |
| 2 | historical_mean | 8233 | 0.084986 | 0.562128 | 0.611320 | 0.679461 | 0.325829 |
| 2 | ols_predictive | 8233 | 0.084989 | 0.562371 | 0.611320 | 0.678732 | 0.327092 |
| 2 | always_positive | 8233 | 0.988104 | 0.611320 | 0.611320 | 1.000000 | 0.638260 |
| 3 | random_forest_panel | 8182 | 0.097132 | 0.632975 | 0.610120 | 0.728306 | 1.135005 |
| 3 | distributed_lag | 8182 | 0.097900 | 0.633830 | 0.610120 | 0.679052 | 1.070054 |
| 3 | elastic_net_panel | 8182 | 0.100656 | 0.601076 | 0.610120 | 0.674896 | 0.746728 |
| 3 | historical_mean | 8182 | 0.101965 | 0.560865 | 0.610120 | 0.660108 | 0.403320 |
| 3 | ols_predictive | 8182 | 0.101981 | 0.560254 | 0.610120 | 0.658763 | 0.408401 |
| 3 | always_positive | 8182 | 0.981350 | 0.610120 | 0.610120 | 1.000000 | 0.821257 |
| 4 | random_forest_panel | 8096 | 0.106090 | 0.675025 | 0.653903 | 0.743454 | 1.289360 |
| 4 | distributed_lag | 8096 | 0.108839 | 0.637969 | 0.653903 | 0.665143 | 1.018385 |
| 4 | elastic_net_panel | 8096 | 0.111116 | 0.614377 | 0.653903 | 0.685524 | 0.756135 |
| 4 | historical_mean | 8096 | 0.112340 | 0.585104 | 0.653903 | 0.664155 | 0.559120 |
| 4 | ols_predictive | 8096 | 0.112374 | 0.584857 | 0.653903 | 0.661932 | 0.575726 |
| 4 | always_positive | 8096 | 0.970857 | 0.653903 | 0.653903 | 1.000000 | 1.120319 |
| 5 | random_forest_panel | 7979 | 0.116808 | 0.665622 | 0.677904 | 0.746585 | 1.325107 |
| 5 | distributed_lag | 7979 | 0.121376 | 0.625768 | 0.677904 | 0.656348 | 0.969332 |
| 5 | elastic_net_panel | 7979 | 0.123194 | 0.610102 | 0.677904 | 0.654719 | 0.805811 |
| 5 | historical_mean | 7979 | 0.124328 | 0.585412 | 0.677904 | 0.640306 | 0.606811 |
| 5 | ols_predictive | 7979 | 0.124383 | 0.587417 | 0.677904 | 0.635543 | 0.613042 |
| 5 | always_positive | 7979 | 0.962941 | 0.677904 | 0.677904 | 1.000000 | 1.292014 |

## Best Forecasts

- h=1: best direction `random_forest_panel` (0.8403); best RMSE `distributed_lag` (0.051791)
- h=2: best direction `distributed_lag` (0.6854); best RMSE `distributed_lag` (0.079457)
- h=3: best direction `distributed_lag` (0.6338); best RMSE `random_forest_panel` (0.097132)
- h=4: best direction `random_forest_panel` (0.6750); best RMSE `random_forest_panel` (0.106090)
- h=5: best direction `always_positive` (0.6779); best RMSE `random_forest_panel` (0.116808)

## MAD Portfolio

| model_id | forecast_horizon_months | signal_variant | lookback_months | n_periods | mean_period_return | period_vol | mean_monthly_return | monthly_vol | annualized_sharpe | equal_weight_mean_period_return | equal_weight_period_vol | equal_weight_mean_monthly_return | equal_weight_monthly_vol | equal_weight_annualized_sharpe | mean_excess_vs_equal_weight | hit_rate_vs_equal_weight | cumulative_log_return | compound_return | max_drawdown | equal_weight_compound_return | equal_weight_max_drawdown | mean_turnover | optimal_solver_rate | mean_n_assets | mean_n_scenarios |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distributed_lag | 1 | raw | 5 | 71 | 0.013280 | 0.041022 | 0.013280 | 0.041022 | 1.121448 | 0.009119 | 0.042258 | 0.009119 | 0.042258 | 0.747525 | 0.004161 | 0.563380 | 0.942905 | 1.567429 | -0.194416 | 0.910660 | -0.235974 | 0.525705 | 1.000000 | 19.000000 | 4.000000 |
| elastic_net_panel | 5 | direction_risk_scaled | 7 | 65 | 0.052139 | 0.072335 | 0.010428 | 0.032349 | 1.116656 | 0.039592 | 0.084117 | 0.007918 | 0.037618 | 0.729178 | 0.012546 | 0.569231 | 0.677802 | 0.969545 | -0.117510 | 0.673136 | -0.174954 | 0.373371 | 1.000000 | 19.000000 | 6.000000 |
| distributed_lag | 5 | direction_risk_scaled | 7 | 65 | 0.051686 | 0.071925 | 0.010337 | 0.032166 | 1.113261 | 0.039592 | 0.084117 | 0.007918 | 0.037618 | 0.729178 | 0.012093 | 0.569231 | 0.671915 | 0.957983 | -0.116401 | 0.673136 | -0.174954 | 0.379153 | 1.000000 | 19.000000 | 6.000000 |
| historical_mean | 5 | direction_risk_scaled | 7 | 65 | 0.051818 | 0.072426 | 0.010364 | 0.032390 | 1.108387 | 0.039592 | 0.084117 | 0.007918 | 0.037618 | 0.729178 | 0.012226 | 0.569231 | 0.673633 | 0.961349 | -0.120594 | 0.673136 | -0.174954 | 0.378256 | 1.000000 | 19.000000 | 6.000000 |
| ols_predictive | 5 | direction_risk_scaled | 7 | 65 | 0.051965 | 0.072644 | 0.010393 | 0.032487 | 1.108189 | 0.039592 | 0.084117 | 0.007918 | 0.037618 | 0.729178 | 0.012372 | 0.569231 | 0.675542 | 0.965098 | -0.121095 | 0.673136 | -0.174954 | 0.370572 | 1.000000 | 19.000000 | 6.000000 |
| random_forest_panel | 5 | direction_risk_scaled | 7 | 65 | 0.050440 | 0.071258 | 0.010088 | 0.031867 | 1.096591 | 0.039592 | 0.084117 | 0.007918 | 0.037618 | 0.729178 | 0.010847 | 0.538462 | 0.655715 | 0.926519 | -0.115842 | 0.673136 | -0.174954 | 0.382175 | 1.000000 | 19.000000 | 6.000000 |
| distributed_lag | 5 | direction_risk_scaled | 8 | 64 | 0.049056 | 0.069498 | 0.009811 | 0.031081 | 1.093508 | 0.037385 | 0.082882 | 0.007477 | 0.037066 | 0.698779 | 0.011671 | 0.593750 | 0.627913 | 0.873696 | -0.108437 | 0.613690 | -0.174954 | 0.354588 | 1.000000 | 19.000000 | 7.000000 |
| ols_predictive | 5 | direction | 7 | 65 | 0.051495 | 0.074038 | 0.010299 | 0.033111 | 1.077495 | 0.039592 | 0.084117 | 0.007918 | 0.037618 | 0.729178 | 0.011902 | 0.584615 | 0.669432 | 0.953128 | -0.124965 | 0.673136 | -0.174954 | 0.369299 | 1.000000 | 19.000000 | 6.000000 |
| distributed_lag | 1 | raw | 6 | 70 | 0.011289 | 0.036315 | 0.011289 | 0.036315 | 1.076877 | 0.008511 | 0.042249 | 0.008511 | 0.042249 | 0.697806 | 0.002779 | 0.528571 | 0.790248 | 1.203944 | -0.155897 | 0.814383 | -0.235974 | 0.509958 | 1.000000 | 19.000000 | 5.000000 |
| historical_mean | 5 | direction | 7 | 65 | 0.051547 | 0.074205 | 0.010309 | 0.033186 | 1.076154 | 0.039592 | 0.084117 | 0.007918 | 0.037618 | 0.729178 | 0.011955 | 0.584615 | 0.670109 | 0.954450 | -0.125901 | 0.673136 | -0.174954 | 0.370611 | 1.000000 | 19.000000 | 6.000000 |

## Treasury-Adjusted MAD Portfolio

The table above reports the original horizon-normalized Sharpe ratios from the MAD run. Treasury-adjusted metrics are recomputed in `mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv` by subtracting the horizon-matched sum of monthly U.S. Treasury 1M log returns before annualizing Sharpe. The risk-free input is `inputs/treasury_1mo_monthly_rf_used.csv`, and the calculation has `0` missing risk-free rows.

Top Treasury-adjusted result: `distributed_lag`, `h=1`, `raw`, `L=5`, Treasury-adjusted Sharpe `0.915243` versus matched equal-weight Treasury-adjusted Sharpe `0.548530`.

Best h=5 GNC-informed result: `elastic_net_panel`, `direction_risk_scaled`, `L=7`, Treasury-adjusted Sharpe `0.837631` versus matched equal-weight Treasury-adjusted Sharpe `0.499941`. The closest no-GNC baseline is `historical_mean`, `h=5`, `direction_risk_scaled`, `L=7`, Treasury-adjusted Sharpe `0.830063`.

Supporting artifacts:

- `mad_portfolio_cleaned/treasury_adjusted_summary.md`
- `mad_portfolio_cleaned/top20_treasury_adjusted.csv`
- `mad_portfolio_cleaned/plots/top_mad_treasury_adjusted_sharpe.svg`
