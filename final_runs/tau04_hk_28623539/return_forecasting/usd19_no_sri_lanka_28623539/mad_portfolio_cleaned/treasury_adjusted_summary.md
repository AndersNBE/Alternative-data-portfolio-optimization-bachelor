# Treasury-Adjusted MAD Portfolio

Sharpe ratios subtract the horizon-matched sum of monthly U.S. Treasury 1M log returns before horizon-normalizing and annualizing.

- Return rows: `78750`
- Missing risk-free rows: `0`
- Portfolio date range: `2020-06` to `2026-04`

## Top Configurations

| model_id | forecast_horizon_months | signal_variant | lookback_months | n_periods_with_rf | annualized_sharpe_treasury_excess | equal_weight_annualized_sharpe_treasury_excess | annualized_sharpe_original | equal_weight_annualized_sharpe_original | compound_excess_return_treasury | equal_weight_compound_excess_return_treasury | max_drawdown_excess_treasury | equal_weight_max_drawdown_excess_treasury | mean_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distributed_lag | 1 | raw | 5 | 71 | 0.915243 | 0.548530 | 1.121448 | 0.747525 | 1.160238 | 0.607631 | -0.202345 | -0.241810 | 0.525705 |
| distributed_lag | 1 | raw | 6 | 70 | 0.841495 | 0.496425 | 1.076877 | 0.697806 | 0.854602 | 0.526790 | -0.164205 | -0.241810 | 0.509958 |
| elastic_net_panel | 5 | direction_risk_scaled | 7 | 65 | 0.837631 | 0.499941 | 1.116656 | 0.729178 | 0.667857 | 0.416851 | -0.129014 | -0.179751 | 0.373371 |
| distributed_lag | 5 | direction_risk_scaled | 7 | 65 | 0.830918 | 0.499941 | 1.113261 | 0.729178 | 0.658066 | 0.416851 | -0.127920 | -0.179751 | 0.379153 |
| historical_mean | 5 | direction_risk_scaled | 7 | 65 | 0.830063 | 0.499941 | 1.108387 | 0.729178 | 0.660916 | 0.416851 | -0.132058 | -0.179751 | 0.378256 |
| ols_predictive | 5 | direction_risk_scaled | 7 | 65 | 0.829937 | 0.499941 | 1.108189 | 0.729178 | 0.664091 | 0.416851 | -0.132553 | -0.179751 | 0.370572 |
| distributed_lag | 1 | direction | 5 | 71 | 0.819383 | 0.548530 | 1.027566 | 0.747525 | 0.974706 | 0.607631 | -0.224569 | -0.241810 | 0.527420 |
| random_forest_panel | 5 | direction_risk_scaled | 7 | 65 | 0.812881 | 0.499941 | 1.096591 | 0.729178 | 0.631422 | 0.416851 | -0.124763 | -0.179751 | 0.382175 |
| ols_predictive | 5 | raw | 7 | 65 | 0.812656 | 0.499941 | 1.075496 | 0.729178 | 0.675702 | 0.416851 | -0.135617 | -0.179751 | 0.379527 |
| elastic_net_panel | 5 | direction | 7 | 65 | 0.809138 | 0.499941 | 1.075860 | 0.729178 | 0.662779 | 0.416851 | -0.138384 | -0.179751 | 0.390856 |
| ols_predictive | 5 | direction | 7 | 65 | 0.808344 | 0.499941 | 1.077495 | 0.729178 | 0.653955 | 0.416851 | -0.136373 | -0.179751 | 0.369299 |
| historical_mean | 5 | direction | 7 | 65 | 0.807476 | 0.499941 | 1.076154 | 0.729178 | 0.655074 | 0.416851 | -0.137297 | -0.179751 | 0.370611 |
| random_forest_panel | 5 | raw | 7 | 65 | 0.805643 | 0.499941 | 1.069374 | 0.729178 | 0.671125 | 0.416851 | -0.137814 | -0.179751 | 0.384563 |
| distributed_lag | 5 | direction_risk_scaled | 8 | 64 | 0.804689 | 0.464219 | 1.093508 | 0.698779 | 0.586801 | 0.366607 | -0.120060 | -0.179751 | 0.354588 |
| distributed_lag | 1 | raw | 10 | 66 | 0.804544 | 0.501822 | 1.037132 | 0.711877 | 0.795122 | 0.506561 | -0.183113 | -0.241810 | 0.374107 |
| distributed_lag | 5 | raw | 7 | 65 | 0.802244 | 0.499941 | 1.065790 | 0.729178 | 0.669196 | 0.416851 | -0.139143 | -0.179751 | 0.378094 |
| distributed_lag | 1 | direction | 6 | 70 | 0.795597 | 0.496425 | 1.025056 | 0.697806 | 0.825210 | 0.526790 | -0.167162 | -0.241810 | 0.494140 |
| distributed_lag | 5 | direction | 7 | 65 | 0.793607 | 0.499941 | 1.061140 | 0.729178 | 0.643886 | 0.416851 | -0.138603 | -0.179751 | 0.394542 |
| historical_mean | 5 | raw | 7 | 65 | 0.790177 | 0.499941 | 1.049058 | 0.729178 | 0.661815 | 0.416851 | -0.140683 | -0.179751 | 0.388729 |
| elastic_net_panel | 5 | raw | 7 | 65 | 0.788395 | 0.499941 | 1.048469 | 0.729178 | 0.659448 | 0.416851 | -0.140480 | -0.179751 | 0.382217 |

## Best Configuration by Model

| model_id | forecast_horizon_months | signal_variant | lookback_months | n_periods_with_rf | annualized_sharpe_treasury_excess | equal_weight_annualized_sharpe_treasury_excess | annualized_sharpe_original | equal_weight_annualized_sharpe_original | compound_excess_return_treasury | equal_weight_compound_excess_return_treasury | max_drawdown_excess_treasury | equal_weight_max_drawdown_excess_treasury | mean_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distributed_lag | 1 | raw | 5 | 71 | 0.915243 | 0.548530 | 1.121448 | 0.747525 | 1.160238 | 0.607631 | -0.202345 | -0.241810 | 0.525705 |
| elastic_net_panel | 5 | direction_risk_scaled | 7 | 65 | 0.837631 | 0.499941 | 1.116656 | 0.729178 | 0.667857 | 0.416851 | -0.129014 | -0.179751 | 0.373371 |
| historical_mean | 5 | direction_risk_scaled | 7 | 65 | 0.830063 | 0.499941 | 1.108387 | 0.729178 | 0.660916 | 0.416851 | -0.132058 | -0.179751 | 0.378256 |
| ols_predictive | 5 | direction_risk_scaled | 7 | 65 | 0.829937 | 0.499941 | 1.108189 | 0.729178 | 0.664091 | 0.416851 | -0.132553 | -0.179751 | 0.370572 |
| random_forest_panel | 5 | direction_risk_scaled | 7 | 65 | 0.812881 | 0.499941 | 1.096591 | 0.729178 | 0.631422 | 0.416851 | -0.124763 | -0.179751 | 0.382175 |
| always_positive | 5 | raw | 7 | 65 | 0.768995 | 0.499941 | 1.030209 | 0.729178 | 0.631500 | 0.416851 | -0.147270 | -0.179751 | 0.380170 |