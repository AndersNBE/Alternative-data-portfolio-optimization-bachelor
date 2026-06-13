# GNC Ablation

- Cleaned segmentation model: `03_bce_dice_group_100ep_20260521_204006`
- Threshold: `0.4`
- Target: `cumulative_forward_log_return_month_t_to_t_plus_h_minus_1`
- Test start: `2020-01-01`

## Delta Metrics

Positive `rmse_improvement_from_gnc` means GNC reduced RMSE. Positive `direction_delta_from_gnc` means GNC improved directional accuracy.

| forecast_horizon_months | model_id | rmse_improvement_from_gnc | corr_delta_from_gnc | direction_delta_from_gnc | sharpe_delta_from_gnc |
| --- | --- | --- | --- | --- | --- |
| 1 | always_positive | 0.000000 | nan | 0.000000 | 0.000000 |
| 1 | distributed_lag | 0.000022 | 0.004516 | -0.005085 | -0.062506 |
| 1 | elastic_net_panel | 0.000028 | 0.002097 | 0.001453 | 0.008208 |
| 1 | historical_mean | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | ols_predictive | -0.005013 | -0.504790 | -0.177019 | -1.281321 |
| 1 | random_forest_panel | -0.000231 | 0.000047 | 0.001332 | -0.044519 |
| 2 | always_positive | 0.000000 | nan | 0.000000 | 0.000000 |
| 2 | distributed_lag | -0.000039 | -0.000759 | -0.003765 | -0.044078 |
| 2 | elastic_net_panel | 0.000039 | 0.002968 | 0.002672 | 0.011143 |
| 2 | historical_mean | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | ols_predictive | -0.003168 | -0.244422 | -0.080530 | -0.537617 |
| 2 | random_forest_panel | -0.000447 | -0.009177 | -0.000729 | -0.016865 |
| 3 | always_positive | 0.000000 | nan | 0.000000 | 0.000000 |
| 3 | distributed_lag | -0.000048 | -0.001467 | -0.003422 | -0.029041 |
| 3 | elastic_net_panel | 0.000031 | 0.001934 | 0.004033 | 0.037152 |
| 3 | historical_mean | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | ols_predictive | -0.002333 | -0.136280 | -0.044243 | -0.370144 |
| 3 | random_forest_panel | -0.000608 | -0.019575 | -0.002322 | -0.032481 |
| 4 | always_positive | 0.000000 | nan | 0.000000 | 0.000000 |
| 4 | distributed_lag | -0.000015 | 0.000654 | -0.003458 | -0.040680 |
| 4 | elastic_net_panel | 0.000014 | 0.000228 | -0.002594 | -0.019432 |
| 4 | historical_mean | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | ols_predictive | -0.002149 | -0.100027 | -0.035450 | -0.271896 |
| 4 | random_forest_panel | -0.000762 | -0.019969 | -0.009387 | -0.083260 |
| 5 | always_positive | 0.000000 | nan | 0.000000 | 0.000000 |
| 5 | distributed_lag | -0.000046 | -0.000620 | -0.000501 | -0.001786 |
| 5 | elastic_net_panel | 0.000037 | 0.001104 | -0.000752 | -0.001211 |
| 5 | historical_mean | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | ols_predictive | -0.001713 | -0.068989 | -0.018549 | -0.179488 |
| 5 | random_forest_panel | -0.000997 | -0.024692 | -0.009525 | -0.117992 |

## Full Metrics

| forecast_horizon_months | feature_set | model_id | rmse | corr | directional_accuracy | strategy_annualized_sharpe |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | with_gnc | distributed_lag | 0.051791 | 0.717063 | 0.828672 | 2.341333 |
| 1 | with_gnc | random_forest_panel | 0.053181 | 0.606205 | 0.840538 | 2.142342 |
| 1 | with_gnc | elastic_net_panel | 0.057628 | 0.351111 | 0.711830 | 1.139125 |
| 1 | with_gnc | ols_predictive | 0.060198 | -0.033777 | 0.559148 | 0.115054 |
| 1 | with_gnc | historical_mean | 0.060198 | -0.034340 | 0.560601 | 0.117654 |
| 1 | with_gnc | always_positive | 0.994243 | nan | 0.591476 | 0.438728 |
| 2 | with_gnc | distributed_lag | 0.079457 | 0.418936 | 0.685412 | 1.351545 |
| 2 | with_gnc | random_forest_panel | 0.079677 | 0.369903 | 0.676910 | 1.324721 |
| 2 | with_gnc | elastic_net_panel | 0.083236 | 0.179054 | 0.624195 | 0.802032 |
| 2 | with_gnc | historical_mean | 0.084986 | 0.022118 | 0.562128 | 0.325829 |
| 2 | with_gnc | ols_predictive | 0.084989 | 0.021785 | 0.562371 | 0.327092 |
| 2 | with_gnc | always_positive | 0.988104 | nan | 0.611320 | 0.638260 |
| 3 | with_gnc | random_forest_panel | 0.097095 | 0.300087 | 0.632975 | 1.133968 |
| 3 | with_gnc | distributed_lag | 0.097900 | 0.294151 | 0.633830 | 1.070054 |
| 3 | with_gnc | elastic_net_panel | 0.100656 | 0.143209 | 0.601076 | 0.746728 |
| 3 | with_gnc | historical_mean | 0.101965 | 0.065231 | 0.560865 | 0.403320 |
| 3 | with_gnc | ols_predictive | 0.101981 | 0.064797 | 0.560254 | 0.408401 |
| 3 | with_gnc | always_positive | 0.981350 | nan | 0.610120 | 0.821257 |
| 4 | with_gnc | random_forest_panel | 0.106093 | 0.321244 | 0.674160 | 1.286687 |
| 4 | with_gnc | distributed_lag | 0.108839 | 0.251470 | 0.637969 | 1.018385 |
| 4 | with_gnc | elastic_net_panel | 0.111116 | 0.144003 | 0.614377 | 0.756135 |
| 4 | with_gnc | historical_mean | 0.112340 | 0.090512 | 0.585104 | 0.559120 |
| 4 | with_gnc | ols_predictive | 0.112374 | 0.090095 | 0.584857 | 0.575726 |
| 4 | with_gnc | always_positive | 0.970857 | nan | 0.653903 | 1.120319 |
| 5 | with_gnc | random_forest_panel | 0.116814 | 0.325695 | 0.666500 | 1.332168 |
| 5 | with_gnc | distributed_lag | 0.121376 | 0.210851 | 0.625768 | 0.969332 |
| 5 | with_gnc | elastic_net_panel | 0.123194 | 0.135993 | 0.610102 | 0.805811 |
| 5 | with_gnc | historical_mean | 0.124328 | 0.096521 | 0.585412 | 0.606811 |
| 5 | with_gnc | ols_predictive | 0.124383 | 0.095921 | 0.587417 | 0.613042 |
| 5 | with_gnc | always_positive | 0.962941 | nan | 0.677904 | 1.292014 |
| 1 | no_gnc | distributed_lag | 0.051813 | 0.712547 | 0.833757 | 2.403839 |
| 1 | no_gnc | random_forest_panel | 0.052950 | 0.606158 | 0.839206 | 2.186861 |
| 1 | no_gnc | ols_predictive | 0.055184 | 0.471013 | 0.736167 | 1.396375 |
| 1 | no_gnc | elastic_net_panel | 0.057656 | 0.349014 | 0.710377 | 1.130918 |
| 1 | no_gnc | historical_mean | 0.060198 | -0.034340 | 0.560601 | 0.117654 |
| 1 | no_gnc | always_positive | 0.994243 | nan | 0.591476 | 0.438728 |
| 2 | no_gnc | random_forest_panel | 0.079231 | 0.379080 | 0.677639 | 1.341586 |
| 2 | no_gnc | distributed_lag | 0.079417 | 0.419695 | 0.689178 | 1.395624 |
| 2 | no_gnc | ols_predictive | 0.081821 | 0.266207 | 0.642901 | 0.864709 |
| 2 | no_gnc | elastic_net_panel | 0.083275 | 0.176086 | 0.621523 | 0.790889 |
| 2 | no_gnc | historical_mean | 0.084986 | 0.022118 | 0.562128 | 0.325829 |
| 2 | no_gnc | always_positive | 0.988104 | nan | 0.611320 | 0.638260 |
| 3 | no_gnc | random_forest_panel | 0.096487 | 0.319662 | 0.635297 | 1.166449 |
| 3 | no_gnc | distributed_lag | 0.097853 | 0.295618 | 0.637253 | 1.099095 |
| 3 | no_gnc | ols_predictive | 0.099649 | 0.201077 | 0.604498 | 0.778545 |
| 3 | no_gnc | elastic_net_panel | 0.100687 | 0.141275 | 0.597042 | 0.709576 |
| 3 | no_gnc | historical_mean | 0.101965 | 0.065231 | 0.560865 | 0.403320 |
| 3 | no_gnc | always_positive | 0.981350 | nan | 0.610120 | 0.821257 |
| 4 | no_gnc | random_forest_panel | 0.105331 | 0.341214 | 0.683547 | 1.369947 |
| 4 | no_gnc | distributed_lag | 0.108825 | 0.250815 | 0.641428 | 1.059065 |
| 4 | no_gnc | ols_predictive | 0.110225 | 0.190122 | 0.620306 | 0.847621 |
| 4 | no_gnc | elastic_net_panel | 0.111131 | 0.143775 | 0.616971 | 0.775567 |
| 4 | no_gnc | historical_mean | 0.112340 | 0.090512 | 0.585104 | 0.559120 |
| 4 | no_gnc | always_positive | 0.970857 | nan | 0.653903 | 1.120319 |
| 5 | no_gnc | random_forest_panel | 0.115817 | 0.350387 | 0.676025 | 1.450159 |
| 5 | no_gnc | distributed_lag | 0.121330 | 0.211471 | 0.626269 | 0.971117 |
| 5 | no_gnc | ols_predictive | 0.122671 | 0.164910 | 0.605966 | 0.792530 |
| 5 | no_gnc | elastic_net_panel | 0.123231 | 0.134890 | 0.610853 | 0.807022 |
| 5 | no_gnc | historical_mean | 0.124328 | 0.096521 | 0.585412 | 0.606811 |
| 5 | no_gnc | always_positive | 0.962941 | nan | 0.677904 | 1.292014 |