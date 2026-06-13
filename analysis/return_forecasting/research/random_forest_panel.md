# Random forest panel model for country index returns

## Proposed model

`random_forest_panel` is a pooled monthly panel model for expected country stock-index returns. For each forecast month, it fits a `sklearn.ensemble.RandomForestRegressor` only on observations with `date` strictly before the prediction date. The target is `target_return`; predictors are six GNC lags, six return lags, GNC moving averages and deltas, rolling volatility, data-coverage controls, and known country/time features. Current-month `gnc` is available behind `include_current_gnc=True`, but the default excludes it to reduce timing and leakage risk.

The default specification is intentionally conservative:

- pooled panel rather than country-by-country fits, because the country-month sample is likely small;
- `max_depth=3`, `min_samples_leaf=10`, and `max_features="sqrt"` to limit variance;
- train-window-only winsorization of predictors and target at the 2nd/98th percentiles;
- prediction clipping at train-window 2nd/98th target percentiles;
- 60% shrinkage of the random-forest forecast toward the same past-only historical-mean fallback used by the benchmark;
- fallback to a past-only country historical mean, then global historical mean, when the expanding training set is too small or a prediction row has incomplete features.

## Round 1 challenge update

The first benchmark window showed the random forest had the best strategy Sharpe among tested models, but worse RMSE and MAE than the historical-mean and elastic-net baselines. That pattern suggests the model may contain useful ranking or sign information while producing overlarge point forecasts. The Round 2 default therefore does not add more features or deeper trees. It reduces variance and forecast amplitude through shallower trees, larger leaves, stricter clipping, exclusion of contemporaneous `gnc`, and explicit shrinkage to the historical mean.

This is a defensible compromise for a portfolio signal: keep the nonlinear interaction channel that helped Sharpe, but move the level forecast closer to the low-variance benchmark so RMSE is not dominated by noisy tree forecasts.

## Research basis

1. Gu, Kelly, and Xiu (2020), "Empirical Asset Pricing via Machine Learning", *Review of Financial Studies*, DOI: [10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009). The paper compares machine-learning methods for expected-return prediction and finds that trees and neural networks can add value by capturing nonlinearities and interactions that linear models miss. This supports using a tree ensemble when container activity, lagged returns, and volatility may interact in state-dependent ways.

2. Ciner (2019), "Do industry returns predict the stock market? A reprise using the random forest", *Quarterly Review of Economics and Finance*, DOI: [10.1016/j.qref.2018.11.001](https://doi.org/10.1016/j.qref.2018.11.001). This is directly related to market-index return forecasting with random forests. It motivates using multiple lagged market features jointly rather than testing predictors one at a time.

3. Rapach, Strauss, and Zhou (2013), "International stock return predictability: What is the role of the United States?", *Journal of Finance*, DOI: [10.1111/jofi.12041](https://doi.org/10.1111/jofi.12041). The paper studies international equity market predictability and cross-market information flow. It supports the panel framing and the use of lagged market variables for country index forecasts.

4. Qi (1999), "Nonlinear Predictability of Stock Returns Using Financial and Economic Variables", *Journal of Business & Economic Statistics*, DOI: [10.1080/07350015.1999.10524830](https://doi.org/10.1080/07350015.1999.10524830). Although it uses neural networks rather than random forests, it is an early empirical basis for nonlinear return forecasting with economic and financial predictors.

## Assumptions

- The monthly panel has already been built without look-ahead bias, especially for `gnc_lag_*`, `return_lag_*`, rolling volatility, and moving averages.
- `target_return` is the next-month or same-month forecast target intended by the upstream panel builder; the model only enforces that training rows have earlier dates than the row being predicted.
- Country identity and calendar month are known at forecast time, so country/time features are allowed.
- Missing predictor rows are not imputed by default; they use historical-mean fallback to avoid introducing accidental future information.
- The shrinkage weight is fixed ex ante in the config. It is not tuned inside the evaluation window.

## Failure modes

- Small samples can make tree splits unstable. The defaults reduce variance, but forecasts should be benchmarked against historical means and linear models.
- Random forests are poor extrapolators. If the GNC signal moves outside the historical range, clipping and tree averaging can mute the forecast.
- A pooled country code can capture persistent country-level return differences, but it may also overfit if countries enter or leave the panel.
- If market regimes shift, expanding-window training may overweight stale relationships. A rolling-window variant could be tested as a robustness check, but the required implementation uses expanding windows.
- Winsorization protects against extreme errors but can suppress genuine crisis signals if those extremes are economically meaningful.
- Shrinkage can reduce RMSE while weakening the cross-sectional spread that generated Sharpe in Round 1. If Sharpe drops sharply, the next robustness test should compare fixed shrinkage levels chosen before evaluation, not optimized on the reported window.
