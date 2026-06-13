# Distributed-lag ARX model for country index returns

## Model choice

This model forecasts monthly country equity-index returns from lagged returns and
lagged container GNC. It is an ARX/distributed-lag specification, not a pure OLS
or Elastic Net benchmark. The return lags capture short-run return dynamics,
while the GNC lags are compressed into a small set of constrained lag summaries:

- geometric lag aggregates, where recent GNC receives higher weight;
- Almon-style level, slope, and curvature factors across the GNC lag profile;
- optional lagged coverage/risk controls already present in the panel.

The implementation uses expanding-window out-of-sample estimation. For each
country-month forecast, the fitting sample is restricted to rows strictly before
the prediction date. The primary model is country-specific. If a country has too
few complete historical rows, the model falls back to a pooled ridge regression
with country fixed effects. If the pooled regression is also underidentified, it
uses a historical-mean forecast.

## Research basis

1. Almon (1965), "The Distributed Lag Between Capital Appropriations and
   Expenditures", *Econometrica*, 33(1), 178-196.
   Link: https://www.econometricsociety.org/publications/econometrica/1965/01/01/distributed-lag-between-capital-appropriations-and-expenditures
   Stable link: https://www.jstor.org/stable/1911894

   Almon's polynomial distributed lag idea is useful when individual lag
   coefficients are hard to estimate separately. The GNC panel has a limited
   monthly history per country, so estimating six unconstrained GNC-lag
   coefficients per country would be noisy. The model therefore uses low-order
   lag-profile factors: level, slope, and curvature.

2. Ghysels, Sinko, and Valkanov (2007), "MIDAS Regressions: Further Results and
   New Directions", *Econometric Reviews*, 26(1), 53-90.
   DOI: https://doi.org/10.1080/07474930600972467

   MIDAS regressions show how parsimonious lag weighting can make time-series
   forecasting feasible when the lag distribution matters but the available
   sample is short. This model is MIDAS-inspired rather than a true mixed-
   frequency model: all inputs are monthly, but the same principle is used to
   summarize several GNC lags with a few structured features.

3. Ghysels, Santa-Clara, and Valkanov (2005), "There is a Risk-Return Tradeoff
   After All", *Journal of Financial Economics*, 76(3), 509-548.
   NBER working paper DOI: https://doi.org/10.3386/w10913

   This paper applies MIDAS-style forecasting in equity markets and motivates
   using lagged information in a tightly parameterized way for conditional
   expected-return problems. The country GNC signal is a slow-moving real-
   activity proxy, so a lag-distribution approach is a natural fit.

4. Jordà (2005), "Estimation and Inference of Impulse Responses by Local
   Projections", *American Economic Review*, 95(1), 161-182.
   DOI: https://doi.org/10.1257/0002828053828518

   Jordà's local-projection framework motivates direct horizon-specific
   forecasting rather than relying on a fully specified dynamic system. This
   implementation is the one-month horizon version: it directly predicts next
   observed monthly returns using lagged variables and can be extended to
   separate horizon-specific regressions if the panel builder creates
   multi-month target returns.

## Why this fits container GNC

Container activity should affect market expectations with delay and possible
decay rather than only contemporaneously. A distributed-lag ARX model lets the
forecast use the whole recent GNC profile while keeping the number of estimated
parameters small. The fallback hierarchy also matches the likely data shape:
large countries may have enough monthly history for country-specific estimates,
while smaller countries benefit from pooled information.

## Assumptions

- GNC is observed and lagged correctly before the forecast date.
- The one-month target return is aligned so that `gnc_lag_1` and
  `return_lag_1` are known at prediction time.
- The lag profile is smooth enough that geometric and polynomial summaries are
  more stable than separate unrestricted lag coefficients.
- A linear conditional-mean approximation is adequate for this benchmark.
- Pooled fallback assumes some common return-GNC relationship across countries,
  after allowing country intercept shifts.

## Failure modes

- If market returns are dominated by abrupt policy, war, liquidity, or currency
  shocks, lagged container activity may have little short-run predictive value.
- If country GNC coverage changes over time, lag summaries can confound true
  activity shifts with measurement changes.
- If the lag alignment is wrong, the model can either leak future information or
  miss the predictive window entirely.
- Country-specific fits can be unstable with short histories; ridge shrinkage
  and pooled fallback reduce but do not eliminate that risk.
- A linear lag profile can miss threshold effects, nonlinear supply-chain
  disruptions, or asymmetric reactions to positive and negative GNC changes.
