from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / "bachelor-return-forecast-mpl"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = Path(tempfile.gettempdir()) / "bachelor-return-forecast-cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.return_forecasting.data import (
    FEATURE_COLUMNS,
    add_usd_market_returns,
    file_fingerprint,
    load_country_gnc,
    load_market_monthly_returns,
)
from analysis.return_forecasting.evaluation import add_relative_metrics, evaluate_predictions


DEFAULT_CLEANED_BUNDLE = Path("final_runs/tau04_hk_28623539/daily_container_index")
DEFAULT_SELECTED_MODEL_JSON = Path("pipelines/final_selected_model_tau04_bd5d48e.json")
DEFAULT_MODELS = [
    "historical_mean",
    "always_positive",
    "ols_predictive",
    "elastic_net_panel",
    "distributed_lag",
    "random_forest_panel",
]
DEFAULT_LOOKBACKS = list(range(5, 19))
DEFAULT_SIGNAL_VARIANTS = ["raw", "direction", "direction_risk_scaled"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run h=1..5 cleaned-model return forecasts and MAD portfolio backtests. "
            "Targets are cumulative forward monthly log returns."
        )
    )
    parser.add_argument("--country-gnc", type=Path, default=DEFAULT_CLEANED_BUNDLE / "country_gnc.csv")
    parser.add_argument("--port-timeseries", type=Path, default=DEFAULT_CLEANED_BUNDLE / "port_timeseries.csv")
    parser.add_argument("--selected-model-json", type=Path, default=DEFAULT_SELECTED_MODEL_JSON)
    parser.add_argument("--cleaned-model-id", default="", help="Override the model id recorded in output metadata.")
    parser.add_argument(
        "--cleaned-model-threshold",
        type=float,
        default=None,
        help="Override the segmentation threshold recorded in output metadata.",
    )
    parser.add_argument("--market-indices", type=Path, required=True)
    parser.add_argument(
        "--fx-rates",
        type=Path,
        default=None,
        help="Optional USD-per-local-currency rates. When supplied, forecasts use unhedged USD returns.",
    )
    parser.add_argument(
        "--exclude-countries",
        default="",
        help="Comma-separated country names to exclude before panel construction.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/outputs/return_forecasting"))
    parser.add_argument("--suite-id", default="")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--horizons", default="1,2,3,4,5")
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--min-train-months", type=int, default=36)
    parser.add_argument("--min-non-null-features", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mad-lookbacks", default=",".join(str(x) for x in DEFAULT_LOOKBACKS))
    parser.add_argument("--mad-max-weight", type=float, default=0.35)
    parser.add_argument("--skip-mad", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test cap after panel preparation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_id = args.suite_id or f"cleaned_return_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = args.out / suite_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    log_path = suite_dir / "run.log"

    model_ids = _parse_list(args.models)
    horizons = [int(x) for x in _parse_list(args.horizons)]
    lookbacks = [int(x) for x in _parse_list(args.mad_lookbacks)]
    excluded_countries = _parse_list(args.exclude_countries)

    selected_model = _read_json(args.selected_model_json) if args.selected_model_json.exists() else {}
    cleaned_model_id = args.cleaned_model_id or selected_model.get("model_id")
    cleaned_model_threshold = (
        args.cleaned_model_threshold
        if args.cleaned_model_threshold is not None
        else selected_model.get("threshold")
    )
    config = {
        "suite_id": suite_id,
        "country_gnc": str(args.country_gnc),
        "port_timeseries": str(args.port_timeseries),
        "selected_model_json": str(args.selected_model_json),
        "cleaned_model_id": cleaned_model_id,
        "cleaned_model_threshold": cleaned_model_threshold,
        "market_indices": str(args.market_indices),
        "fx_rates": str(args.fx_rates) if args.fx_rates else None,
        "return_currency": "USD" if args.fx_rates else "local",
        "target_definition": "cumulative_forward_log_return_month_t_to_t_plus_h_minus_1",
        "feature_timing": "all predictors are lagged at least one month relative to the first forecast month",
        "horizons": horizons,
        "models": model_ids,
        "excluded_countries": excluded_countries,
        "test_start": args.test_start,
        "min_train_months": args.min_train_months,
        "min_non_null_features": args.min_non_null_features,
        "seed": args.seed,
        "mad_lookbacks": lookbacks,
        "mad_max_weight": args.mad_max_weight,
    }
    _write_json(suite_dir / "config.json", config)
    _write_json(
        suite_dir / "input_fingerprints.json",
        {
            "country_gnc": file_fingerprint(args.country_gnc),
            "port_timeseries": file_fingerprint(args.port_timeseries)
            if args.port_timeseries.exists()
            else None,
            "selected_model_json": file_fingerprint(args.selected_model_json)
            if args.selected_model_json.exists()
            else None,
            "market_indices": file_fingerprint(args.market_indices),
            "fx_rates": file_fingerprint(args.fx_rates) if args.fx_rates else None,
        },
    )

    with Tee(log_path):
        print(f"Suite: {suite_id}")
        print(f"Output: {suite_dir}")
        print(f"Cleaned GNC: {args.country_gnc}")
        print(f"Market indices: {args.market_indices}")
        print(f"Return currency: {config['return_currency']}")
        print(f"Models: {', '.join(model_ids)}")
        print(f"Horizons: {horizons}")
        if excluded_countries:
            print(f"Excluded countries: {', '.join(excluded_countries)}")
        if selected_model:
            print(
                "Selected segmentation model: "
                f"{cleaned_model_id} | threshold={cleaned_model_threshold}"
            )

        country_gnc = load_country_gnc(args.country_gnc)
        market_returns = load_market_monthly_returns(args.market_indices)
        if args.fx_rates:
            market_returns = add_usd_market_returns(market_returns, args.fx_rates)
            market_returns["market_return"] = market_returns["market_return_usd"]
            market_returns = market_returns.dropna(subset=["market_return"]).copy()
        if excluded_countries:
            country_gnc = country_gnc[~country_gnc["country"].isin(excluded_countries)].copy()
            market_returns = market_returns[~market_returns["country"].isin(excluded_countries)].copy()
        print(f"Loaded GNC rows: {len(country_gnc):,}")
        print(f"Loaded monthly market return rows: {len(market_returns):,}")

        combined_metrics: list[pd.DataFrame] = []
        all_eval_predictions: list[pd.DataFrame] = []

        for horizon in horizons:
            run_dir = suite_dir / f"forecast_cleaned_h{horizon}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 80)
            print(f"Forecast horizon h={horizon}")
            panel, diagnostics = build_cumulative_forecast_panel(
                country_gnc=country_gnc,
                market_returns=market_returns,
                forecast_horizon_months=horizon,
                min_non_null_features=args.min_non_null_features,
            )
            if args.max_rows > 0:
                panel = panel.sort_values(["date", "country"]).head(args.max_rows).copy()
                diagnostics["max_rows_cap"] = args.max_rows

            eval_start = pd.Timestamp(args.test_start)
            diagnostics.update(
                {
                    "rows_after_feature_drop": int(len(panel)),
                    "rows_in_eval_window": int((pd.to_datetime(panel["date"]) >= eval_start).sum()),
                    "countries_after_filter": sorted(panel["country"].dropna().unique()),
                    "date_min": _date_min(panel["date"]),
                    "date_max": _date_max(panel["date"]),
                    "eval_start": str(eval_start.date()),
                }
            )
            panel.to_csv(run_dir / "panel.csv", index=False)
            _write_json(run_dir / "data_diagnostics.json", diagnostics)
            _write_json(run_dir / "config.json", {**config, "forecast_horizon_months": horizon})
            print(
                "Panel: "
                f"{len(panel):,} rows | {panel['country'].nunique()} countries | "
                f"eval rows={diagnostics['rows_in_eval_window']:,}"
            )

            predictions_df, model_meta = run_models(panel, model_ids, config)
            predictions_df.to_csv(run_dir / "predictions.csv", index=False)
            _write_json(run_dir / "model_meta.json", model_meta)

            eval_predictions = predictions_df[pd.to_datetime(predictions_df["date"]) >= eval_start].copy()
            eval_predictions["forecast_horizon_months"] = horizon
            eval_predictions.to_csv(run_dir / "predictions_eval_window.csv", index=False)
            all_eval_predictions.append(eval_predictions)

            metrics = add_relative_metrics(evaluate_predictions(eval_predictions))
            metrics.insert(0, "forecast_horizon_months", horizon)
            metrics.to_csv(run_dir / "metrics.csv", index=False)
            combined_metrics.append(metrics)
            write_forecast_summary(run_dir, metrics, diagnostics)
            plot_forecast_metrics(metrics, run_dir)
            print(metrics.to_string(index=False))

        combined = pd.concat(combined_metrics, ignore_index=True) if combined_metrics else pd.DataFrame()
        combined.to_csv(suite_dir / "combined_metrics.csv", index=False)
        if all_eval_predictions:
            pd.concat(all_eval_predictions, ignore_index=True).to_csv(
                suite_dir / "all_predictions_eval_window.csv", index=False
            )

        mad_metrics = pd.DataFrame()
        if not args.skip_mad and all_eval_predictions:
            print("\n" + "=" * 80)
            print("Running MAD portfolio grid")
            mad_dir = suite_dir / "mad_portfolio_cleaned"
            mad_metrics = run_mad_grid(
                predictions=pd.concat(all_eval_predictions, ignore_index=True),
                out_dir=mad_dir,
                lookbacks=lookbacks,
                signal_variants=DEFAULT_SIGNAL_VARIANTS,
                max_weight=args.mad_max_weight,
            )
            print(mad_metrics.sort_values("annualized_sharpe", ascending=False).head(20).to_string(index=False))

        write_suite_summary(suite_dir, combined, mad_metrics, config)
        plot_suite_direction(combined, suite_dir)
        print("\nDone.")
        print(f"All outputs: {suite_dir}")


def build_cumulative_forecast_panel(
    *,
    country_gnc: pd.DataFrame,
    market_returns: pd.DataFrame,
    forecast_horizon_months: int,
    min_non_null_features: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if forecast_horizon_months < 1:
        raise ValueError("forecast_horizon_months must be >= 1")

    market = market_returns.sort_values(["country", "date"]).copy()
    market["target_return"] = market.groupby("country", group_keys=False)["market_return"].transform(
        lambda s: pd.concat(
            [s.shift(-offset) for offset in range(forecast_horizon_months)],
            axis=1,
        ).sum(axis=1, min_count=forecast_horizon_months)
    )

    panel = market[
        ["country", "index_name", "symbol", "date", "month_str", "target_return", "market_return"]
    ].merge(country_gnc, on=["country", "date", "month_str"], how="left")
    panel = panel.sort_values(["country", "date"]).reset_index(drop=True)

    group = panel.groupby("country", group_keys=False)
    panel["gnc"] = group["gnc_same_month"].shift(1)
    for lag in range(1, 7):
        panel[f"gnc_lag_{lag}"] = group["gnc_same_month"].shift(lag)
        panel[f"return_lag_{lag}"] = group["market_return"].shift(lag)
    panel["gnc_ma_3"] = group["gnc_same_month"].transform(lambda s: s.shift(1).rolling(3).mean())
    panel["gnc_ma_6"] = group["gnc_same_month"].transform(lambda s: s.shift(1).rolling(6).mean())
    panel["gnc_delta_1"] = panel["gnc_lag_1"] - panel["gnc_lag_2"]
    panel["gnc_delta_3"] = panel["gnc_lag_1"] - panel["gnc_lag_4"]
    panel["rolling_vol_3"] = group["market_return"].transform(lambda s: s.shift(1).rolling(3).std())
    panel["rolling_vol_6"] = group["market_return"].transform(lambda s: s.shift(1).rolling(6).std())
    for col in ["n_ports", "mean_NC", "mean_observations_per_port"]:
        panel[col] = group[col].shift(1)

    panel["target_month_str"] = panel["month_str"]
    panel = panel.dropna(subset=["target_return"]).reset_index(drop=True)
    panel[FEATURE_COLUMNS] = panel[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    feature_non_null = panel[FEATURE_COLUMNS].notna().sum(axis=1)
    panel = panel[feature_non_null >= min_non_null_features].copy().reset_index(drop=True)

    market_countries = set(market_returns["country"].dropna().unique())
    gnc_countries = set(country_gnc["country"].dropna().unique())
    diagnostics = {
        "forecast_horizon_months": forecast_horizon_months,
        "target_definition": "cumulative_forward_log_return_month_t_to_t_plus_h_minus_1",
        "market_countries": sorted(market_countries),
        "gnc_countries": sorted(gnc_countries),
        "countries_missing_market_index": sorted(gnc_countries - market_countries),
        "countries_missing_gnc": sorted(market_countries - gnc_countries),
        "rows_before_feature_drop": int(len(market)),
    }
    return panel, diagnostics


def run_models(
    panel: pd.DataFrame,
    model_ids: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_predictions: list[pd.DataFrame] = []
    model_meta: dict[str, Any] = {}
    model_config = {
        "min_train_months": config["min_train_months"],
        "min_train_size": config["min_train_months"],
        "min_pooled_train": config["min_train_months"],
        "seed": config["seed"],
        "random_state": config["seed"],
        "available_feature_columns": FEATURE_COLUMNS,
    }
    for model_id in model_ids:
        print(f"Running model: {model_id}")
        try:
            module = importlib.import_module(f"analysis.return_forecasting.models.{model_id}")
            predictions, meta = module.run_model(panel.copy(), model_config)
            required = {
                "model_id",
                "country",
                "date",
                "month_str",
                "actual_return",
                "predicted_return",
                "train_size",
            }
            missing = required - set(predictions.columns)
            if missing:
                raise ValueError(f"{model_id} predictions missing columns: {sorted(missing)}")
            predictions["date"] = pd.to_datetime(predictions["date"]).dt.to_period("M").dt.to_timestamp()
            all_predictions.append(predictions)
            model_meta[model_id] = meta
            print(f"  produced {len(predictions):,} predictions")
        except Exception as exc:
            model_meta[model_id] = {"model_id": model_id, "error": str(exc)}
            print(f"  failed: {exc}")

    if not all_predictions:
        raise RuntimeError("No models produced predictions.")
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df = predictions_df.sort_values(["model_id", "date", "country"]).reset_index(drop=True)
    return predictions_df, model_meta


def run_mad_grid(
    *,
    predictions: pd.DataFrame,
    out_dir: Path,
    lookbacks: list[int],
    signal_variants: list[str],
    max_weight: float,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = predictions.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.to_period("M").dt.to_timestamp()
    work["actual_return"] = pd.to_numeric(work["actual_return"], errors="coerce")
    work["predicted_return"] = pd.to_numeric(work["predicted_return"], errors="coerce")
    work = work.dropna(subset=["actual_return", "predicted_return"])

    metrics_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for (model_id, horizon), group in work.groupby(["model_id", "forecast_horizon_months"]):
        actual_pivot = group.pivot_table(
            index="date", columns="country", values="actual_return", aggfunc="mean"
        ).sort_index()
        pred_pivot = group.pivot_table(
            index="date", columns="country", values="predicted_return", aggfunc="mean"
        ).sort_index()

        for lookback in lookbacks:
            for variant in signal_variants:
                backtest = backtest_mad(
                    actual_returns=actual_pivot,
                    predicted_returns=pred_pivot,
                    lookback_months=lookback,
                    signal_variant=variant,
                    max_weight=max_weight,
                )
                if backtest["returns"].empty:
                    continue

                returns = backtest["returns"]
                metrics = portfolio_metrics(returns, forecast_horizon_months=int(horizon))
                row = {
                    "model_id": model_id,
                    "forecast_horizon_months": int(horizon),
                    "signal_variant": variant,
                    "lookback_months": int(lookback),
                    **metrics,
                }
                metrics_rows.append(row)

                p = returns.copy()
                p.insert(0, "model_id", model_id)
                p.insert(1, "forecast_horizon_months", int(horizon))
                p.insert(2, "signal_variant", variant)
                p.insert(3, "lookback_months", int(lookback))
                portfolio_rows.extend(p.to_dict("records"))

                w = backtest["weights"].copy()
                w.insert(0, "model_id", model_id)
                w.insert(1, "forecast_horizon_months", int(horizon))
                w.insert(2, "signal_variant", variant)
                w.insert(3, "lookback_months", int(lookback))
                weight_rows.extend(w.to_dict("records"))

    metrics_df = pd.DataFrame(metrics_rows)
    portfolio_df = pd.DataFrame(portfolio_rows)
    weights_df = pd.DataFrame(weight_rows)
    metrics_df.to_csv(out_dir / "portfolio_metrics.csv", index=False)
    portfolio_df.to_csv(out_dir / "portfolio_returns.csv", index=False)
    weights_df.to_csv(out_dir / "weights.csv", index=False)
    _write_json(
        out_dir / "manifest.json",
        {
            "signal_variants": signal_variants,
            "lookbacks": lookbacks,
            "max_weight": max_weight,
            "optimizer": "linear_program_minimize_mean_absolute_deviation_with_equal_weight_signal_floor",
            "return_metric_note": (
                "Forecast targets are cumulative h-month log returns. MAD reports period returns "
                "and horizon-normalized monthly return/volatility/Sharpe."
            ),
        },
    )
    write_mad_summary(out_dir, metrics_df)
    plot_mad_top(metrics_df, out_dir)
    return metrics_df


def backtest_mad(
    *,
    actual_returns: pd.DataFrame,
    predicted_returns: pd.DataFrame,
    lookback_months: int,
    signal_variant: str,
    max_weight: float,
) -> dict[str, pd.DataFrame]:
    dates = sorted(actual_returns.index.intersection(predicted_returns.index))
    return_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    previous_weights: pd.Series | None = None

    for idx, date in enumerate(dates):
        if idx < lookback_months:
            continue
        history = actual_returns.loc[dates[idx - lookback_months : idx - 1]]
        realised = actual_returns.loc[date]
        signal = transform_signal(
            signal=predicted_returns.loc[date],
            history=history,
            variant=signal_variant,
        )
        countries = sorted(
            set(history.dropna(axis=1, how="any").columns)
            .intersection(signal.dropna().index)
            .intersection(realised.dropna().index)
        )
        if len(countries) < 3:
            continue
        scenarios = history[countries].to_numpy(dtype=float)
        expected = signal[countries].to_numpy(dtype=float)
        weights, optimal = solve_mad_weights(
            scenarios=scenarios,
            expected=expected,
            max_weight=max_weight,
        )
        if weights is None:
            weights = np.full(len(countries), 1.0 / len(countries))
            optimal = False

        realised_vec = realised[countries].to_numpy(dtype=float)
        port_return = float(np.dot(weights, realised_vec))
        equal_weight_return = float(np.mean(realised_vec))
        weights_series = pd.Series(weights, index=countries)
        turnover = (
            float(0.5 * np.abs(weights_series.subtract(previous_weights, fill_value=0.0)).sum())
            if previous_weights is not None
            else float("nan")
        )
        previous_weights = weights_series

        return_rows.append(
            {
                "date": pd.Timestamp(date),
                "portfolio_return": port_return,
                "equal_weight_return": equal_weight_return,
                "excess_return": port_return - equal_weight_return,
                "turnover": turnover,
                "optimal": bool(optimal),
                "n_assets": len(countries),
                "n_scenarios": scenarios.shape[0],
            }
        )
        for country, weight in zip(countries, weights):
            weight_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "country": country,
                    "weight": float(weight),
                    "signal": float(signal[country]),
                }
            )

    return {"returns": pd.DataFrame(return_rows), "weights": pd.DataFrame(weight_rows)}


def transform_signal(signal: pd.Series, history: pd.DataFrame, variant: str) -> pd.Series:
    out = pd.to_numeric(signal, errors="coerce").astype(float)
    if variant == "raw":
        return out
    if variant == "direction":
        return np.sign(out)
    if variant == "direction_risk_scaled":
        risk = history.std(axis=0, ddof=0).replace(0.0, np.nan)
        return np.sign(out) / risk
    raise ValueError(f"Unknown signal variant: {variant}")


def solve_mad_weights(
    *,
    scenarios: np.ndarray,
    expected: np.ndarray,
    max_weight: float,
) -> tuple[np.ndarray | None, bool]:
    scenarios = np.asarray(scenarios, dtype=float)
    n_scenarios, n_assets = scenarios.shape
    if n_assets == 0 or n_scenarios == 0:
        return None, False

    # Variables: weights n, absolute deviations n_scenarios, scenario mean 1.
    n_vars = n_assets + n_scenarios + 1
    mean_idx = n_assets + n_scenarios
    c = np.zeros(n_vars)
    c[n_assets : n_assets + n_scenarios] = 1.0 / n_scenarios

    a_eq = np.zeros((2, n_vars))
    a_eq[0, :n_assets] = 1.0
    # Bind mu to the mean scenario portfolio return for classical mean-centered MAD.
    a_eq[1, :n_assets] = scenarios.mean(axis=0)
    a_eq[1, mean_idx] = -1.0
    b_eq = np.array([1.0, 0.0])

    a_ub_rows = []
    b_ub_rows = []
    for s_idx in range(n_scenarios):
        row = np.zeros(n_vars)
        row[:n_assets] = scenarios[s_idx]
        row[n_assets + s_idx] = -1.0
        row[mean_idx] = -1.0
        a_ub_rows.append(row)
        b_ub_rows.append(0.0)

        row = np.zeros(n_vars)
        row[:n_assets] = -scenarios[s_idx]
        row[n_assets + s_idx] = -1.0
        row[mean_idx] = 1.0
        a_ub_rows.append(row)
        b_ub_rows.append(0.0)

    expected = np.asarray(expected, dtype=float)
    if np.isfinite(expected).all() and np.nanstd(expected) > 0:
        row = np.zeros(n_vars)
        row[:n_assets] = -expected
        a_ub_rows.append(row)
        b_ub_rows.append(-float(np.mean(expected)))

    bounds = [(0.0, max_weight) for _ in range(n_assets)]
    bounds += [(0.0, None) for _ in range(n_scenarios)]
    bounds += [(None, None)]

    result = linprog(
        c,
        A_ub=np.asarray(a_ub_rows, dtype=float),
        b_ub=np.asarray(b_ub_rows, dtype=float),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None, False
    weights = np.asarray(result.x[:n_assets], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return None, False
    weights = weights / weights.sum()
    return weights, True


def portfolio_metrics(returns: pd.DataFrame, forecast_horizon_months: int) -> dict[str, Any]:
    r = pd.to_numeric(returns["portfolio_return"], errors="coerce").dropna()
    ew = pd.to_numeric(returns["equal_weight_return"], errors="coerce").dropna()
    excess = pd.to_numeric(returns["excess_return"], errors="coerce").dropna()
    turnover = pd.to_numeric(returns["turnover"], errors="coerce").dropna()
    if r.empty:
        return {}
    period_vol = float(r.std(ddof=0))
    ew_vol = float(ew.std(ddof=0))
    horizon = max(int(forecast_horizon_months), 1)
    monthly = r / horizon
    ew_monthly = ew / horizon
    monthly_vol = float(period_vol / math.sqrt(horizon)) if period_vol > 0 else float("nan")
    ew_monthly_vol = float(ew_vol / math.sqrt(horizon)) if ew_vol > 0 else float("nan")
    cumulative = monthly.cumsum()
    ew_cumulative = ew_monthly.cumsum()
    return {
        "n_periods": int(len(r)),
        "mean_period_return": float(r.mean()),
        "period_vol": period_vol,
        "mean_monthly_return": float(monthly.mean()),
        "monthly_vol": monthly_vol,
        "annualized_sharpe": float(math.sqrt(12) * monthly.mean() / monthly_vol)
        if np.isfinite(monthly_vol) and monthly_vol > 0
        else float("nan"),
        "equal_weight_mean_period_return": float(ew.mean()) if not ew.empty else float("nan"),
        "equal_weight_period_vol": ew_vol,
        "equal_weight_mean_monthly_return": float(ew_monthly.mean()) if not ew_monthly.empty else float("nan"),
        "equal_weight_monthly_vol": ew_monthly_vol,
        "equal_weight_annualized_sharpe": float(math.sqrt(12) * ew_monthly.mean() / ew_monthly_vol)
        if np.isfinite(ew_monthly_vol) and ew_monthly_vol > 0
        else float("nan"),
        "mean_excess_vs_equal_weight": float(excess.mean()) if not excess.empty else float("nan"),
        "hit_rate_vs_equal_weight": float((excess > 0).mean()) if not excess.empty else float("nan"),
        "cumulative_log_return": float(cumulative.iloc[-1]),
        "compound_return": float(np.exp(cumulative.iloc[-1]) - 1.0),
        "max_drawdown": max_drawdown(cumulative),
        "equal_weight_compound_return": float(np.exp(ew_cumulative.iloc[-1]) - 1.0) if not ew.empty else float("nan"),
        "equal_weight_max_drawdown": max_drawdown(ew_cumulative) if not ew.empty else float("nan"),
        "mean_turnover": float(turnover.mean()) if not turnover.empty else float("nan"),
        "optimal_solver_rate": float(pd.to_numeric(returns["optimal"]).mean()),
        "mean_n_assets": float(pd.to_numeric(returns["n_assets"]).mean()),
        "mean_n_scenarios": float(pd.to_numeric(returns["n_scenarios"]).mean()),
    }


def max_drawdown(cumulative_log_return: pd.Series) -> float:
    wealth = np.exp(cumulative_log_return)
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def write_forecast_summary(run_dir: Path, metrics: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    lines = [
        f"# Cleaned Return Forecast h={diagnostics['forecast_horizon_months']}",
        "",
        f"- Target: `{diagnostics['target_definition']}`",
        f"- Rows after feature filter: {diagnostics['rows_after_feature_drop']:,}",
        f"- Eval rows: {diagnostics['rows_in_eval_window']:,}",
        f"- Countries: {len(diagnostics['countries_after_filter'])}",
        f"- Date range: {diagnostics['date_min']} to {diagnostics['date_max']}",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(markdown_table(metrics))
    run_dir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_mad_summary(out_dir: Path, metrics: pd.DataFrame) -> None:
    lines = ["# Cleaned MAD Portfolio", ""]
    if metrics.empty:
        lines.append("No MAD backtests produced metrics.")
    else:
        top = metrics.sort_values("annualized_sharpe", ascending=False).head(20)
        lines.extend(markdown_table(top))
    out_dir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_suite_summary(
    suite_dir: Path,
    combined: pd.DataFrame,
    mad_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    lines = [
        "# Cleaned Return Forecast Suite",
        "",
        f"- Cleaned segmentation model: `{config.get('cleaned_model_id')}`",
        f"- Threshold: `{config.get('cleaned_model_threshold')}`",
        f"- Target: `{config['target_definition']}`",
        f"- Test start: `{config['test_start']}`",
        "",
        "## Forecast Metrics",
        "",
    ]
    if combined.empty:
        lines.append("No forecast metrics were produced.")
    else:
        cols = [
            "forecast_horizon_months",
            "model_id",
            "n_predictions",
            "rmse",
            "directional_accuracy",
            "actual_positive_rate",
            "predicted_positive_rate",
            "strategy_annualized_sharpe",
        ]
        lines.extend(markdown_table(combined[[c for c in cols if c in combined.columns]]))
        lines.extend(["", "## Best Forecasts", ""])
        for horizon, group in combined.groupby("forecast_horizon_months"):
            best_dir = group.sort_values("directional_accuracy", ascending=False).iloc[0]
            best_rmse = group.sort_values("rmse", ascending=True).iloc[0]
            lines.append(
                f"- h={horizon}: best direction `{best_dir['model_id']}` "
                f"({best_dir['directional_accuracy']:.4f}); best RMSE `{best_rmse['model_id']}` "
                f"({best_rmse['rmse']:.6f})"
            )
    lines.extend(["", "## MAD Portfolio", ""])
    if mad_metrics.empty:
        lines.append("MAD was skipped or produced no metrics.")
    else:
        top = mad_metrics.sort_values("annualized_sharpe", ascending=False).head(10)
        lines.extend(markdown_table(top))
    suite_dir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_forecast_metrics(metrics: pd.DataFrame, run_dir: Path) -> None:
    if metrics.empty:
        return
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ordered = metrics.sort_values("directional_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(ordered["model_id"], ordered["directional_accuracy"], color="#2c7fb8")
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Direction hit rate")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_dir / "directional_accuracy_by_model.png", dpi=160)
    plt.close(fig)

    ordered = metrics.sort_values("rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(ordered["model_id"], ordered["rmse"], color="#d95f0e")
    ax.set_ylabel("RMSE")
    ax.set_title("Forecast RMSE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_dir / "rmse_by_model.png", dpi=160)
    plt.close(fig)


def plot_suite_direction(combined: pd.DataFrame, suite_dir: Path) -> None:
    if combined.empty:
        return
    plot_dir = suite_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for model_id, group in combined.groupby("model_id"):
        g = group.sort_values("forecast_horizon_months")
        ax.plot(g["forecast_horizon_months"], g["directional_accuracy"], marker="o", label=model_id)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Cleaned model direction hit rate by horizon")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "directional_accuracy_by_horizon.png", dpi=170)
    plt.close(fig)


def plot_mad_top(metrics: pd.DataFrame, out_dir: Path) -> None:
    if metrics.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    top = metrics.sort_values("annualized_sharpe", ascending=False).head(15).copy()
    labels = [
        f"{r.model_id}\\nh{r.forecast_horizon_months} {r.signal_variant} lb{r.lookback_months}"
        for r in top.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(labels, top["annualized_sharpe"], color="#238b45")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title("Top cleaned MAD portfolio configurations")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_mad_sharpe.png", dpi=170)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_Empty table._"]
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _date_min(values: pd.Series) -> str:
    return str(pd.to_datetime(values).min().date()) if len(values) else ""


def _date_max(values: pd.Series) -> str:
    return str(pd.to_datetime(values).max().date()) if len(values) else ""


class Tee:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.file = None
        self.stdout = sys.stdout

    def __enter__(self):
        self.file = self.log_path.open("w", encoding="utf-8")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self.stdout
        if self.file is not None:
            self.file.close()

    def write(self, data: str) -> int:
        self.stdout.write(data)
        if self.file is not None:
            self.file.write(data)
        return len(data)

    def flush(self) -> None:
        self.stdout.flush()
        if self.file is not None:
            self.file.flush()


if __name__ == "__main__":
    main()
