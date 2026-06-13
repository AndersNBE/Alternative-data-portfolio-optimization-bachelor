from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.return_forecasting.data import (
    FEATURE_COLUMNS,
    add_usd_market_returns,
    file_fingerprint,
    load_country_gnc,
    load_market_monthly_returns,
)
from analysis.return_forecasting.evaluation import add_relative_metrics, evaluate_predictions
from analysis.return_forecasting.run_cleaned_research_suite import (
    DEFAULT_CLEANED_BUNDLE,
    DEFAULT_MODELS,
    DEFAULT_SELECTED_MODEL_JSON,
    build_cumulative_forecast_panel,
    markdown_table,
    plot_forecast_metrics,
    plot_suite_direction,
)


WITH_GNC_FEATURES = [
    "gnc",
    "gnc_lag_1",
    "gnc_lag_2",
    "gnc_lag_3",
    "gnc_lag_4",
    "gnc_lag_5",
    "gnc_lag_6",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_5",
    "return_lag_6",
    "gnc_ma_3",
    "gnc_ma_6",
    "gnc_delta_1",
    "gnc_delta_3",
    "rolling_vol_3",
    "rolling_vol_6",
    "n_ports",
    "mean_NC",
    "mean_observations_per_port",
]

NO_GNC_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_5",
    "return_lag_6",
    "rolling_vol_3",
    "rolling_vol_6",
]

WITH_GNC_LAG_FEATURES = [
    "gnc_lag_1",
    "gnc_lag_2",
    "gnc_lag_3",
    "gnc_lag_4",
    "gnc_lag_5",
    "gnc_lag_6",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_5",
    "return_lag_6",
]

NO_GNC_LAG_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_5",
    "return_lag_6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run with-GNC vs no-GNC return model ablation.")
    parser.add_argument("--country-gnc", type=Path, default=DEFAULT_CLEANED_BUNDLE / "country_gnc.csv")
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
    parser.add_argument("--out", type=Path, default=Path("data/outputs/return_forecasting"))
    parser.add_argument("--suite-id", default="")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--horizons", default="1,2,3,4,5")
    parser.add_argument(
        "--exclude-countries",
        default="",
        help="Comma-separated country names to exclude from the forecast ablation.",
    )
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--min-train-months", type=int, default=36)
    parser.add_argument("--min-non-null-features", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_id = args.suite_id or f"gnc_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = args.out / suite_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    model_ids = [part.strip() for part in args.models.split(",") if part.strip()]
    horizons = [int(part.strip()) for part in args.horizons.split(",") if part.strip()]
    excluded_countries = [part.strip() for part in args.exclude_countries.split(",") if part.strip()]
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
        "selected_model_json": str(args.selected_model_json),
        "cleaned_model_id": cleaned_model_id,
        "cleaned_model_threshold": cleaned_model_threshold,
        "market_indices": str(args.market_indices),
        "fx_rates": str(args.fx_rates) if args.fx_rates else None,
        "return_currency": "USD" if args.fx_rates else "local",
        "target_definition": "cumulative_forward_log_return_month_t_to_t_plus_h_minus_1",
        "feature_sets": {
            "with_gnc": WITH_GNC_FEATURES,
            "no_gnc": NO_GNC_FEATURES,
        },
        "models": model_ids,
        "horizons": horizons,
        "excluded_countries": excluded_countries,
        "test_start": args.test_start,
        "min_train_months": args.min_train_months,
        "min_non_null_features": args.min_non_null_features,
        "seed": args.seed,
    }
    _write_json(suite_dir / "config.json", config)
    _write_json(
        suite_dir / "input_fingerprints.json",
        {
            "country_gnc": file_fingerprint(args.country_gnc),
            "selected_model_json": file_fingerprint(args.selected_model_json)
            if args.selected_model_json.exists()
            else None,
            "market_indices": file_fingerprint(args.market_indices),
            "fx_rates": file_fingerprint(args.fx_rates) if args.fx_rates else None,
        },
    )

    log_path = suite_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        def log(message: str) -> None:
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

        log(f"Suite: {suite_id}")
        log(f"Output: {suite_dir}")
        log(f"Cleaned GNC: {args.country_gnc}")
        log(f"Market indices: {args.market_indices}")
        log(f"Return currency: {config['return_currency']}")
        log(f"Models: {', '.join(model_ids)}")
        log(f"Horizons: {horizons}")
        if excluded_countries:
            log(f"Excluded countries: {', '.join(excluded_countries)}")
        log(f"Selected segmentation model: {cleaned_model_id} | threshold={cleaned_model_threshold}")

        country_gnc = load_country_gnc(args.country_gnc)
        market_returns = load_market_monthly_returns(args.market_indices)
        if args.fx_rates:
            market_returns = add_usd_market_returns(market_returns, args.fx_rates)
            market_returns["market_return"] = market_returns["market_return_usd"]
            market_returns = market_returns.dropna(subset=["market_return"]).copy()
        if excluded_countries:
            country_gnc = country_gnc[~country_gnc["country"].isin(excluded_countries)].copy()
            market_returns = market_returns[~market_returns["country"].isin(excluded_countries)].copy()
        log(f"Loaded GNC rows: {len(country_gnc):,}")
        log(f"Loaded monthly market return rows: {len(market_returns):,}")

        all_metrics: list[pd.DataFrame] = []
        for feature_set in ["with_gnc", "no_gnc"]:
            feature_columns = WITH_GNC_FEATURES if feature_set == "with_gnc" else NO_GNC_FEATURES
            for horizon in horizons:
                log("")
                log("=" * 80)
                log(f"Feature set={feature_set} | h={horizon}")
                run_dir = suite_dir / feature_set / f"forecast_h{horizon}"
                run_dir.mkdir(parents=True, exist_ok=True)
                panel, diagnostics = build_cumulative_forecast_panel(
                    country_gnc=country_gnc,
                    market_returns=market_returns,
                    forecast_horizon_months=horizon,
                    min_non_null_features=args.min_non_null_features,
                )
                eval_start = pd.Timestamp(args.test_start)
                diagnostics.update(
                    {
                        "feature_set": feature_set,
                        "feature_columns": feature_columns,
                        "rows_after_feature_drop": int(len(panel)),
                        "rows_in_eval_window": int((pd.to_datetime(panel["date"]) >= eval_start).sum()),
                        "countries_after_filter": sorted(panel["country"].dropna().unique()),
                        "eval_start": str(eval_start.date()),
                    }
                )
                panel.to_csv(run_dir / "panel.csv", index=False)
                _write_json(run_dir / "data_diagnostics.json", diagnostics)
                _write_json(run_dir / "config.json", {**config, "feature_set": feature_set, "forecast_horizon_months": horizon})
                log(
                    "Panel: "
                    f"{len(panel):,} rows | {panel['country'].nunique()} countries | "
                    f"eval rows={diagnostics['rows_in_eval_window']:,}"
                )

                predictions, model_meta = run_models_for_feature_set(
                    panel=panel,
                    model_ids=model_ids,
                    feature_set=feature_set,
                    feature_columns=feature_columns,
                    min_train_months=args.min_train_months,
                    seed=args.seed,
                    log=log,
                )
                predictions.to_csv(run_dir / "predictions.csv", index=False)
                _write_json(run_dir / "model_meta.json", model_meta)
                eval_predictions = predictions[pd.to_datetime(predictions["date"]) >= eval_start].copy()
                eval_predictions["forecast_horizon_months"] = horizon
                eval_predictions["feature_set"] = feature_set
                eval_predictions.to_csv(run_dir / "predictions_eval_window.csv", index=False)
                metrics = add_relative_metrics(evaluate_predictions(eval_predictions))
                metrics.insert(0, "forecast_horizon_months", horizon)
                metrics.insert(1, "feature_set", feature_set)
                metrics.to_csv(run_dir / "metrics.csv", index=False)
                plot_forecast_metrics(metrics, run_dir)
                all_metrics.append(metrics)
                log(metrics.to_string(index=False))

        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(suite_dir / "combined_metrics.csv", index=False)
        deltas = build_delta_table(combined)
        deltas.to_csv(suite_dir / "gnc_delta_metrics.csv", index=False)
        write_summary(suite_dir, combined, deltas, config)
        plot_delta(deltas, suite_dir)
        plot_suite_direction(combined[combined["feature_set"] == "with_gnc"].copy(), suite_dir / "with_gnc")
        plot_suite_direction(combined[combined["feature_set"] == "no_gnc"].copy(), suite_dir / "no_gnc")
        log("")
        log("Done.")
        log(f"All outputs: {suite_dir}")


def run_models_for_feature_set(
    *,
    panel: pd.DataFrame,
    model_ids: list[str],
    feature_set: str,
    feature_columns: list[str],
    min_train_months: int,
    seed: int,
    log,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_predictions: list[pd.DataFrame] = []
    model_meta: dict[str, Any] = {}
    for model_id in model_ids:
        log(f"Running model: {model_id}")
        config = model_config_for(model_id, feature_set, feature_columns, min_train_months, seed)
        try:
            module = importlib.import_module(f"analysis.return_forecasting.models.{model_id}")
            predictions, meta = module.run_model(panel.copy(), config)
            predictions["date"] = pd.to_datetime(predictions["date"]).dt.to_period("M").dt.to_timestamp()
            all_predictions.append(predictions)
            model_meta[model_id] = meta
            log(f"  produced {len(predictions):,} predictions")
        except Exception as exc:
            model_meta[model_id] = {"model_id": model_id, "error": str(exc), "config": config}
            log(f"  failed: {exc}")
    if not all_predictions:
        raise RuntimeError("No models produced predictions.")
    return (
        pd.concat(all_predictions, ignore_index=True)
        .sort_values(["model_id", "date", "country"])
        .reset_index(drop=True),
        model_meta,
    )


def model_config_for(
    model_id: str,
    feature_set: str,
    feature_columns: list[str],
    min_train_months: int,
    seed: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "min_train_months": min_train_months,
        "min_train_size": min_train_months,
        "min_pooled_train": min_train_months,
        "seed": seed,
        "random_state": seed,
        "available_feature_columns": FEATURE_COLUMNS,
    }
    if feature_set == "no_gnc":
        config["feature_set"] = feature_set
        if model_id == "ols_predictive":
            config["feature_columns"] = NO_GNC_LAG_FEATURES
        elif model_id in {"elastic_net_panel", "random_forest_panel"}:
            config["feature_columns"] = NO_GNC_FEATURES
        elif model_id == "distributed_lag":
            config["gnc_lags"] = 0
    return config


def build_delta_table(combined: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["forecast_horizon_months", "model_id"]
    metrics = [
        "rmse",
        "mae",
        "corr",
        "directional_accuracy",
        "strategy_annualized_sharpe",
        "predicted_positive_rate",
    ]
    with_gnc = combined[combined["feature_set"] == "with_gnc"][key_cols + metrics]
    no_gnc = combined[combined["feature_set"] == "no_gnc"][key_cols + metrics]
    merged = with_gnc.merge(no_gnc, on=key_cols, suffixes=("_with_gnc", "_no_gnc"))
    merged["rmse_improvement_from_gnc"] = merged["rmse_no_gnc"] - merged["rmse_with_gnc"]
    merged["mae_improvement_from_gnc"] = merged["mae_no_gnc"] - merged["mae_with_gnc"]
    merged["corr_delta_from_gnc"] = merged["corr_with_gnc"] - merged["corr_no_gnc"]
    merged["direction_delta_from_gnc"] = (
        merged["directional_accuracy_with_gnc"] - merged["directional_accuracy_no_gnc"]
    )
    merged["sharpe_delta_from_gnc"] = (
        merged["strategy_annualized_sharpe_with_gnc"]
        - merged["strategy_annualized_sharpe_no_gnc"]
    )
    return merged.sort_values(["forecast_horizon_months", "model_id"]).reset_index(drop=True)


def write_summary(suite_dir: Path, combined: pd.DataFrame, deltas: pd.DataFrame, config: dict[str, Any]) -> None:
    lines = [
        "# GNC Ablation",
        "",
        f"- Cleaned segmentation model: `{config.get('cleaned_model_id')}`",
        f"- Threshold: `{config.get('cleaned_model_threshold')}`",
        f"- Target: `{config['target_definition']}`",
        f"- Test start: `{config['test_start']}`",
        "",
        "## Delta Metrics",
        "",
        "Positive `rmse_improvement_from_gnc` means GNC reduced RMSE. Positive `direction_delta_from_gnc` means GNC improved directional accuracy.",
        "",
    ]
    display_cols = [
        "forecast_horizon_months",
        "model_id",
        "rmse_improvement_from_gnc",
        "corr_delta_from_gnc",
        "direction_delta_from_gnc",
        "sharpe_delta_from_gnc",
    ]
    lines.extend(markdown_table(deltas[display_cols]))
    lines.extend(["", "## Full Metrics", ""])
    full_cols = [
        "forecast_horizon_months",
        "feature_set",
        "model_id",
        "rmse",
        "corr",
        "directional_accuracy",
        "strategy_annualized_sharpe",
    ]
    lines.extend(markdown_table(combined[full_cols]))
    (suite_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_delta(deltas: pd.DataFrame, suite_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = suite_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in [
        ("rmse_improvement_from_gnc", "RMSE improvement from GNC", "rmse_improvement_from_gnc.png"),
        ("direction_delta_from_gnc", "Directional accuracy delta from GNC", "direction_delta_from_gnc.png"),
        ("corr_delta_from_gnc", "Correlation delta from GNC", "corr_delta_from_gnc.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for model_id, group in deltas.groupby("model_id"):
            g = group.sort_values("forecast_horizon_months")
            ax.plot(g["forecast_horizon_months"], g[metric], marker="o", label=model_id)
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xlabel("Forecast horizon (months)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=170)
        plt.close(fig)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
